#!/usr/bin/env julia
#
# voxelize_dynamic_contrast.jl — emit a (X, Y, Z, T) iodine-concentration
# array for the phantom by voxelizing the coronary trees at multiple time
# frames in a single CSV pass.
#
# Algorithm:
#   1. Load phantom (UInt8 → UInt16). Identify myocardium voxels (labels 15–18).
#   2. Allocate accumulators:
#        f_blood[X,Y,Z]             — time-invariant blood volume fraction
#        f_iodine_w[X,Y,Z, 1..T]    — sum over segs (sub-voxel frac × C_seg(t))
#   3. For each tree:
#        - Parse *_segments.csv (Float32 endpoint matrix)
#        - Load {name}_arrival_time.f32 (per-segment hemo-derived bolus arrival)
#        - Threaded sub-voxel MC: for each segment with finite arrival, compute
#          C[1..T] once from gamma-variate + dispersion, then push (vidx, frac,
#          C[1..T]) into per-thread flat buffers.
#        - Single-threaded merge: write into shared f_blood / f_iodine_w.
#   4. C_iodine[X,Y,Z,t] = f_iodine_w[X,Y,Z,t] / max(f_blood, eps)
#      → iodine mg/mL in the **blood phase** of each voxel (i.e., what the
#        blood at that voxel/time would have). Voxels with f_blood=0 stay 0.
#   5. Write per-frame Float32 .raw files + a manifest TOML.
#
# Output (in OUTPUT_DIR):
#   dynamic_f_blood.f32                        Float32 1600×1400×500
#   dynamic_c_iodine_t{ti:03d}.f32             Float32 1600×1400×500 per frame
#   dynamic_manifest.toml                      shape, frame times, units, sources
#
# Caveats:
#   - Only myocardium voxels get blood-volume contributions (matches
#     apply_contrast_at_peak.jl). XCAT chamber blood (labels 19-28) is NOT
#     time-resolved here — chambers' bolus is a separate model.
#   - 4D output is **large**: 1.12 G voxels × T × 4 B. T=20 → 90 GB on disk.
#     Frames are written separately so consumers can lazy-load / mmap.
#
# Usage:
#   julia --project=. --threads=auto scripts/voxelize_dynamic_contrast.jl \
#         TREE_DIR  PHANTOM_RAW_IN  PEAK_IODINE_DIR  CONFIG_TOML  OUTPUT_DIR \
#         [N_FRAMES]  [T_START_S]  [T_END_S]  [N_SUB]

using LinearAlgebra
using Base.Threads
using Printf
using TOML

# ── Phantom geometry (vmale50, 0.02 cm isotropic) ────────────────────────────
const PHANTOM_DIMS = (1600, 1400, 500)
const VOXEL_CM = 0.02
const NRB_TO_PHANTOM_OFFSET = (2.1443, -9.5553, -20.0068)
const XCAT_ORIGIN_CM = (2.846980, -9.773884, -20.600891)

const MYO_LABELS = (UInt16(15), UInt16(16), UInt16(17), UInt16(18))
const DEFAULT_N_SUB = 5
const MIN_RENDER_DIAMETER_CM = 0.0

@inline is_myo(v::UInt16) =
    v == MYO_LABELS[1] || v == MYO_LABELS[2] ||
    v == MYO_LABELS[3] || v == MYO_LABELS[4]

# ── Phantom load (UInt8 → UInt16) ────────────────────────────────────────────
function load_phantom_raw_u16(path::String)
    nx, ny, nz = PHANTOM_DIMS
    expected_u8 = nx * ny * nz
    actual = filesize(path)
    actual == expected_u8 ||
        error("Phantom raw must be UInt8 ($expected_u8 B); got $actual B at $path")
    src = Array{UInt8}(undef, nx, ny, nz)
    t0 = time()
    read!(path, src)
    dst = Array{UInt16}(undef, nx, ny, nz)
    @threads :static for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            dst[i, j, k] = UInt16(src[i, j, k])
        end
    end
    @printf("[load] %s: %d×%d×%d UInt8 → UInt16 (%.1fs)\n",
            basename(path), nx, ny, nz, time()-t0); flush(stdout)
    dst
end

# ── Parse *_segments.csv → Float32 [7 × n_segs] (NRB→phantom shift) ─────────
function parse_segments_csv(csv_path::String)
    ox, oy, oz = Float32.(NRB_TO_PHANTOM_OFFSET)
    rmin = Float32(MIN_RENDER_DIAMETER_CM / 2)

    t0 = time()
    n_newlines = 0
    open(csv_path, "r") do io
        buf = Vector{UInt8}(undef, 64 * 1024 * 1024)
        while !eof(io)
            nread = readbytes!(io, buf)
            @inbounds for k in 1:nread
                buf[k] == 0x0A && (n_newlines += 1)
            end
        end
    end
    n_segs = n_newlines - 1
    @printf("[csv] %s: %d rows (%.1fs)\n", basename(csv_path), n_segs, time()-t0); flush(stdout)

    A = Matrix{Float32}(undef, 7, n_segs)
    t0 = time()
    open(csv_path, "r") do io
        readline(io)  # header
        i = 0
        while !eof(io)
            line = readline(io)
            isempty(line) && continue
            i += 1
            c1  = findfirst(',', line);              c2  = findnext(',', line, c1+1)
            c3  = findnext(',', line, c2+1);          c4  = findnext(',', line, c3+1)
            c5  = findnext(',', line, c4+1);          c6  = findnext(',', line, c5+1)
            c7  = findnext(',', line, c6+1);          c8  = findnext(',', line, c7+1)
            c9  = findnext(',', line, c8+1);          c10 = findnext(',', line, c9+1)
            c11 = findnext(',', line, c10+1);         c12 = findnext(',', line, c11+1)
            c13 = findnext(',', line, c12+1);         c14 = findnext(',', line, c13+1)
            x1 = parse(Float32, SubString(line, c3+1,  c4-1))
            y1 = parse(Float32, SubString(line, c4+1,  c5-1))
            z1 = parse(Float32, SubString(line, c5+1,  c6-1))
            x2 = parse(Float32, SubString(line, c6+1,  c7-1))
            y2 = parse(Float32, SubString(line, c7+1,  c8-1))
            z2 = parse(Float32, SubString(line, c8+1,  c9-1))
            d_um = parse(Float32, SubString(line, c13+1, c14-1))
            @inbounds A[1,i] = x1 + ox;  @inbounds A[2,i] = y1 + oy;  @inbounds A[3,i] = z1 + oz
            @inbounds A[4,i] = x2 + ox;  @inbounds A[5,i] = y2 + oy;  @inbounds A[6,i] = z2 + oz
            @inbounds A[7,i] = max((d_um * Float32(1e-4)) / Float32(2), rmin)
        end
    end
    @printf("[csv] %s: parsed in %.1fs (%.2f Mrows/s)\n",
            basename(csv_path), time()-t0, n_segs / (time()-t0) / 1e6); flush(stdout)
    A
end

function load_arrival_f32(path::String, n_segs::Int)
    expected = n_segs * 4
    actual = filesize(path)
    actual == expected ||
        error("arrival_time mismatch at $path: $actual B vs $expected B (n_segs=$n_segs)")
    v = Vector{Float32}(undef, n_segs)
    read!(path, v)
    v
end

# ── Gamma-variate bolus with dispersion ──────────────────────────────────────
struct BolusParams
    amplitude::Float32
    t0::Float32
    tmax::Float32
    alpha::Float32
    t_dispersion_s::Float32
end

@inline function c_at_time(t::Float32, arrival::Float32, bp::BolusParams)
    !isfinite(arrival) && return 0f0
    t_shifted = t - arrival
    t_shifted <= 0f0 && return 0f0
    disp = sqrt(1f0 + arrival / bp.t_dispersion_s)
    t_input = bp.t0 + (t_shifted - bp.t0) / disp
    t_input <= bp.t0 && return 0f0
    tp = (t_input - bp.t0) / (bp.tmax - bp.t0)
    tp <= 0f0 && return 0f0
    c = bp.amplitude * tp^bp.alpha * exp(bp.alpha * (1f0 - tp)) / disp
    return max(c, 0f0)
end

# Per-segment C[1..T] — assumes pre-allocated `out` length T.
@inline function fill_c_series!(out::AbstractVector{Float32},
                                arrival::Float32, times::Vector{Float32},
                                bp::BolusParams)
    @inbounds for ti in 1:length(times)
        out[ti] = c_at_time(times[ti], arrival, bp)
    end
    return out
end

# ── Per-thread flat buffer of (vidx, frac, C[1..T]) ──────────────────────────
mutable struct DynBuf
    idx::Vector{Int32}              # voxel linear index
    frac::Vector{Float32}           # sub-voxel inside-fraction
    C::Vector{Float32}              # flat: T values per entry, stride T
end
DynBuf() = DynBuf(Int32[], Float32[], Float32[])

# Rasterize a tree's segments into f_blood + f_iodine_t (cross-product with C).
function rasterize_capsules_dynamic!(f_blood::Array{Float32,3},
                                     f_iodine_t::Array{Float32,4},
                                     phantom::Array{UInt16,3},
                                     A::Matrix{Float32},
                                     arrival::Vector{Float32},
                                     times::Vector{Float32},
                                     bp::BolusParams;
                                     n_sub::Int = DEFAULT_N_SUB)
    nx, ny, nz = size(f_blood)
    T = length(times)
    n_segs = size(A, 2)
    n_segs == length(arrival) ||
        error("arrival length mismatch: $(length(arrival)) vs n_segs=$n_segs")
    voxel_cm = Float32(VOXEL_CM)
    inv_voxel_cm = Float32(1 / VOXEL_CM)
    inv_n_sub_total = Float32(1 / n_sub^3)
    sub_offsets = Float32[((2 * i - 1 - n_sub) / Float32(2 * n_sub)) * voxel_cm for i in 1:n_sub]

    nt = nthreads()
    bufs = [DynBuf() for _ in 1:nt]
    # Per-thread reusable C[1..T] scratch (avoids 357M small allocs).
    c_scratch = [Vector{Float32}(undef, T) for _ in 1:nt]
    # Pre-size per-thread buffers (rough: 3 voxel-incs per segment on average,
    # 4 bytes idx + 4 bytes frac + 4T bytes C)
    for t in 1:nt
        h = max(1024, div(3 * n_segs, nt))
        sizehint!(bufs[t].idx, h)
        sizehint!(bufs[t].frac, h)
        sizehint!(bufs[t].C, h * T)
    end

    t0 = time()
    @threads :static for s in 1:n_segs
        tid = threadid()
        buf = bufs[tid]
        C_local = c_scratch[tid]
        @inbounds begin
            # Always rasterize: f_blood is ANATOMICAL (independent of bolus
            # arrival), so segments must contribute even when their bolus
            # hasn't arrived in [0, t_end]. The C[1..T] vector is just set to
            # zero for those segments — they show up as blood-without-iodine
            # in the voxel grid, which is physically correct.
            a = arrival[s]
            if isfinite(a)
                fill_c_series!(C_local, a, times, bp)
            else
                @inbounds for ti in 1:T
                    C_local[ti] = 0f0
                end
            end

            ax = A[1,s]; ay = A[2,s]; az = A[3,s]
            bx = A[4,s]; by = A[5,s]; bz = A[6,s]
            r_cm = A[7,s]
            r2 = r_cm * r_cm

            lo_x = min(ax, bx) - r_cm;  hi_x = max(ax, bx) + r_cm
            lo_y = min(ay, by) - r_cm;  hi_y = max(ay, by) + r_cm
            lo_z = min(az, bz) - r_cm;  hi_z = max(az, bz) + r_cm

            i0 = max(1,  floor(Int, lo_x * inv_voxel_cm) + 1)
            j0 = max(1,  floor(Int, lo_y * inv_voxel_cm) + 1)
            k0 = max(1,  floor(Int, lo_z * inv_voxel_cm) + 1)
            i1 = min(nx, ceil(Int,  hi_x * inv_voxel_cm) + 1)
            j1 = min(ny, ceil(Int,  hi_y * inv_voxel_cm) + 1)
            k1 = min(nz, ceil(Int,  hi_z * inv_voxel_cm) + 1)
            (i0 > i1 || j0 > j1 || k0 > k1) && continue

            abx = bx - ax;  aby = by - ay;  abz = bz - az
            ab_len2 = abx*abx + aby*aby + abz*abz
            degenerate = ab_len2 <= 1f-24

            for kk in k0:k1
                cz = (Float32(kk) - 0.5f0) * voxel_cm
                for jj in j0:j1
                    cy = (Float32(jj) - 0.5f0) * voxel_cm
                    for ii in i0:i1
                        v = phantom[ii, jj, kk]
                        is_myo(v) || continue
                        cx = (Float32(ii) - 0.5f0) * voxel_cm
                        inside = 0
                        for k_sub in 1:n_sub
                            spz = cz + sub_offsets[k_sub]
                            apz = spz - az
                            for j_sub in 1:n_sub
                                spy = cy + sub_offsets[j_sub]
                                apy = spy - ay
                                for i_sub in 1:n_sub
                                    spx = cx + sub_offsets[i_sub]
                                    apx = spx - ax
                                    dist2 = if degenerate
                                        apx*apx + apy*apy + apz*apz
                                    else
                                        tparam = clamp((apx*abx + apy*aby + apz*abz) / ab_len2, 0f0, 1f0)
                                        dx = apx - tparam*abx;  dy = apy - tparam*aby;  dz = apz - tparam*abz
                                        dx*dx + dy*dy + dz*dz
                                    end
                                    dist2 <= r2 && (inside += 1)
                                end
                            end
                        end
                        if inside > 0
                            frac = Float32(inside) * inv_n_sub_total
                            lidx = Int32(((kk - 1) * ny + (jj - 1)) * nx + ii)
                            push!(buf.idx, lidx)
                            push!(buf.frac, frac)
                            @inbounds for ti in 1:T
                                push!(buf.C, C_local[ti])
                            end
                        end
                    end
                end
            end
        end
    end
    t_raster = time() - t0

    # Single-threaded merge: walk per-thread flat buffers, accumulate into
    # shared f_blood (linear idx) + f_iodine_t (4D linear idx = lidx + (ti-1)*N3).
    t0 = time()
    N3 = nx * ny * nz
    n_writes = 0
    for tid in 1:nt
        buf = bufs[tid]
        bi = buf.idx; bf = buf.frac; bC = buf.C
        n = length(bi)
        @inbounds for k in 1:n
            lidx = bi[k]
            frac = bf[k]
            f_blood[lidx] += frac
            base = (k - 1) * T
            for ti in 1:T
                f_iodine_t[lidx + (ti - 1) * N3] += frac * bC[base + ti]
            end
        end
        n_writes += n
        empty!(buf.idx);  empty!(buf.frac);  empty!(buf.C)
    end
    t_merge = time() - t0
    @printf("[raster] %d segs → %d voxel-incs × %d frames (%.1fs raster + %.1fs merge, %.2f Mseg/s)\n",
            n_segs, n_writes, T, t_raster, t_merge,
            n_segs / max(t_raster + t_merge, 1e-6) / 1e6)
    flush(stdout)
    n_writes
end

# ── Output writers ───────────────────────────────────────────────────────────
function write_float32_raw(path::String, arr::AbstractArray{Float32})
    t0 = time()
    open(path, "w") do io
        write(io, arr)
    end
    sz = filesize(path)
    @printf("[write] %s  %.1f MB  (%.1fs)\n", basename(path), sz/1024^2, time()-t0)
    flush(stdout)
end

function write_manifest(out_dir::String, n_frames::Int, times::Vector{Float32},
                        bp::BolusParams, peak_dir::String, csvs::Vector{String},
                        seg_counts::Vector{Int}, n_writes::Int, n_sub::Int,
                        c_max_over_frames::Vector{Float32},
                        f_blood_max::Float32, f_blood_n_nonzero::Int)
    path = joinpath(out_dir, "dynamic_manifest.toml")
    nx, ny, nz = PHANTOM_DIMS
    open(path, "w") do io
        println(io, "# dynamic_manifest.toml — output of voxelize_dynamic_contrast.jl")
        println(io, "# 4D dynamic iodine concentration: phantom (X,Y,Z) × time (T)")
        println(io)
        println(io, "[phantom]")
        println(io, "dims = [$nx, $ny, $nz]")
        println(io, "voxel_size_cm = [$VOXEL_CM, $VOXEL_CM, $VOXEL_CM]")
        println(io, "voxel_ordering = \"x-fastest\"")
        println(io, "dtype = \"Float32\"")
        println(io, "byte_order = \"little-endian\"")
        println(io, "xcat_origin_cm = [$(XCAT_ORIGIN_CM[1]), $(XCAT_ORIGIN_CM[2]), $(XCAT_ORIGIN_CM[3])]")
        println(io, "nrb_to_phantom_offset_cm = [$(NRB_TO_PHANTOM_OFFSET[1]), $(NRB_TO_PHANTOM_OFFSET[2]), $(NRB_TO_PHANTOM_OFFSET[3])]")
        println(io, "writable_base_labels = [15, 16, 17, 18]  # only myocardium voxels carry coronary blood")
        println(io)
        println(io, "[frames]")
        println(io, "n_frames = $n_frames")
        println(io, "times_s = [", join([@sprintf("%.4f", t) for t in times], ", "), "]")
        println(io, "f_blood_file = \"dynamic_f_blood.f32\"  # time-invariant blood VF")
        println(io, "c_iodine_file_pattern = \"dynamic_c_iodine_t{ti:03d}.f32\"  # iodine mg/mL in blood phase")
        println(io, "units = \"mg_iodine_per_mL_blood\"")
        println(io, "value_at_voxel = \"f_iodine_w / max(f_blood, eps)\"")
        println(io, "c_iodine_max_per_frame = [", join([@sprintf("%.6f", c) for c in c_max_over_frames], ", "), "]")
        println(io)
        println(io, "[blood_fraction]")
        println(io, "# f_blood file is CLIPPED to [0, 1] for downstream physics.")
        println(io, "# f_blood_max_raw_pre_clip reflects multi-segment overlap in the accumulator —")
        println(io, "# the C_iodine division uses the raw (unclipped) accumulator internally so")
        println(io, "# the blood-phase concentration is overlap-invariant.")
        @printf(io, "f_blood_max_raw_pre_clip = %.6f\n", f_blood_max)
        println(io, "f_blood_max_in_file = 1.0")
        println(io, "n_voxels_with_blood = $f_blood_n_nonzero")
        println(io)
        println(io, "[bolus]")
        @printf(io, "amplitude_mg_per_mL = %.4f\n", bp.amplitude)
        @printf(io, "t0_s = %.4f\n", bp.t0)
        @printf(io, "tmax_s = %.4f\n", bp.tmax)
        @printf(io, "alpha = %.4f\n", bp.alpha)
        @printf(io, "t_dispersion_s = %.4f\n", bp.t_dispersion_s)
        println(io)
        println(io, "[embed]")
        println(io, "method = \"sub_voxel_monte_carlo_dynamic\"")
        println(io, "n_sub_per_dim = $n_sub")
        println(io, "n_sub_total = $(n_sub^3)")
        println(io, "voxel_increment_writes = $n_writes")
        println(io, "peak_iodine_dir = \"$peak_dir\"")
        println(io, "tree_csv_files = [", join(["\"$(basename(f))\"" for f in csvs], ", "), "]")
        println(io, "tree_segment_counts = [", join(string.(seg_counts), ", "), "]")
    end
    @printf("[write] %s\n", path); flush(stdout)
end

# ─────────────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 5
        println("Usage: julia --project=. --threads=auto scripts/voxelize_dynamic_contrast.jl \\")
        println("              TREE_DIR  PHANTOM_RAW_IN  PEAK_IODINE_DIR  CONFIG_TOML  OUTPUT_DIR \\")
        println("              [N_FRAMES=20]  [T_START_S=0.0]  [T_END_S=15.0]  [N_SUB=5]")
        exit(1)
    end
    tree_dir   = ARGS[1]
    raw_in     = ARGS[2]
    peak_dir   = ARGS[3]
    config_tl  = ARGS[4]
    out_dir    = ARGS[5]
    n_frames   = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 20
    t_start_s  = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : 0.0
    t_end_s    = length(ARGS) >= 8 ? parse(Float64, ARGS[8]) : 15.0
    n_sub      = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : DEFAULT_N_SUB
    isdir(tree_dir)   || error("tree_dir not found: $tree_dir")
    isfile(raw_in)    || error("phantom raw not found: $raw_in")
    isdir(peak_dir)   || error("peak_iodine_dir not found: $peak_dir")
    isfile(config_tl) || error("config toml not found: $config_tl")
    isdir(out_dir)    || mkpath(out_dir)

    cfg = TOML.parsefile(config_tl)
    bp = BolusParams(
        Float32(get(cfg, "contrast_amplitude", 5.0)),
        Float32(get(cfg, "contrast_t0", 0.5)),
        Float32(get(cfg, "contrast_tmax", 4.0)),
        Float32(get(cfg, "contrast_alpha", 3.0)),
        Float32(get(cfg, "contrast_t_dispersion_s", 3.0)),
    )
    times = Float32.(collect(LinRange(t_start_s, t_end_s, n_frames)))

    @printf("threads = %d\n", nthreads())
    println("tree_dir         = $tree_dir")
    println("phantom_raw_in   = $raw_in")
    println("peak_iodine_dir  = $peak_dir")
    println("config_toml      = $config_tl")
    println("output_dir       = $out_dir")
    @printf("n_frames         = %d  (t_start=%.2fs  t_end=%.2fs  dt=%.4fs)\n",
            n_frames, t_start_s, t_end_s, n_frames > 1 ? (t_end_s - t_start_s) / (n_frames - 1) : 0.0)
    @printf("bolus            = amp=%.2f mg/mL  t0=%.2f  tmax=%.2f  α=%.2f  t_disp=%.2fs\n",
            bp.amplitude, bp.t0, bp.tmax, bp.alpha, bp.t_dispersion_s)
    @printf("n_sub_per_dim    = %d (%d samples/voxel)\n", n_sub, n_sub^3)
    flush(stdout)

    csvs = sort(filter(f -> endswith(f, "_segments.csv"), readdir(tree_dir; join=true)))
    isempty(csvs) && error("No *_segments.csv in $tree_dir")
    println("Trees:");  for f in csvs; println("  $(basename(f))"); end;  flush(stdout)

    # 1) Load phantom (UInt8 → UInt16)
    phantom = load_phantom_raw_u16(raw_in)
    nx, ny, nz = size(phantom)

    # 2) Allocate accumulators. Verify RAM headroom before allocating f_iodine_t.
    bytes_f_blood = nx * ny * nz * 4
    bytes_f_iod   = nx * ny * nz * n_frames * 4
    @printf("[alloc] f_blood = %.2f GB; f_iodine_t = %.2f GB (4D %d×%d×%d×%d Float32)\n",
            bytes_f_blood / 1024^3, bytes_f_iod / 1024^3, nx, ny, nz, n_frames)
    flush(stdout)
    f_blood    = zeros(Float32, nx, ny, nz)
    f_iodine_t = zeros(Float32, nx, ny, nz, n_frames)

    # 3) Per-tree: parse CSV, load arrival_time.f32, rasterize
    n_writes_total = 0
    seg_counts = Int[]
    for csv_path in csvs
        @printf("\n=== %s ===\n", basename(csv_path));  flush(stdout)
        name = lowercase(replace(basename(csv_path), "_segments.csv" => ""))
        arr_path = joinpath(peak_dir, "$(name)_arrival_time.f32")
        isfile(arr_path) || error("missing arrival_time for $name at $arr_path (run extract_peak_iodine.jl first)")

        A = parse_segments_csv(csv_path)
        nsegs = size(A, 2)
        push!(seg_counts, nsegs)
        arrival = load_arrival_f32(arr_path, nsegs)
        n_reach = count(isfinite, arrival)
        @printf("[arrival] %s: %d/%d reachable  min=%.2fs  max=%.2fs (finite)\n",
                name, n_reach, nsegs,
                n_reach > 0 ? minimum(filter(isfinite, arrival)) : NaN,
                n_reach > 0 ? maximum(filter(isfinite, arrival)) : NaN)
        flush(stdout)

        n_writes_total += rasterize_capsules_dynamic!(f_blood, f_iodine_t, phantom,
                                                     A, arrival, times, bp; n_sub=n_sub)
        A = nothing
        arrival = nothing
        GC.gc()
    end

    # 4) Free phantom (no longer needed)
    phantom = nothing
    GC.gc()

    # 5) f_blood: report raw (overlap-inflated) stats, then write CLIPPED [0,1].
    # Multi-segment overlap can push the raw accumulator past 1.0 (many capillaries
    # packing into one 200 μm voxel). The clipped version is the physical blood
    # volume fraction (≤ 1). The C_iodine division below still uses the raw
    # value, so overlap-weighted averaging stays correct.
    raw_max = maximum(f_blood)
    raw_nnz = count(>(0f0), f_blood)
    raw_over1 = count(>(1f0), f_blood)
    @printf("\n[summary] f_blood (raw): max=%.4f  nnz=%d  voxels>1=%d (overlap-inflated)\n",
            raw_max, raw_nnz, raw_over1)
    f_blood_max = raw_max
    f_blood_n_nonzero = raw_nnz
    # Clip f_blood for output (physical volume fraction). Write a copy so the
    # raw accumulator stays available for the unbiased C_iodine divide below.
    f_blood_clipped = similar(f_blood)
    @threads :static for k in 1:nz
        @inbounds for j in 1:ny, i in 1:nx
            f_blood_clipped[i, j, k] = min(f_blood[i, j, k], 1f0)
        end
    end
    write_float32_raw(joinpath(out_dir, "dynamic_f_blood.f32"), f_blood_clipped)
    f_blood_clipped = nothing
    GC.gc()

    # 6) Per-frame C_iodine = f_iodine_w[:,:,:,ti] / max(f_blood, eps)
    c_max_per_frame = Float32[]
    eps32 = eps(Float32)
    for ti in 1:n_frames
        t0 = time()
        # Compute C in place using f_iodine_t slice; write directly.
        C_frame = Array{Float32,3}(undef, nx, ny, nz)
        @threads :static for k in 1:nz
            @inbounds for j in 1:ny, i in 1:nx
                fb = f_blood[i, j, k]
                C_frame[i, j, k] = fb > 0f0 ? f_iodine_t[i, j, k, ti] / fb : 0f0
            end
        end
        cmax = maximum(C_frame)
        push!(c_max_per_frame, cmax)
        path = joinpath(out_dir, @sprintf("dynamic_c_iodine_t%03d.f32", ti))
        write_float32_raw(path, C_frame)
        @printf("[frame %3d/%d] t=%.3fs  C_max=%.4f mg/mL  %.1fs\n",
                ti, n_frames, times[ti], cmax, time() - t0)
        flush(stdout)
        C_frame = nothing
    end

    # 7) Manifest
    println()
    write_manifest(out_dir, n_frames, times, bp, peak_dir,
                   csvs, seg_counts, n_writes_total, n_sub,
                   c_max_per_frame, f_blood_max, f_blood_n_nonzero)
    println("\ndone.")
end

main()
