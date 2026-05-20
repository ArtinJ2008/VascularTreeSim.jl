#!/usr/bin/env julia
#
# Stream coronary tree CSVs (from run_coronary_growth.jl) and rasterize them
# into a UInt16 XCAT phantom mask with **100-bin sub-voxel Monte-Carlo
# volume-fraction accounting**.
#
# This is the "no-contrast" step.  Each myocardium voxel ends up labeled by
# its blood volume fraction at 1 % resolution; downstream the loader
# turns those labels into proper mass-weighted (blood + muscle) XA.Materials.
# A later step (apply_contrast_at_peak.jl, to be written) consumes both
# this phantom and a FlowContrastSim iodine field to produce a cross-product
# (blood %, iodine %) UInt16 phantom for contrast-enhanced CT simulation.
#
# Pipeline:
#   1. Load source phantom raw (UInt8) and promote to UInt16 (preserves all
#      XCAT labels in 0..255; leaves headroom for new labels 256+).
#   2. For each *_segments.csv: parse into packed Float32 matrix
#      (Nx7: ax,ay,az,bx,by,bz,r_cm with NRB→phantom shift applied), then
#      rasterize each capsule with N×N×N sub-voxel samples per voxel.
#      Each sample inside the capsule contributes 1/N³ to a Float32
#      f_blood accumulator at that voxel.  Only myocardium voxels
#      (labels 15-18) participate.
#   3. Quantize f_blood per voxel into 100 bins (1 % each):
#        bin == 0  (round(f*100) == 0): keep original myo label
#        bin ∈ 1..100                  : label = 255 + bin  (= 256..355)
#      Bin labels map at load time to mass-weighted (blood, muscle) Materials.
#   4. Write UInt16 raw + phantom_manifest.toml + phantom_info.txt.
#
# Usage:
#   julia --project=. --threads=auto scripts/voxelize_csv_into_phantom.jl \
#         TREE_DIR  PHANTOM_RAW_IN  OUTPUT_DIR  [N_SUB]
#
#   TREE_DIR        directory holding *_segments.csv
#   PHANTOM_RAW_IN  copy of the XCAT vmale50 1600×1400×500 8-bit raw
#   OUTPUT_DIR      where the modified UInt16 raw + manifest + info are written
#   N_SUB           optional; sub-voxel samples per dim (default 5 → 125 samples
#                   for ~0.8 % f_blood precision, sub-bin at 1 %; raise to 8 or
#                   10 for finer at proportional cost)

using LinearAlgebra
using Base.Threads
using Printf
using TOML

# ── Phantom geometry (vmale50, 0.02 cm isotropic) ────────────────────────────
const PHANTOM_DIMS = (1600, 1400, 500)
const VOXEL_CM = 0.02
const NRB_TO_PHANTOM_OFFSET = (2.1443, -9.5553, -20.0068)
const XCAT_ORIGIN_CM = (2.846980, -9.773884, -20.600891)

# ── Label scheme (UInt16 mask) ───────────────────────────────────────────────
const N_BINS = 100
const BIN_LABEL_BASE = UInt16(256)        # bin b ∈ 1..N_BINS  →  label 255 + b
                                          # so 256 = ~1% blood, 355 = ~100% blood.
const MYO_LABELS = (UInt16(15), UInt16(16), UInt16(17), UInt16(18))

# Output raw size: 2 bytes / voxel for UInt16
const N_VOXELS = PHANTOM_DIMS[1] * PHANTOM_DIMS[2] * PHANTOM_DIMS[3]
const OUT_RAW_BASENAME = "vmale50_with_grown_coronaries_no_contrast_u16.raw"

# ── Render tuning ────────────────────────────────────────────────────────────
const DEFAULT_N_SUB = 5      # n_sub=5 → 125 sub-samples per voxel → ~0.8% precision
const MIN_RENDER_DIAMETER_CM = 0.0   # use the segment's true diameter; sub-voxel MC handles tiny segs.

# ─────────────────────────────────────────────────────────────────────────────

@inline is_myo(v::UInt16) =
    v == MYO_LABELS[1] || v == MYO_LABELS[2] ||
    v == MYO_LABELS[3] || v == MYO_LABELS[4]

# Load the UInt8 source raw and promote to a UInt16 array (preserves XCAT
# 0..255 labels unchanged; UInt16 headroom is for the new 256+ bin labels).
function load_phantom_raw_u16(path::String)
    nx, ny, nz = PHANTOM_DIMS
    expected_u8 = nx * ny * nz
    actual = filesize(path)
    actual == expected_u8 ||
        error("Source phantom must be UInt8 ($expected_u8 B); got $actual B at $path")
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
            basename(path), nx, ny, nz, time()-t0)
    flush(stdout)
    dst
end

# Label histogram over the full UInt16 range (~65 k slots; ~520 KB).
function label_histogram(phantom::Array{UInt16,3})
    nx, ny, nz = size(phantom)
    counts = [zeros(Int64, 65536) for _ in 1:nthreads()]
    @threads :static for k in 1:nz
        c = counts[threadid()]
        @inbounds for j in 1:ny, i in 1:nx
            c[Int(phantom[i,j,k]) + 1] += 1
        end
    end
    hist = zeros(Int64, 65536)
    for c in counts;  hist .+= c;  end
    hist
end

# Parse a *_segments.csv → 7×Nseg Float32 matrix [ax;ay;az;bx;by;bz;r_cm]
# (NRB→phantom shift applied, radius in cm).
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
    n_segs > 0 || error("CSV $csv_path has no data rows")
    @printf("[csv] %s: counted %d data rows (%.1fs)\n",
            basename(csv_path), n_segs, time()-t0)
    flush(stdout)

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
            @inbounds A[1,i] = x1 + ox
            @inbounds A[2,i] = y1 + oy
            @inbounds A[3,i] = z1 + oz
            @inbounds A[4,i] = x2 + ox
            @inbounds A[5,i] = y2 + oy
            @inbounds A[6,i] = z2 + oz
            @inbounds A[7,i] = max((d_um * Float32(1e-4)) / Float32(2), rmin)
        end
    end
    @printf("[csv] %s: parsed in %.1fs (%.2f Mrows/s)\n",
            basename(csv_path), time()-t0, n_segs / (time()-t0) / 1e6)
    flush(stdout)
    A
end

# Sub-voxel Monte-Carlo rasterizer with volume-fraction accumulation.
# Each thread accumulates (linear_idx, frac) into thread-local buffers; merged
# single-threaded into the shared `f_blood` array at the end.
function rasterize_capsules_vf!(f_blood::Array{Float32,3},
                                phantom::Array{UInt16,3},
                                A::Matrix{Float32};
                                n_sub::Int = DEFAULT_N_SUB)
    nx, ny, nz = size(f_blood)
    @assert size(phantom) == size(f_blood) "phantom/f_blood size mismatch"
    n_segs = size(A, 2)
    voxel_cm = Float32(VOXEL_CM)
    inv_voxel_cm = Float32(1 / VOXEL_CM)
    n_sub_total = Float32(n_sub^3)
    inv_n_sub_total = Float32(1 / n_sub_total)
    sub_offsets = Float32[((2 * i - 1 - n_sub) / Float32(2 * n_sub)) * voxel_cm
                          for i in 1:n_sub]

    nt = nthreads()
    buf_idx = [Vector{Int32}() for _ in 1:nt]
    buf_val = [Vector{Float32}() for _ in 1:nt]
    for t in 1:nt
        sizehint!(buf_idx[t], max(1024, div(3 * n_segs, nt)))
        sizehint!(buf_val[t], max(1024, div(3 * n_segs, nt)))
    end

    t0 = time()
    @threads :static for s in 1:n_segs
        tid = threadid()
        bi = buf_idx[tid]
        bv = buf_val[tid]
        @inbounds begin
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
                                        t = clamp((apx*abx + apy*aby + apz*abz) / ab_len2, 0f0, 1f0)
                                        dx = apx - t*abx;  dy = apy - t*aby;  dz = apz - t*abz
                                        dx*dx + dy*dy + dz*dz
                                    end
                                    dist2 <= r2 && (inside += 1)
                                end
                            end
                        end
                        if inside > 0
                            frac = Float32(inside) * inv_n_sub_total
                            lidx = Int32(((kk - 1) * ny + (jj - 1)) * nx + ii)
                            push!(bi, lidx)
                            push!(bv, frac)
                        end
                    end
                end
            end
        end
    end
    t_raster = time() - t0

    t0 = time()
    n_writes = 0
    for t in 1:nt
        bi = buf_idx[t];  bv = buf_val[t]
        @inbounds for k in eachindex(bi)
            f_blood[bi[k]] += bv[k]
        end
        n_writes += length(bi)
        empty!(bi);  empty!(bv)
    end
    t_merge = time() - t0

    @printf("[raster-vf] %d segs → %d voxel-increments (%.1fs raster + %.1fs merge, %.2f Mseg/s)\n",
            n_segs, n_writes, t_raster, t_merge, n_segs / (t_raster + t_merge) / 1e6)
    flush(stdout)
    n_writes
end

# Quantize f_blood per voxel → label.
#   bin = round(f_blood * N_BINS)
#   bin == 0     → keep original myo label  (sub-0.5% f_blood, below CT resolution)
#   bin ∈ 1..100 → label = 255 + bin        (= 256..355)
# Only myocardium voxels (15-18) eligible.  Clamps bin ≥ N_BINS to N_BINS.
function quantize_and_apply!(phantom::Array{UInt16,3}, f_blood::Array{Float32,3})
    nx, ny, nz = size(phantom)
    bin_counts = zeros(Int64, N_BINS + 1)   # indices 1..N_BINS+1 ↔ bins 0..N_BINS
    per_thread = [zeros(Int64, N_BINS + 1) for _ in 1:nthreads()]
    t0 = time()
    @threads :static for k in 1:nz
        local_c = per_thread[threadid()]
        @inbounds for j in 1:ny, i in 1:nx
            v = phantom[i, j, k]
            is_myo(v) || continue
            f = f_blood[i, j, k]
            f <= 0f0 && continue
            f_clipped = min(f, 1f0)
            bin = round(Int, f_clipped * N_BINS)
            bin > N_BINS && (bin = N_BINS)
            local_c[bin + 1] += 1
            if bin == 0
                continue
            else
                phantom[i, j, k] = UInt16(255) + UInt16(bin)
            end
        end
    end
    for c in per_thread;  bin_counts .+= c;  end
    @printf("[quant] f_blood → labels (%.1fs)\n", time() - t0)
    # Per-bin report (only non-zero bins to keep output readable)
    println("  bin distribution (non-empty bins):")
    for b in 0:N_BINS
        n = bin_counts[b + 1]
        n > 0 || continue
        f_mid = b / Float64(N_BINS)
        lbl = b == 0 ? "(unchanged)" : "$(255 + b)"
        @printf("    bin %3d  f≈%.2f  label=%-7s  voxels=%d\n", b, f_mid, lbl, n)
    end
    flush(stdout)
    bin_counts
end

function write_phantom_raw_u16(path::String, phantom::Array{UInt16,3})
    t0 = time()
    nx, ny, nz = size(phantom)
    open(path, "w") do io
        write(io, phantom)
    end
    fsize = filesize(path)
    @printf("[write] %s: %d×%d×%d UInt16 = %.0f MB (%.1fs)\n",
            basename(path), nx, ny, nz, fsize/1024^2, time()-t0)
    flush(stdout)
    fsize
end

# Material map for non-mixture base labels (XCAT activ_material_table + PIMMD).
# Mixture labels 256..355 are NOT listed here — loader builds them from
# [mixture_materials].
const LABEL_TO_MATERIAL = Dict{Int, String}(
    0  => "air",
    1  => "softtissue",
    2  => "softtissue",
    3  => "cortical_bone",
    4  => "muscle",
    5  => "cortical_bone",
    6  => "lung",
    7  => "softtissue",
    8  => "cortical_bone",
    9  => "cortical_bone",
    10 => "softtissue",
    11 => "softtissue",
    12 => "softtissue",
    13 => "softtissue",
    14 => "softtissue",
    15 => "muscle",
    16 => "muscle",
    17 => "muscle",
    18 => "muscle",
    19 => "blood",
    20 => "blood",
    21 => "blood",
    22 => "blood",
    23 => "softtissue",
    24 => "blood",
    25 => "blood",
    26 => "blood",          # XCAT coronary_art / vessel_lumen
    27 => "blood",          # XCAT coronary_vein
    28 => "blood",          # PIMMD overlay: dias_aorta
    29 => "softtissue",     # pericardium
    30 => "muscle",         # PIMMD overlay: vessel_wall
    31 => "adipose",        # PIMMD overlay: lipid core
    32 => "cortical_bone",  # PIMMD overlay: calcification
    70 => "air",            # sinus
)

function write_manifest(out_dir::String, raw_basename::String,
                        hist_pre::Vector{Int64}, hist_post::Vector{Int64},
                        n_writes_total::Int, n_sub::Int,
                        tree_files::Vector{String}, seg_counts::Vector{Int},
                        bin_counts::Vector{Int64})
    path = joinpath(out_dir, "phantom_manifest.toml")

    # Compact present-label lists: only emit labels with count > 0
    present_pre  = sort([(i-1, hist_pre[i])  for i in 1:length(hist_pre)  if hist_pre[i]  > 0])
    present_post = sort([(i-1, hist_post[i]) for i in 1:length(hist_post) if hist_post[i] > 0])

    open(path, "w") do io
        println(io, "# phantom_manifest.toml")
        println(io, "# Auto-generated by VascularTreeSim.jl/scripts/voxelize_csv_into_phantom.jl")
        println(io, "# Consumed by load_phantom.jl → builds a BasisSimulator.Phantom.")
        println(io)
        println(io, "[phantom]")
        println(io, "raw_path = \"$raw_basename\"")
        println(io, "dims = [$(PHANTOM_DIMS[1]), $(PHANTOM_DIMS[2]), $(PHANTOM_DIMS[3])]")
        println(io, "dtype = \"UInt16\"")
        println(io, "byte_order = \"little-endian\"")
        println(io, "voxel_ordering = \"x-fastest\"   # Fortran/column-major (Julia native)")
        println(io, "voxel_size_cm = [$VOXEL_CM, $VOXEL_CM, $VOXEL_CM]")
        println(io, "xcat_origin_cm = [$(XCAT_ORIGIN_CM[1]), $(XCAT_ORIGIN_CM[2]), $(XCAT_ORIGIN_CM[3])]")
        println(io)
        println(io, "[embed]")
        println(io, "stage = \"no_contrast\"")
        println(io, "method = \"sub_voxel_monte_carlo\"")
        println(io, "n_sub_per_dim = $n_sub")
        println(io, "n_sub_total = $(n_sub^3)")
        println(io, "voxel_increment_writes = $n_writes_total")
        println(io, "nrb_to_phantom_offset_cm = [$(NRB_TO_PHANTOM_OFFSET[1]), $(NRB_TO_PHANTOM_OFFSET[2]), $(NRB_TO_PHANTOM_OFFSET[3])]")
        println(io, "tree_csv_files = [", join(["\"$f\"" for f in tree_files], ", "), "]")
        println(io, "tree_segment_counts = [", join(string.(seg_counts), ", "), "]")
        println(io, "writable_base_labels = [15, 16, 17, 18]   # myocardium (LV/RV/LA/RA)")
        println(io)
        println(io, "[mixture_materials]")
        println(io, "# Blood-only mixture: 100-bin volume-fraction quantization of (blood + muscle)")
        println(io, "# per myocardial voxel. Loader builds 100 XA.Material objects at load time.")
        println(io, "# label = label_base + (bin - 1)        for bin ∈ 1..n_bins")
        println(io, "# f_blood at label = bin / n_bins        (i.e. label 256 ≈ 1% blood, 355 = 100%)")
        println(io, "components = [\"blood\", \"muscle\"]   # XA.Materials.* symbols")
        println(io, "n_bins = $N_BINS")
        println(io, "label_base = $(Int(BIN_LABEL_BASE))")
        println(io, "bin_voxel_counts = [", join(string.(bin_counts[1:N_BINS+1]), ", "), "]   # bins 0..$N_BINS")
        println(io)
        println(io, "[present_labels_pre_embed]")
        println(io, "# Histogram of labels in the source phantom (UInt8, before promotion).")
        for (l, n) in present_pre
            println(io, "\"$l\" = $n")
        end
        println(io)
        println(io, "[present_labels_post_embed]")
        println(io, "# Histogram of labels in the embedded UInt16 phantom (after VF quantization).")
        for (l, n) in present_post
            println(io, "\"$l\" = $n")
        end
        println(io)
        println(io, "[materials]")
        println(io, "# label → BasisSimulator material symbol (resolved against")
        println(io, "# BasisSimulator.MATERIALS_REGISTRY → XrayAttenuation.Materials.*).")
        println(io, "# Mixture labels 256..", 255 + N_BINS, " are NOT listed here — the loader")
        println(io, "# constructs them from [mixture_materials] above.")
        for k in sort(collect(keys(LABEL_TO_MATERIAL)))
            println(io, "\"$k\" = \"$(LABEL_TO_MATERIAL[k])\"")
        end
    end
    @printf("[write] %s\n", path)
    path
end

function write_info(out_dir::String, raw_basename::String,
                    n_sub::Int, n_writes_total::Int,
                    tree_files::Vector{String}, seg_counts::Vector{Int},
                    bin_counts::Vector{Int64})
    path = joinpath(out_dir, "phantom_info.txt")
    open(path, "w") do io
        println(io, "XCAT vmale50 with Grown Coronary Trees (no-contrast, 100-bin VF)")
        println(io, "=" ^ 60)
        println(io)
        println(io, "Raw file: $raw_basename")
        println(io, "  Dimensions: $(PHANTOM_DIMS[1]) × $(PHANTOM_DIMS[2]) × $(PHANTOM_DIMS[3])")
        println(io, "  Data type:  UInt16 little-endian, x-fastest (Fortran order)")
        println(io, "  Voxel size: $(VOXEL_CM*10) mm isotropic")
        println(io)
        println(io, "ImageJ raw import: 16-bit unsigned, $(PHANTOM_DIMS[1]) × $(PHANTOM_DIMS[2]), $(PHANTOM_DIMS[3]) images, little-endian.")
        println(io)
        println(io, "Embed method: sub-voxel Monte-Carlo with $(n_sub)^3 = $(n_sub^3) samples per voxel")
        println(io, "  (f_blood precision ≈ $(round(1 / n_sub^3 * 100; digits=2))%; sub-1% bin at n_sub ≥ 5).")
        println(io, "  Only myocardium voxels (labels 15-18) are eligible to be relabeled.")
        println(io)
        println(io, "Label scheme post-embed:")
        println(io, "    0-32   original XCAT labels (preserved; UInt8 range — fits in UInt16)")
        println(io, "  256-355  blood mixture labels (label X → bin = X-255 → f_blood ≈ bin/100)")
        println(io, "           e.g. label 256 ≈ 1% blood, label 355 = 100% blood")
        println(io)
        println(io, "UInt16 headroom: ~65500 labels free beyond the current 355 ceiling — used by")
        println(io, "the next stage (apply_contrast_at_peak.jl) to expand into a cross-product of")
        println(io, "(blood %, iodine %) labels, all still within the 16-bit budget.")
        println(io)
        println(io, "Bin distribution (non-empty only):")
        for b in 0:N_BINS
            n = bin_counts[b + 1]
            n > 0 || continue
            f_mid = b / Float64(N_BINS)
            lbl = b == 0 ? "(unchanged)" : "$(255 + b)"
            @printf(io, "  bin %3d  f≈%.2f  label=%-7s  voxels=%d\n", b, f_mid, lbl, n)
        end
        println(io)
        println(io, "Total voxel-increment writes (from rasterizer): $n_writes_total")
        println(io)
        println(io, "Tree CSV sources:")
        for (f, n) in zip(tree_files, seg_counts)
            println(io, "  $f  ($(n) segments)")
        end
    end
    @printf("[write] %s\n", path)
    path
end

# ─────────────────────────────────────────────────────────────────────────────

function main()
    if length(ARGS) < 3
        println("Usage: julia --project=. --threads=auto scripts/voxelize_csv_into_phantom.jl \\")
        println("              TREE_DIR  PHANTOM_RAW_IN  OUTPUT_DIR  [N_SUB]")
        exit(1)
    end
    tree_dir = ARGS[1]
    raw_in   = ARGS[2]
    out_dir  = ARGS[3]
    n_sub    = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : DEFAULT_N_SUB
    n_sub >= 1 || error("N_SUB must be ≥ 1, got $n_sub")
    isdir(tree_dir) || error("tree_dir not found: $tree_dir")
    isfile(raw_in)  || error("phantom raw not found: $raw_in")
    isdir(out_dir)  || mkpath(out_dir)

    @printf("threads = %d\n", nthreads())
    println("tree_dir       = $tree_dir")
    println("phantom_raw_in = $raw_in")
    println("output_dir     = $out_dir")
    println("n_sub_per_dim  = $n_sub  ($(n_sub^3) sub-samples / voxel,",
            " f_blood precision ≈ $(round(1 / n_sub^3 * 100; digits=2))%)")
    flush(stdout)

    csvs = sort(filter(f -> endswith(f, "_segments.csv"),
                       readdir(tree_dir; join=true)))
    isempty(csvs) && error("No *_segments.csv found in $tree_dir")
    println("Trees discovered:")
    for f in csvs;  println("  $(basename(f))");  end
    flush(stdout)

    phantom = load_phantom_raw_u16(raw_in)
    t0 = time()
    hist_pre = label_histogram(phantom)
    @printf("[hist] pre-embed (%.1fs)\n", time()-t0); flush(stdout)

    nx, ny, nz = size(phantom)
    @printf("[alloc] f_blood Float32[%d×%d×%d] = %.1f GB\n",
            nx, ny, nz, nx*ny*nz*4/1024^3)
    flush(stdout)
    f_blood = zeros(Float32, nx, ny, nz)

    n_writes_total = 0
    seg_counts = Int[]
    for csv_path in csvs
        @printf("\n=== %s ===\n", basename(csv_path));  flush(stdout)
        A = parse_segments_csv(csv_path)
        push!(seg_counts, size(A, 2))
        n_writes_total += rasterize_capsules_vf!(f_blood, phantom, A; n_sub=n_sub)
        A = nothing
        GC.gc()
    end

    println()
    bin_counts = quantize_and_apply!(phantom, f_blood)
    f_blood = nothing
    GC.gc()

    raw_out = joinpath(out_dir, OUT_RAW_BASENAME)
    println()
    write_phantom_raw_u16(raw_out, phantom)

    t0 = time()
    hist_post = label_histogram(phantom)
    @printf("[hist] post-embed (%.1fs)\n", time()-t0); flush(stdout)

    write_manifest(out_dir, OUT_RAW_BASENAME, hist_pre, hist_post,
                   n_writes_total, n_sub,
                   [basename(f) for f in csvs], seg_counts, bin_counts)
    write_info(out_dir, OUT_RAW_BASENAME, n_sub, n_writes_total,
               [basename(f) for f in csvs], seg_counts, bin_counts)

    n_bin_total = sum(bin_counts[2:N_BINS+1])
    n_full = bin_counts[N_BINS + 1]
    n_partial = sum(bin_counts[2:N_BINS])
    voxel_vol_ml = VOXEL_CM^3
    f_eq = 0.0
    for b in 1:N_BINS
        f_eq += bin_counts[b + 1] * (b / Float64(N_BINS))
    end
    blood_eq_ml = f_eq * voxel_vol_ml
    println()
    @printf("done. (n_sub=%d, %d bins)\n", n_sub, N_BINS)
    @printf("  full(label 355)=%d  partial(256..354)=%d  total touched=%d\n",
            n_full, n_partial, n_bin_total)
    @printf("  f_blood-weighted grown volume = %.3f mL  (%.0f voxel-equivalents)\n",
            blood_eq_ml, f_eq)
end

main()
