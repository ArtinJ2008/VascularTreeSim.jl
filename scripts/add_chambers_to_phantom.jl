#!/usr/bin/env julia
#
# add_chambers_to_phantom.jl — patch an existing apply_contrast_at_peak.jl
# output (UInt16 cross-product phantom + manifest) so that XCAT chamber
# blood pools and great vessels also carry iodine (from the Bae 1998 PBPK
# central-circulation model at scan time).
#
# Without this step, chambers (LV blood pool, RV blood pool, aorta, etc.)
# would render as un-enhanced blood in the CT sim — making beam-hardening
# and chamber-HU evaluation impossible.
#
# Patches the following XCAT labels in-place (overwrites them with
# cross-product UInt16 labels at f_blood = 100%, iodine bin set from Bae C):
#
#   19  LV blood pool        ← C_LV_scan
#   20  RV blood pool        ← C_RV_scan
#   21  LA blood pool        ← C_pulm_vein_scan (≈ post-pulmonary blood)
#   22  RA blood pool        ← C_RV_scan       (well-mixed pre-RV)
#   24  pulmonary artery     ← C_pulm_artery_scan
#   25  pulmonary veins      ← C_pulm_vein_scan
#   28  great vessels/aorta  ← C_aorta_scan
#
# Cross-product encoding: same as apply_contrast_at_peak.jl
#   label = 256 + (bin_b - 1) × (n_iodine_bins + 1) + bin_i
#   For pure-blood chambers: bin_b = N_BLOOD_BINS = 100.
#   bin_i = round(C_iodine_mgI_ml / iodine_max × N_IODINE_BINS).
#
# CLI:
#   julia --project=. --threads=auto scripts/add_chambers_to_phantom.jl \
#         PEAK_PHANTOM_DIR  PEAK_IODINE_DIR  OUTPUT_DIR
#
# PEAK_PHANTOM_DIR    — contains apply_contrast_at_peak.jl output:
#                        vmale50_with_grown_coronaries_peak_contrast_u16.raw
#                        phantom_manifest.toml
# PEAK_IODINE_DIR     — extract_peak_iodine_bae.jl output: peak_metadata.toml
#                        with [chamber_concentrations].
# OUTPUT_DIR          — where to write the patched raw + new manifest.

using Base.Threads
using Printf
using TOML

const N_BLOOD_BINS  = 100
const N_IODINE_BINS = 100
const LABEL_BASE    = UInt16(256)
const FULL_BLOOD_BIN_B = N_BLOOD_BINS    # chamber → 100% blood

const CHAMBER_LABELS = Dict{UInt16, Symbol}(
    UInt16(19) => :LV_blood_pool,
    UInt16(20) => :RV_blood_pool,
    UInt16(21) => :LA_blood_pool,
    UInt16(22) => :RA_blood_pool,
    UInt16(24) => :pulm_artery,
    UInt16(25) => :pulm_veins,
    UInt16(28) => :great_vessels,
)

const OUT_RAW_BASENAME = "vmale50_with_grown_coronaries_peak_contrast_chambers_u16.raw"

# Encoding helper
@inline function crossprod_label(C_iodine::Float64, iodine_max::Float64)::UInt16
    if iodine_max <= 0
        bin_i = 0
    else
        bin_i = round(Int, clamp(C_iodine / iodine_max, 0.0, 1.0) * N_IODINE_BINS)
    end
    bin_i = clamp(bin_i, 0, N_IODINE_BINS)
    return LABEL_BASE + UInt16((FULL_BLOOD_BIN_B - 1) * (N_IODINE_BINS + 1) + bin_i)
end

# Resolve which chamber concentration applies to which XCAT label
function chamber_C(label::UInt16, cc::Dict)
    if     label == UInt16(19);  return Float64(cc["left_heart_mgI_ml"])    # LV blood
    elseif label == UInt16(20);  return Float64(cc["right_heart_mgI_ml"])   # RV blood
    elseif label == UInt16(21);  return Float64(cc["pulm_vein_mgI_ml"])     # LA ≈ pulm_vein
    elseif label == UInt16(22);  return Float64(cc["right_heart_mgI_ml"])   # RA ≈ RV
    elseif label == UInt16(24);  return Float64(cc["pulm_artery_mgI_ml"])   # pulm artery
    elseif label == UInt16(25);  return Float64(cc["pulm_vein_mgI_ml"])     # pulm veins
    elseif label == UInt16(28);  return Float64(cc["aorta_root_mgI_ml"])    # aorta + great vessels
    else
        error("Unhandled chamber label $label")
    end
end

function main()
    if length(ARGS) < 3
        println("Usage: julia --project=. --threads=auto scripts/add_chambers_to_phantom.jl \\")
        println("              PEAK_PHANTOM_DIR  PEAK_IODINE_DIR  OUTPUT_DIR")
        exit(1)
    end
    phantom_dir = ARGS[1]
    peak_dir    = ARGS[2]
    out_dir     = ARGS[3]
    mkpath(out_dir)

    manifest_in  = joinpath(phantom_dir, "phantom_manifest.toml")
    isfile(manifest_in) || error("manifest not found: $manifest_in")
    cfg = TOML.parsefile(manifest_in)
    raw_in = joinpath(phantom_dir, cfg["phantom"]["raw_path"])
    isfile(raw_in) || error("raw not found: $raw_in")
    dims = Tuple(Int.(cfg["phantom"]["dims"]))
    nx, ny, nz = dims

    peak_meta_path = joinpath(peak_dir, "peak_metadata.toml")
    isfile(peak_meta_path) || error("peak_metadata.toml not found: $peak_meta_path")
    pm = TOML.parsefile(peak_meta_path)
    haskey(pm, "chamber_concentrations") ||
        error("[chamber_concentrations] section missing from $peak_meta_path " *
              "(use extract_peak_iodine_bae.jl to generate it)")
    cc = pm["chamber_concentrations"]
    iodine_max = Float64(cfg["mixture_materials"]["iodine_max_mg_per_mL"])
    @info "Patching chambers" iodine_max_mg_per_mL=iodine_max
    println("Chamber concentrations (Bae, scan time):")
    for (lbl, name) in sort(collect(CHAMBER_LABELS); by=x->x[1])
        C = chamber_C(lbl, cc)
        new_label = crossprod_label(C, iodine_max)
        bin_i = Int(new_label) - Int(LABEL_BASE) - (FULL_BLOOD_BIN_B - 1) * (N_IODINE_BINS + 1)
        @printf("  XCAT %3d (%s)  →  cross-product label %d  (bin_b=100, bin_i=%d, C=%.2f mgI/mL)\n",
                Int(lbl), name, Int(new_label), bin_i, C)
    end

    # Load phantom (must be UInt16 — apply_contrast_at_peak.jl already up-cast it)
    @info "Loading phantom $(basename(raw_in))"
    t0 = time()
    phantom = Array{UInt16}(undef, nx, ny, nz)
    open(raw_in, "r") do io; read!(io, phantom); end
    @info "  loaded $(nx)×$(ny)×$(nz) UInt16 in $(round(time()-t0,digits=1))s"

    # Precompute label → new label
    chamber_remap = Dict{UInt16, UInt16}()
    for lbl in keys(CHAMBER_LABELS)
        chamber_remap[lbl] = crossprod_label(chamber_C(lbl, cc), iodine_max)
    end

    # Patch (threaded)
    n_thr = nthreads()
    counts = [Dict{UInt16, Int}() for _ in 1:n_thr]
    t0 = time()
    @threads :static for k in 1:nz
        tid = threadid()
        c = counts[tid]
        @inbounds for j in 1:ny, i in 1:nx
            v = phantom[i,j,k]
            if haskey(chamber_remap, v)
                phantom[i,j,k] = chamber_remap[v]
                c[v] = get(c, v, 0) + 1
            end
        end
    end
    total_counts = Dict{UInt16, Int}()
    for c in counts, (k, v) in c
        total_counts[k] = get(total_counts, k, 0) + v
    end
    @info "Patched in $(round(time()-t0,digits=1))s ($(n_thr) threads)"
    for lbl in sort(collect(keys(CHAMBER_LABELS)); by=x->x)
        n = get(total_counts, lbl, 0)
        @printf("  XCAT %3d  →  %d voxels patched\n", Int(lbl), n)
    end

    # Write raw
    out_raw = joinpath(out_dir, OUT_RAW_BASENAME)
    t0 = time()
    open(out_raw, "w") do io; write(io, phantom); end
    fsize = filesize(out_raw)
    @info "Wrote $(basename(out_raw)): $(round(fsize/1024^2,digits=0)) MB ($(round(time()-t0,digits=1))s)"

    # Write updated manifest. Identical to source, with chamber metadata added.
    out_manifest = joinpath(out_dir, "phantom_manifest.toml")
    open(out_manifest, "w") do io
        println(io, "# phantom_manifest.toml — peak-contrast + chamber-patched")
        println(io, "# Generated by VascularTreeSim/scripts/add_chambers_to_phantom.jl")
        println(io)
        println(io, "[phantom]")
        println(io, "raw_path = \"$(OUT_RAW_BASENAME)\"")
        println(io, "dims = [$(nx), $(ny), $(nz)]")
        println(io, "dtype = \"UInt16\"")
        println(io, "byte_order = \"little-endian\"")
        println(io, "voxel_ordering = \"x-fastest\"")
        v = cfg["phantom"]["voxel_size_cm"]
        println(io, "voxel_size_cm = [$(v[1]), $(v[2]), $(v[3])]")
        if haskey(cfg["phantom"], "xcat_origin_cm")
            xo = cfg["phantom"]["xcat_origin_cm"]
            println(io, "xcat_origin_cm = [$(xo[1]), $(xo[2]), $(xo[3])]")
        end
        # Helper: TOML-quote a single scalar
        fmt(v) = v isa AbstractString ? "\"$v\"" : string(v)
        # Helper: TOML-format any value (scalar or vector)
        function fmt_val(v)
            if v isa AbstractString
                return "\"$v\""
            elseif v isa AbstractVector
                return "[" * join(fmt.(v), ", ") * "]"
            else
                return string(v)
            end
        end
        # Copy [embed]
        if haskey(cfg, "embed")
            println(io)
            println(io, "[embed]")
            for (k, v) in cfg["embed"]
                println(io, "$k = $(fmt_val(v))")
            end
        end
        # [mixture_materials] — identical, iodine_max from upstream
        println(io)
        println(io, "[mixture_materials]")
        for (k, v) in cfg["mixture_materials"]
            println(io, "$k = $(fmt_val(v))")
        end
        # [chamber_patch] — record what we did
        println(io)
        println(io, "[chamber_patch]")
        println(io, "source_raw = \"$(basename(raw_in))\"")
        println(io, "peak_metadata = \"$(basename(peak_meta_path))\"")
        @printf(io, "iodine_max_mg_per_mL = %.6f\n", iodine_max)
        for (lbl, name) in sort(collect(CHAMBER_LABELS); by=x->x[1])
            new_lbl = chamber_remap[lbl]
            C = chamber_C(lbl, cc)
            n = get(total_counts, lbl, 0)
            @printf(io, "label_%d = { name = \"%s\", new_label = %d, C_mgI_ml = %.6f, n_voxels = %d }\n",
                    Int(lbl), name, Int(new_lbl), C, n)
        end
        # [materials] — copy as-is
        println(io)
        println(io, "[materials]")
        for k in sort(collect(keys(cfg["materials"])); by=x->parse(Int,x))
            println(io, "\"$k\" = \"$(cfg["materials"][k])\"")
        end
    end
    @info "Wrote $(basename(out_manifest))"
end

main()
