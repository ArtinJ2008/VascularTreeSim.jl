"""
    thigh_xcat_right_100um_test.jl

Build a right-thigh muscle mask from XCAT color-code output that contains both
legs, keep only the image half containing `leg_right`, and run a 100 um vessel
growth test using the richer femoral viewer.

Usage:
    julia --project=. examples/thigh_xcat_right_100um_test.jl [raw_path] [organ_ids] [xcat_log] [max_branches] [block_size] [terminal_um]
"""

include(joinpath(@__DIR__, "thigh_raw_nrrd_femoral_smoke.jl"))

const XCAT_RIGHT_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_thigh_100um_test")
const DEFAULT_XCAT_RAW = raw"C:\Users\msdjv\Downloads\both_thigh_corrected_act_1.raw"
const DEFAULT_XCAT_ORGAN_IDS = raw"C:\Users\msdjv\Downloads\organ_ids.txt"
const DEFAULT_XCAT_LOG = raw"C:\Users\msdjv\Downloads\both_thigh_corrected_log"
const DEFAULT_XCAT_TERMINAL_UM = 100.0
const DEFAULT_XCAT_MAX_BRANCHES = 1000
const DEFAULT_XCAT_BLOCK_SIZE = 5

const RIGHT_THIGH_MUSCLES = [
    "musc160", "musc161", "musc162", "musc164", "musc1663",
    "musc158", "musc165", "musc159", "musc141", "musc230",
    "musc167", "musc166", "musc79", "musc78", "musc80",
    "musc30", "musc29", "musc27", "musc26", "musc40",
    "musc34", "musc32", "musc28", "musc33", "musc35",
]

function parse_xcat_log(path::AbstractString)
    text = read(path, String)
    parse_int(pattern) = parse(Int, match(pattern, text).captures[1])
    parse_float(pattern) = parse(Float64, match(pattern, text).captures[1])
    nx = parse_int(r"x_array_size\s*=\s*(\d+)")
    ny = parse_int(r"y_array_size\s*=\s*(\d+)")
    start_slice = parse_int(r"starting slice number\s*=\s*(\d+)")
    end_slice = parse_int(r"ending slice number\s*=\s*(\d+)")
    pixel_width_cm = parse_float(r"pixel width\s*=\s*([0-9.]+)")
    slice_width_cm = parse_float(r"slice width\s*=\s*([0-9.]+)")
    nz = end_slice - start_slice + 1
    return (
        dims=(nx, ny, nz),
        spacing_cm=SVector(pixel_width_cm, pixel_width_cm, slice_width_cm),
        start_slice=start_slice,
        end_slice=end_slice,
    )
end

function parse_organ_ids(path::AbstractString)
    name_to_labels = Dict{String, Vector{UInt16}}()
    label_to_names = Dict{UInt16, Vector{String}}()
    for line in eachline(path)
        m = match(r"^\s*(.*?)\s*=\s*(\d+)\s*$", line)
        m === nothing && continue
        name = strip(m.captures[1])
        label = UInt16(parse(Int, m.captures[2]))
        push!(get!(name_to_labels, name, UInt16[]), label)
        push!(get!(label_to_names, label, String[]), name)
    end
    return name_to_labels, label_to_names
end

function labels_for(name_to_labels, names)
    labels = UInt16[]
    for name in names
        append!(labels, get(name_to_labels, name, UInt16[]))
    end
    return labels
end

function load_xcat_uint16_raw(path::AbstractString, dims)
    bytes = read(path)
    expected = prod(dims) * sizeof(UInt16)
    length(bytes) == expected || error("Raw size mismatch: expected $expected bytes, got $(length(bytes))")
    return reinterpret(UInt16, bytes)
end

function choose_right_half(raw_labels, dims, right_leg_labels::Vector{UInt16})
    label_set = Set(right_leg_labels)
    nx, ny, nz = dims
    split = div(nx, 2)
    lower = 0
    upper = 0
    for idx0 in 0:(length(raw_labels) - 1)
        raw_labels[idx0 + 1] in label_set || continue
        x = (idx0 % nx) + 1
        if x <= split
            lower += 1
        else
            upper += 1
        end
    end
    lower + upper > 0 || error("No leg_right voxels found; cannot choose right half")
    return lower >= upper ? (1:split, lower, upper, "lower-x") : ((split + 1):nx, lower, upper, "upper-x")
end

function build_right_thigh_mask(raw_labels, dims, target_labels::Vector{UInt16}, keep_x)
    target_set = Set(target_labels)
    nx, ny, nz = dims
    mask = falses(nx, ny, nz)
    keep = Set(keep_x)
    for idx0 in 0:(length(raw_labels) - 1)
        lab = raw_labels[idx0 + 1]
        lab in target_set || continue
        x = (idx0 % nx) + 1
        x in keep || continue
        y = (div(idx0, nx) % ny) + 1
        z = div(idx0, nx * ny) + 1
        mask[x, y, z] = true
    end
    count(mask) > 0 || error("Right-thigh target mask is empty")
    return mask
end

function crop_mask(mask::BitArray{3}, spacing_cm::SVector{3, Float64}; pad::Int=2)
    dims = size(mask)
    lo = [typemax(Int), typemax(Int), typemax(Int)]
    hi = [0, 0, 0]
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        mask[i, j, k] || continue
        lo[1] = min(lo[1], i); hi[1] = max(hi[1], i)
        lo[2] = min(lo[2], j); hi[2] = max(hi[2], j)
        lo[3] = min(lo[3], k); hi[3] = max(hi[3], k)
    end
    lo = max.(lo .- pad, 1)
    hi = min.(hi .+ pad, collect(dims))
    cropped = mask[lo[1]:hi[1], lo[2]:hi[2], lo[3]:hi[3]]
    origin_cm = SVector((lo[1] - 1) * spacing_cm[1], (lo[2] - 1) * spacing_cm[2], (lo[3] - 1) * spacing_cm[3])
    return BitArray(cropped), origin_cm, Tuple(lo), Tuple(hi)
end

function write_mask_artifacts(output_dir::AbstractString, mask::BitArray{3},
                              origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    raw_path = joinpath(output_dir, "right_thigh_muscle_mask.raw")
    nhdr_path = joinpath(output_dir, "right_thigh_muscle_mask.nhdr")
    open(raw_path, "w") do io
        for k in axes(mask, 3), j in axes(mask, 2), i in axes(mask, 1)
            write(io, UInt8(mask[i, j, k] ? 1 : 0))
        end
    end
    open(nhdr_path, "w") do io
        println(io, "NRRD0005")
        println(io, "type: uchar")
        println(io, "dimension: 3")
        println(io, "space: left-posterior-superior")
        println(io, "sizes: $(size(mask, 1)) $(size(mask, 2)) $(size(mask, 3))")
        println(io, "space directions: ($(spacing_cm[1] * 10),0,0) (0,$(spacing_cm[2] * 10),0) (0,0,$(spacing_cm[3] * 10))")
        println(io, "space origin: ($(origin_cm[1] * 10),$(origin_cm[2] * 10),$(origin_cm[3] * 10))")
        println(io, "encoding: raw")
        println(io, "endian: little")
        println(io, "data file: right_thigh_muscle_mask.raw")
    end
    return nhdr_path, raw_path
end

function write_xcat_summary(path, raw_path, organ_ids_path, log_path, xcat_info, keep_desc,
                            kept_counts, crop_lo, crop_hi, domain, mask_info, tree, stats;
                            max_branches::Int, block_size::Int, terminal_um::Float64)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    fixed_segments = count(label -> haskey(FIXED_BRANCH_DIAMETER_CM, label), tree.segment_label)
    open(path, "w") do io
        println(io, "# XCAT Right-Thigh 100um Test")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- XCAT raw: $(raw_path)")
        println(io, "- Organ IDs: $(organ_ids_path)")
        println(io, "- XCAT log: $(log_path)")
        println(io, "- Source dims: $(xcat_info.dims)")
        println(io, "- Source slice range: $(xcat_info.start_slice)-$(xcat_info.end_slice)")
        println(io, "- Source spacing: $(round.(Tuple(xcat_info.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Kept right-leg half: $(keep_desc)")
        println(io, "- leg_right half counts: lower-x=$(kept_counts[1]), upper-x=$(kept_counts[2])")
        println(io, "- Crop lo index: $(crop_lo)")
        println(io, "- Crop hi index: $(crop_hi)")
        println(io, "- Cropped dims: $(size(domain.mask))")
        println(io, "- Foreground muscle voxels: $(mask_info.foreground)")
        println(io, "- Cropped extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        println(io, "- Terminal branch diameter: $(round(terminal_um; digits=3)) um")
        println(io, "- Max branches requested: $(max_branches)")
        println(io, "- Coverage/route block size: $(block_size)")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- Fixed seed segments: $(fixed_segments)")
        println(io, "- Predicted/grown segments: $(length(tree.segment_start) - fixed_segments)")
        println(io, "- Total segments: $(length(tree.segment_start))")
        println(io, "- Terminal vertices: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum diameter: $(round(maximum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "The growth domain is the target right-thigh muscle label set from XCAT, with the source volume split to the half containing leg_right.")
    end
    return path
end

function main_xcat_right()
    Random.seed!(42)
    raw_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_XCAT_RAW
    organ_ids_path = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_XCAT_ORGAN_IDS
    log_path = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_XCAT_LOG
    max_branches = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : DEFAULT_XCAT_MAX_BRANCHES
    block_size = length(ARGS) >= 5 ? parse(Int, ARGS[5]) : DEFAULT_XCAT_BLOCK_SIZE
    terminal_um = length(ARGS) >= 6 ? parse(Float64, ARGS[6]) : DEFAULT_XCAT_TERMINAL_UM
    terminal_cm = terminal_um / 1e4

    mkpath(XCAT_RIGHT_OUTPUT_DIR)
    xcat_info = parse_xcat_log(log_path)
    name_to_labels, _ = parse_organ_ids(organ_ids_path)
    raw_labels = load_xcat_uint16_raw(raw_path, xcat_info.dims)

    target_labels = labels_for(name_to_labels, RIGHT_THIGH_MUSCLES)
    right_leg_labels = labels_for(name_to_labels, ["leg_right"])
    keep_x, lower_count, upper_count, keep_desc = choose_right_half(raw_labels, xcat_info.dims, right_leg_labels)
    full_mask = build_right_thigh_mask(raw_labels, xcat_info.dims, target_labels, keep_x)
    mask, origin_cm, crop_lo, crop_hi = crop_mask(full_mask, xcat_info.spacing_cm; pad=2)
    nhdr_path, mask_raw_path = write_mask_artifacts(XCAT_RIGHT_OUTPUT_DIR, mask, origin_cm, xcat_info.spacing_cm)

    domain, mask_info = build_domain_from_mask(mask, origin_cm, xcat_info.spacing_cm)
    trunk_points = approximate_femoral_trunk(mask, origin_cm, xcat_info.spacing_cm, mask_info)
    tree = make_anatomic_seed_tree(mask, origin_cm, xcat_info.spacing_cm, mask_info, trunk_points)
    tree.terminal_diameter_cm = terminal_cm
    trees = Dict("XCATRightThigh" => tree)

    coverage_points = coverage_target_points_blockwise(domain; block_size=block_size)
    graph_points = coverage_target_points_blockwise(domain; block_size=block_size)
    viewer_points = coverage_target_points_blockwise(domain; block_size=max(4, block_size))

    println("=" ^ 78)
    println("XCAT right-thigh 100um test")
    println("Raw: $(raw_path)")
    println("Output: $(XCAT_RIGHT_OUTPUT_DIR)")
    println("Source dims: $(xcat_info.dims), spacing_cm=$(xcat_info.spacing_cm)")
    println("Keeping $(keep_desc) half for leg_right (lower=$lower_count, upper=$upper_count)")
    println("Cropped dims: $(size(mask)), foreground=$(count(mask))")
    println("Terminal diameter: $(terminal_um) um")
    println("Max branches: $(max_branches), block size: $(block_size)")
    println("[seeds] fixed_segments=$(length(tree.segment_start)) trunk_points=$(length(trunk_points))")
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1))")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.20,
        capillary_diameter_cm=terminal_cm,
        max_new_branches_per_tree=max_branches,
        graph_neighbors=16,
        min_frontier_separation_cm=0.35,
        max_path_nodes=24,
        target_p95_distance_cm=0.50,
        frontier_batch=16,
        gamma=3.0,
        smooth_passes=10,
        spline_density=4,
        max_segment_length_cm=0.20,
        turn_penalty=0.35,
        graph_jitter_cm=0.025,
        use_gpu=false)

    preserve_fixed_branch_diameters!(tree)
    stats = growth_stats["XCATRightThigh"]

    csv_path = joinpath(XCAT_RIGHT_OUTPUT_DIR, "xcat_right_thigh_100um_segments.csv")
    trunk_csv = joinpath(XCAT_RIGHT_OUTPUT_DIR, "approx_femoral_trunk_points.csv")
    seed_csv = joinpath(XCAT_RIGHT_OUTPUT_DIR, "approx_femoral_seed_segments.csv")
    domain_csv = joinpath(XCAT_RIGHT_OUTPUT_DIR, "xcat_right_thigh_domain_points.csv")
    viewer_path = joinpath(XCAT_RIGHT_OUTPUT_DIR, "xcat_right_thigh_100um_viewer.html")
    summary_path = joinpath(XCAT_RIGHT_OUTPUT_DIR, "summary.md")

    write_growth_csv(csv_path, "XCATRightThigh", tree)
    write_trunk_csv(trunk_csv, trunk_points)
    write_fixed_seed_csv(seed_csv, tree)
    write_points_csv(domain_csv, viewer_points)
    write_femoral_viewer(viewer_path, mask, origin_cm, xcat_info.spacing_cm, mask_info, viewer_points, tree)
    write_xcat_summary(summary_path, raw_path, organ_ids_path, log_path, xcat_info, keep_desc,
        (lower_count, upper_count), crop_lo, crop_hi, domain, mask_info, tree, stats;
        max_branches=max_branches, block_size=block_size, terminal_um=terminal_um)

    println()
    println("XCAT Right-Thigh Test Result")
    println("  added branches       = $(stats.added)")
    println("  total segments       = $(length(tree.segment_start))")
    println("  min diameter         = $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
    println("  p95 coverage         = $(round(stats.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(nhdr_path)")
    println("  $(mask_raw_path)")
    println("  $(csv_path)")
    println("  $(trunk_csv)")
    println("  $(seed_csv)")
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_xcat_right()
end

