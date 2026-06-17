"""
    thigh_xcat_main_arteries_50um_gpu.jl

Large right-thigh GPU run seeded from the selected main XCAT artery patches.
The run samples enough tissue target points for Murray scaling to bring the
root branch to the extracted femoral diameter at a 50 um terminal diameter.

This script intentionally skips the embedded HTML viewer because a 50 um full
run can produce tens of millions of segments. Generate downsampled viewers from
the CSV after the run.

Usage:
    julia --project=. examples/thigh_xcat_main_arteries_50um_gpu.jl [terminal_um] [output_dir] [seed_csv] [raw_path] [organ_ids] [xcat_log] [target_branches_or_auto] [frontier_batch] [graph_block_size] [min_frontier_separation_cm] [max_segment_length_cm] [graph_neighbors] [max_path_nodes] [graph_jitter_cm] [snap_terminal_to_target] [max_terminal_snap_cm] [coverage_multiplier] [use_indexed_anchor] [use_astar_routing] [frontier_candidate_factor]
"""

include(joinpath(@__DIR__, "thigh_xcat_femoral_100um.jl"))

using Dates
using CUDA
using Printf
using Random
using StaticArrays

const DEFAULT_50UM_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_thigh_main_arteries_50um_full")
const DEFAULT_MAIN_ARTERY_SEED_CSV = joinpath(dirname(@__DIR__), "output", "xcat_femoral_geometry", "xcat_main_thigh_arteries_centerline.csv")
const DEFAULT_50UM_FRONTIER_BATCH = 4096
const DEFAULT_50UM_GRAPH_BLOCK_SIZE = 1
const DEFAULT_50UM_MIN_FRONTIER_SEPARATION_CM = 0.01
const DEFAULT_50UM_MAX_SEGMENT_LENGTH_CM = 0.05
const DEFAULT_50UM_GRAPH_NEIGHBORS = 24
const DEFAULT_50UM_MAX_PATH_NODES = 48
const DEFAULT_50UM_GRAPH_JITTER_CM = 0.005
const DEFAULT_50UM_SNAP_TERMINAL_TO_TARGET = true
const DEFAULT_50UM_MAX_TERMINAL_SNAP_CM = 0.15
const DEFAULT_50UM_COVERAGE_MULTIPLIER = 1.10
const DEFAULT_50UM_USE_INDEXED_ANCHOR = true
const DEFAULT_50UM_USE_ASTAR_ROUTING = true
const DEFAULT_50UM_FRONTIER_CANDIDATE_FACTOR = 32

function parse_bool_arg(value::AbstractString)
    text = lowercase(strip(String(value)))
    text in ("true", "t", "yes", "y", "1") && return true
    text in ("false", "f", "no", "n", "0") && return false
    error("Expected a boolean value, got `$value`")
end

function random_points_in_mask(domain::VoxelShellDomain, n::Int; rng_seed::Int=42)
    mask = domain.mask
    dims = size(mask)
    foreground = Vector{UInt32}(undef, count(mask))
    idx = 0
    lin = UInt32(0)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        lin += UInt32(1)
        mask[i, j, k] || continue
        idx += 1
        foreground[idx] = lin
    end
    idx == length(foreground) || resize!(foreground, idx)
    rng = MersenneTwister(rng_seed)
    pts = Matrix{Float64}(undef, n, 3)
    nx, ny, _ = dims
    @inbounds for row in 1:n
        idx0 = Int(foreground[rand(rng, eachindex(foreground))]) - 1
        i = (idx0 % nx) + 1
        j = ((idx0 ÷ nx) % ny) + 1
        k = (idx0 ÷ (nx * ny)) + 1
        pts[row, 1] = domain.origin_cm[1] + (i - rand(rng)) * domain.spacing_cm[1]
        pts[row, 2] = domain.origin_cm[2] + (j - rand(rng)) * domain.spacing_cm[2]
        pts[row, 3] = domain.origin_cm[3] + (k - rand(rng)) * domain.spacing_cm[3]
    end
    return pts
end

function write_large_run_summary(path::AbstractString, raw_path, organ_ids_path, log_path,
                                 seed_csv, xcat_info, keep_desc, kept_counts, crop_lo, crop_hi,
                                 domain, mask_info, seed_paths, connector_gaps, tree, stats;
                                 terminal_um::Float64, target_branches::Int,
                                 coverage_count::Int, graph_count::Int,
                                 frontier_batch::Int, graph_block_size::Int,
                                 min_frontier_separation_cm::Float64,
                                 max_segment_length_cm::Float64,
                                 graph_neighbors::Int, max_path_nodes::Int,
                                 graph_jitter_cm::Float64,
                                 snap_terminal_to_target::Bool,
                                 max_terminal_snap_cm::Float64,
                                 coverage_multiplier::Float64,
                                 use_indexed_anchor::Bool,
                                 use_astar_routing::Bool,
                                 frontier_candidate_factor::Int,
                                 started_at, finished_at)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    fixed_segments = count(tree.is_xcat)
    open(path, "w") do io
        println(io, "# XCAT Main Arteries 50um GPU Full Run")
        println(io)
        println(io, "- Started: $(started_at)")
        println(io, "- Finished: $(finished_at)")
        println(io, "- XCAT raw: $(raw_path)")
        println(io, "- Organ IDs: $(organ_ids_path)")
        println(io, "- XCAT log: $(log_path)")
        println(io, "- XCAT seed CSV: $(seed_csv)")
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
        println(io, "- Target added branches: $(target_branches)")
        println(io, "- Coverage points: $(coverage_count)")
        println(io, "- Coverage multiplier: $(round(coverage_multiplier; digits=3))")
        println(io, "- Graph points: $(graph_count)")
        println(io, "- Graph block size: $(graph_block_size)")
        println(io, "- Frontier batch: $(frontier_batch)")
        println(io, "- Minimum frontier separation: $(round(min_frontier_separation_cm * 10; digits=3)) mm")
        println(io, "- Maximum segment length: $(round(max_segment_length_cm * 10; digits=3)) mm")
        println(io, "- Graph neighbors: $(graph_neighbors)")
        println(io, "- Maximum path nodes: $(max_path_nodes)")
        println(io, "- Graph jitter: $(round(graph_jitter_cm * 10; digits=3)) mm")
        println(io, "- Snap terminal to target: $(snap_terminal_to_target)")
        println(io, "- Maximum terminal snap: $(round(max_terminal_snap_cm * 10; digits=3)) mm")
        println(io, "- Indexed anchor lookup: $(use_indexed_anchor)")
        println(io, "- A* routing: $(use_astar_routing)")
        println(io, "- Frontier candidate factor: $(frontier_candidate_factor)")
        println(io)
        println(io, "## XCAT Seed Geometry")
        println(io)
        for path in seed_paths
            path_len = sum(norm(path.points[i] - path.points[i - 1]) for i in 2:length(path.points))
            @printf(io, "- %s: %d points, %.2f mm centerline length, %.2f-%.2f mm diameter\n",
                path.surface, length(path.points), path_len * 10.0,
                minimum(path.diameters_cm) * 10.0, maximum(path.diameters_cm) * 10.0)
        end
        for (surface, gap) in sort(collect(connector_gaps); by=first)
            @printf(io, "- %s attachment gap: %.2f mm\n", surface, gap * 10.0)
        end
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- XCAT fixed seed segments: $(fixed_segments)")
        println(io, "- Predicted/grown segments: $(length(tree.segment_start) - fixed_segments)")
        println(io, "- Total segments: $(length(tree.segment_start))")
        println(io, "- Terminal vertices: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum diameter: $(round(maximum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
    end
    return path
end

function main_50um_gpu()
    started_at = Dates.now()
    terminal_um = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
    output_dir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_50UM_OUTPUT_DIR
    seed_csv = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_MAIN_ARTERY_SEED_CSV
    raw_path = length(ARGS) >= 4 ? ARGS[4] : DEFAULT_XCAT_RAW
    organ_ids_path = length(ARGS) >= 5 ? ARGS[5] : DEFAULT_XCAT_ORGAN_IDS
    log_path = length(ARGS) >= 6 ? ARGS[6] : DEFAULT_XCAT_LOG
    target_arg = length(ARGS) >= 7 ? ARGS[7] : "auto"
    frontier_batch = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : DEFAULT_50UM_FRONTIER_BATCH
    graph_block_size = length(ARGS) >= 9 ? parse(Int, ARGS[9]) : DEFAULT_50UM_GRAPH_BLOCK_SIZE
    min_frontier_separation_cm = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : DEFAULT_50UM_MIN_FRONTIER_SEPARATION_CM
    max_segment_length_cm = length(ARGS) >= 11 ? parse(Float64, ARGS[11]) : DEFAULT_50UM_MAX_SEGMENT_LENGTH_CM
    graph_neighbors = length(ARGS) >= 12 ? parse(Int, ARGS[12]) : DEFAULT_50UM_GRAPH_NEIGHBORS
    max_path_nodes = length(ARGS) >= 13 ? parse(Int, ARGS[13]) : DEFAULT_50UM_MAX_PATH_NODES
    graph_jitter_cm = length(ARGS) >= 14 ? parse(Float64, ARGS[14]) : DEFAULT_50UM_GRAPH_JITTER_CM
    snap_terminal_to_target = length(ARGS) >= 15 ? parse_bool_arg(ARGS[15]) : DEFAULT_50UM_SNAP_TERMINAL_TO_TARGET
    max_terminal_snap_cm = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : DEFAULT_50UM_MAX_TERMINAL_SNAP_CM
    coverage_multiplier = length(ARGS) >= 17 ? parse(Float64, ARGS[17]) : DEFAULT_50UM_COVERAGE_MULTIPLIER
    use_indexed_anchor = length(ARGS) >= 18 ? parse_bool_arg(ARGS[18]) : DEFAULT_50UM_USE_INDEXED_ANCHOR
    use_astar_routing = length(ARGS) >= 19 ? parse_bool_arg(ARGS[19]) : DEFAULT_50UM_USE_ASTAR_ROUTING
    frontier_candidate_factor = length(ARGS) >= 20 ? parse(Int, ARGS[20]) : DEFAULT_50UM_FRONTIER_CANDIDATE_FACTOR
    terminal_cm = terminal_um / 1e4

    mkpath(output_dir)
    xcat_info = parse_xcat_log(log_path)
    name_to_labels, _ = parse_organ_ids(organ_ids_path)
    raw_labels = load_xcat_uint16_raw(raw_path, xcat_info.dims)

    target_labels = labels_for(name_to_labels, RIGHT_THIGH_MUSCLES)
    right_leg_labels = labels_for(name_to_labels, ["leg_right"])
    keep_x, lower_count, upper_count, keep_desc = choose_right_half(raw_labels, xcat_info.dims, right_leg_labels)
    full_mask = build_right_thigh_mask(raw_labels, xcat_info.dims, target_labels, keep_x)
    mask, origin_cm, crop_lo, crop_hi = crop_mask(full_mask, xcat_info.spacing_cm; pad=2)
    nhdr_path, mask_raw_path = write_mask_artifacts(output_dir, mask, origin_cm, xcat_info.spacing_cm)

    domain, mask_info = build_domain_from_mask(mask, origin_cm, xcat_info.spacing_cm)
    seed_paths = load_xcat_seed_paths(seed_csv)
    tree, ordered_seed_paths, connector_gaps = make_xcat_femoral_seed_tree(seed_paths; terminal_diameter_cm=terminal_cm)
    target_branches = lowercase(target_arg) == "auto" ?
        ceil(Int, (tree.root_diameter_cm / terminal_cm)^3) :
        parse(Int, target_arg)
    coverage_count = max(target_branches, ceil(Int, target_branches * coverage_multiplier))
    trees = Dict("XCATMainArteries50um" => tree)

    println("=" ^ 78)
    println("XCAT main-arteries 50um GPU full run")
    println("Started: $(started_at)")
    println("Raw: $(raw_path)")
    println("Seed CSV: $(seed_csv)")
    println("Output: $(output_dir)")
    println("Terminal diameter: $(terminal_um) um")
    println("Root/femoral diameter target: $(round(tree.root_diameter_cm * 10; digits=3)) mm")
    println("Target added branches: $(target_branches)")
    println("Coverage points: $(coverage_count)")
    println("Graph block size: $(graph_block_size)")
    println("Frontier batch: $(frontier_batch)")
    println("Minimum frontier separation: $(round(min_frontier_separation_cm * 10; digits=3)) mm")
    println("Maximum segment length: $(round(max_segment_length_cm * 10; digits=3)) mm")
    println("Snap terminal to target: $(snap_terminal_to_target)")
    println("Maximum terminal snap: $(round(max_terminal_snap_cm * 10; digits=3)) mm")
    println("Indexed anchor lookup: $(use_indexed_anchor)")
    println("A* routing: $(use_astar_routing)")
    println("Frontier candidate factor: $(frontier_candidate_factor)")
    println("Cropped dims: $(size(mask)), foreground=$(count(mask))")
    println("[xcat seeds] surfaces=$(join([p.surface for p in ordered_seed_paths], ", "))")
    println("[xcat seeds] fixed_segments=$(length(tree.segment_start))")
    println("[coverage] sampling $(coverage_count) random target points inside mask")
    flush(stdout)

    coverage_points = random_points_in_mask(domain, coverage_count; rng_seed=42)
    graph_points = coverage_target_points_blockwise(domain; block_size=graph_block_size)
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) frontier_batch=$(frontier_batch)")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=terminal_cm,
        max_new_branches_per_tree=target_branches,
        graph_neighbors=graph_neighbors,
        min_frontier_separation_cm=min_frontier_separation_cm,
        max_path_nodes=max_path_nodes,
        target_p95_distance_cm=Inf,
        target_max_distance_cm=Inf,
        frontier_batch=frontier_batch,
        gamma=3.0,
        smooth_passes=8,
        spline_density=3,
        max_segment_length_cm=max_segment_length_cm,
        turn_penalty=0.35,
        graph_jitter_cm=graph_jitter_cm,
        snap_terminal_to_target=snap_terminal_to_target,
        max_terminal_snap_cm=max_terminal_snap_cm,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        use_gpu=true)

    restore_xcat_seed_diameters!(tree, ordered_seed_paths)
    stats = growth_stats["XCATMainArteries50um"]

    csv_path = joinpath(output_dir, "xcat_main_arteries_50um_segments.csv")
    seed_out_csv = joinpath(output_dir, "xcat_main_arteries_seed_segments.csv")
    summary_path = joinpath(output_dir, "summary.md")
    write_growth_csv(csv_path, "XCATMainArteries50um", tree)
    write_xcat_seed_csv(seed_out_csv, tree)
    write_large_run_summary(summary_path, raw_path, organ_ids_path, log_path,
        seed_csv, xcat_info, keep_desc, (lower_count, upper_count), crop_lo, crop_hi,
        domain, mask_info, ordered_seed_paths, connector_gaps, tree, stats;
        terminal_um=terminal_um,
        target_branches=target_branches,
        coverage_count=size(coverage_points, 1),
        graph_count=size(graph_points, 1),
        frontier_batch=frontier_batch,
        graph_block_size=graph_block_size,
        min_frontier_separation_cm=min_frontier_separation_cm,
        max_segment_length_cm=max_segment_length_cm,
        graph_neighbors=graph_neighbors,
        max_path_nodes=max_path_nodes,
        graph_jitter_cm=graph_jitter_cm,
        snap_terminal_to_target=snap_terminal_to_target,
        max_terminal_snap_cm=max_terminal_snap_cm,
        coverage_multiplier=coverage_multiplier,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        started_at=started_at,
        finished_at=Dates.now())

    println()
    println("XCAT Main Arteries 50um Result")
    println("  added branches       = $(stats.added)")
    println("  xcat fixed segments  = $(count(tree.is_xcat))")
    println("  total segments       = $(length(tree.segment_start))")
    println("  min diameter         = $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
    println("  max diameter         = $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
    println("  p95 coverage         = $(round(stats.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(nhdr_path)")
    println("  $(mask_raw_path)")
    println("  $(csv_path)")
    println("  $(seed_out_csv)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_50um_gpu()
end
