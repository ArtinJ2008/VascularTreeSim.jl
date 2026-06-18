"""
    right_leg_xcat_50um_gpu.jl

Large right-leg GPU run using the corrected raw-label XCAT vessel workflow.
The right-leg muscle mask, active arterial seeds, and fixed vein exports
come from the voxelized `arteries_rleg` / `veins_rleg` labels rather than the
older NRB `occ*` centerlines.

Usage:
    julia --project=. examples/right_leg_xcat_50um_gpu.jl [terminal_um] [output_dir] [raw_path] [organ_ids] [xcat_log] [target_branches_or_auto] [frontier_batch] [graph_block_size] [min_frontier_separation_cm] [max_segment_length_cm] [graph_neighbors] [max_path_nodes] [graph_jitter_cm] [snap_terminal_to_target] [max_terminal_snap_cm] [coverage_multiplier] [use_indexed_anchor] [use_astar_routing] [frontier_candidate_factor] [fixed_vein_exports] [max_anchor_gap_cm] [growth_artery_seeds] [growth_artery_min_length_cm] [fixed_artery_min_length_cm] [fixed_vein_min_length_cm]
"""

include(joinpath(@__DIR__, "right_leg_xcat_trial.jl"))

using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays

const RIGHT_LEG_50UM_DEFAULT_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_leg_corrected_50um_full")
const RIGHT_LEG_50UM_TREE_NAME = "XCATRightLegCorrected50um"
const RIGHT_LEG_50UM_FRONTIER_BATCH = 4096
const RIGHT_LEG_50UM_GRAPH_BLOCK_SIZE = 1
const RIGHT_LEG_50UM_MIN_FRONTIER_SEPARATION_CM = 0.01
const RIGHT_LEG_50UM_MAX_SEGMENT_LENGTH_CM = 0.05
const RIGHT_LEG_50UM_GRAPH_NEIGHBORS = 24
const RIGHT_LEG_50UM_MAX_PATH_NODES = 48
const RIGHT_LEG_50UM_GRAPH_JITTER_CM = 0.005
const RIGHT_LEG_50UM_SNAP_TERMINAL_TO_TARGET = true
const RIGHT_LEG_50UM_MAX_TERMINAL_SNAP_CM = 0.15
const RIGHT_LEG_50UM_MAX_ANCHOR_GAP_CM = MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM
const RIGHT_LEG_50UM_COVERAGE_MULTIPLIER = 1.10
const RIGHT_LEG_50UM_USE_INDEXED_ANCHOR = true
const RIGHT_LEG_50UM_USE_ASTAR_ROUTING = true
const RIGHT_LEG_50UM_FRONTIER_CANDIDATE_FACTOR = 32
const RIGHT_LEG_50UM_MAIN_VESSEL_OVERLAYS = 6
const RIGHT_LEG_50UM_GROWTH_ARTERY_SEEDS = 0
const RIGHT_LEG_50UM_GROWTH_ARTERY_MIN_LENGTH_CM = 8.0
const RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM = 3.0
const RIGHT_LEG_50UM_FIXED_VEIN_MIN_LENGTH_CM = 8.0
const RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS = 0
const RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS = 1

function parse_bool_arg(value::AbstractString)
    text = lowercase(strip(String(value)))
    text in ("true", "t", "yes", "y", "1") && return true
    text in ("false", "f", "no", "n", "0") && return false
    error("Expected a boolean value, got `$value`")
end

function terminal_tag_um(terminal_um::Float64)
    if isapprox(terminal_um, round(terminal_um); atol=1e-6)
        return "$(round(Int, terminal_um))um"
    end
    return replace(@sprintf("%.3f", terminal_um), "." => "p") * "um"
end

function branch_caps_from_weights(names::Vector{String}, weights::Vector{Float64}, total_cap::Int)
    caps = Dict(name => 0 for name in names)
    total_cap <= 0 && return caps
    total_weight = sum(weights)
    shares = total_weight > 0.0 ? weights ./ total_weight .* total_cap : fill(total_cap / length(names), length(names))
    floors = floor.(Int, shares)
    for (name, cap) in zip(names, floors)
        caps[name] = cap
    end
    remaining = total_cap - sum(values(caps))
    order = sortperm(collect(eachindex(names)); by=i -> shares[i] - floors[i], rev=true)
    for idx in Iterators.take(Iterators.cycle(order), remaining)
        caps[names[idx]] += 1
    end
    return caps
end

function make_right_leg_growth_trees(paths::Vector{XCATSeedPath}; terminal_diameter_cm::Float64)
    trees = Dict{String, GrowthTree}()
    names = String[]
    for (idx, path) in enumerate(paths)
        name = "$(RIGHT_LEG_50UM_TREE_NAME)_seed$(idx)"
        tree = make_single_raw_seed_tree(path;
            terminal_diameter_cm=terminal_diameter_cm,
            root_diameter_cm=proximal_root_diameter_cm(path))
        tree.name = name
        trees[name] = tree
        push!(names, name)
    end
    return trees, names
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
        j = (div(idx0, nx) % ny) + 1
        k = div(idx0, nx * ny) + 1
        pts[row, 1] = domain.origin_cm[1] + (i - rand(rng)) * domain.spacing_cm[1]
        pts[row, 2] = domain.origin_cm[2] + (j - rand(rng)) * domain.spacing_cm[2]
        pts[row, 3] = domain.origin_cm[3] + (k - rand(rng)) * domain.spacing_cm[3]
    end
    return pts
end

function write_right_leg_full_summary(path::AbstractString, raw_path, organ_ids_path, log_path,
                                      xcat_info, keep_desc, kept_counts, crop_lo, crop_hi,
                                      domain, mask_info, route_tissue_voxels::Int,
                                      route_dilation_voxels::Int,
                                      growth_artery_paths, fixed_artery_paths, vein_paths,
                                      artery_overlay_segments::Int, vein_overlay_segments::Int,
                                      tree::GrowthTree, stats;
                                      terminal_um::Float64,
                                      target_branches::Int,
                                      crop_pad_voxels::Int,
                                      muscle_tissue_voxels::Int,
                                      muscle_label_count::Int,
                                      target_label_count::Int,
                                      growth_artery_count::Int,
                                      growth_artery_min_length_cm::Float64,
                                      fixed_artery_min_length_cm::Float64,
                                      fixed_vein_min_length_cm::Float64,
                                      fixed_vessel_radius_voxels::Int,
                                      route_repair_stats,
                                      branch_caps::Dict{String, Int},
                                      coverage_count::Int,
                                      graph_count::Int,
                                      frontier_batch::Int,
                                      graph_block_size::Int,
                                      min_frontier_separation_cm::Float64,
                                      max_segment_length_cm::Float64,
                                      graph_neighbors::Int,
                                      max_path_nodes::Int,
                                      graph_jitter_cm::Float64,
                                      snap_terminal_to_target::Bool,
                                      max_terminal_snap_cm::Float64,
                                      max_anchor_gap_cm::Float64,
                                      coverage_multiplier::Float64,
                                      use_indexed_anchor::Bool,
                                      use_astar_routing::Bool,
                                      frontier_candidate_factor::Int,
                                      started_at,
                                      finished_at)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    xcat_segments = count(tree.is_xcat)
    grown_segments = length(tree.segment_start) - xcat_segments
    growth_surfaces = Set(path.surface for path in growth_artery_paths)
    fixed_non_growth_artery_paths = [path for path in fixed_artery_paths if !(path.surface in growth_surfaces)]
    route_repair_values = collect(values(route_repair_stats))
    route_repair_count = sum((stat.repaired for stat in route_repair_values); init=0)
    route_repair_skipped = sum((stat.skipped for stat in route_repair_values); init=0)
    route_repair_max_shift_cm = isempty(route_repair_values) ? 0.0 :
        maximum(stat.max_shift_cm for stat in route_repair_values)
    open(path, "w") do io
        println(io, "# XCAT Corrected Right-Leg GPU Run")
        println(io)
        println(io, "- Started: $(started_at)")
        println(io, "- Finished: $(finished_at)")
        println(io, "- XCAT raw: $(raw_path)")
        println(io, "- Organ IDs: $(organ_ids_path)")
        println(io, "- XCAT log: $(log_path)")
        println(io, "- Artery source: voxelized raw label volume: arteries_rleg")
        println(io, "- Vein source: voxelized raw label volume: veins_rleg")
        println(io, "- Source dims: $(xcat_info.dims)")
        println(io, "- Source slice range: $(xcat_info.start_slice)-$(xcat_info.end_slice)")
        println(io, "- Source spacing: $(round.(Tuple(xcat_info.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Kept right-leg half: $(keep_desc)")
        println(io, "- leg_right half counts: lower-x=$(kept_counts[1]), upper-x=$(kept_counts[2])")
        println(io, "- Vessel-aware crop pad: $(crop_pad_voxels) voxel(s)")
        println(io, "- Crop lo index: $(crop_lo)")
        println(io, "- Crop hi index: $(crop_hi)")
        println(io, "- Cropped dims: $(size(domain.mask))")
        println(io, "- Foreground right-leg muscle/foot-muscle voxels: $(muscle_tissue_voxels)")
        println(io, "- Muscle label count: $(muscle_label_count)")
        println(io, "- Growth target mask: right-leg soft tissue, including muscle, leg/foot envelope, fat/adipose, skin, and subcutaneous labels; XCAT vessel labels are excluded from target sampling.")
        println(io, "- Growth target tissue voxels: $(mask_info.foreground)")
        println(io, "- Growth target label count: $(target_label_count)")
        println(io, "- Routing mask: cropped right-leg soft-tissue XCAT labels, box-dilated by $(route_dilation_voxels) voxel(s)")
        println(io, "- Routing tissue voxels: $(route_tissue_voxels)")
        println(io, "- Cropped extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        @printf(io, "- Fixed vessel proximity filter: kept path sections within %.1f mm of the soft target mask\n",
            MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM * 10.0)
        @printf(io, "- Active artery growth seed min length: %.2f mm\n", growth_artery_min_length_cm * 10.0)
        @printf(io, "- Fixed artery structural export min length: %.2f mm\n", fixed_artery_min_length_cm * 10.0)
        @printf(io, "- Fixed vein export min length: %.2f mm\n", fixed_vein_min_length_cm * 10.0)
        println(io, "- Fixed artery/vein route-mask radius: $(fixed_vessel_radius_voxels) voxel(s)")
        println(io, "- Terminal branch diameter: $(round(terminal_um; digits=3)) um")
        println(io, "- Target added branches: $(target_branches)")
        println(io, "- Independent artery growth seeds: $(growth_artery_count)")
        println(io, "- Growth seed paths are oriented with the higher-z endpoint as the proximal/root side.")
        println(io, "- Seed capacity uses the maximum diameter in the first 8 proximal points, not the full-path maximum.")
        println(io, "- Growth capacity is distributed by proximal diameter cubed; territory distance weighting uses proximal diameter, so tissue-volume prior scales with diameter cubed rather than diameter to the ninth.")
        println(io, "- Component reachability: enabled; tissue targets can be claimed only by seeds connected through the legal route graph.")
        println(io, "- Coverage targets are sampled from the soft-tissue target mask, not from muscle-only points.")
        println(io, "- Per-seed branch caps: " *
            join(["$(name)=$(branch_caps[name])" for name in sort(collect(keys(branch_caps)))], ", "))
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
        println(io, "- Maximum external anchor entry gap: $(round(max_anchor_gap_cm * 10; digits=3)) mm")
        println(io, "- Indexed anchor lookup: $(use_indexed_anchor)")
        println(io, "- A* routing: $(use_astar_routing)")
        println(io, "- Frontier candidate factor: $(frontier_candidate_factor)")
        @printf(io, "- Post-growth route repairs: %d repaired, %d skipped, max shift %.3f mm\n",
            route_repair_count, route_repair_skipped, route_repair_max_shift_cm * 10.0)
        println(io)
        println(io, "## XCAT Artery Growth Seeds")
        println(io)
        for artery_path in growth_artery_paths
            @printf(io, "- %s (growth seed): %d points, %.2f mm length, %.2f-%.2f mm diameter, %.2f mm proximal capacity diameter, root z %.2f cm\n",
                artery_path.surface, length(artery_path.points),
                path_length_cm(artery_path) * 10.0,
                minimum(artery_path.diameters_cm) * 10.0,
                maximum(artery_path.diameters_cm) * 10.0,
                proximal_root_diameter_cm(artery_path) * 10.0,
                first(artery_path.points)[3])
        end
        println(io, "- Fixed non-growth XCAT artery segments exported after growth: $(artery_overlay_segments)")
        println(io, "- All retained fixed artery paths are admitted to the route mask before growth. Non-growth arteries are fixed structural exports, not prediction seeds.")
        println(io, "- No forced connector segments were created between raw artery paths.")
        println(io)
        println(io, "## Fixed Non-growth XCAT Arteries")
        println(io)
        for artery_path in fixed_non_growth_artery_paths
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter, %.2f mm proximal capacity diameter\n",
                artery_path.surface, length(artery_path.points),
                path_length_cm(artery_path) * 10.0,
                minimum(artery_path.diameters_cm) * 10.0,
                maximum(artery_path.diameters_cm) * 10.0,
                proximal_root_diameter_cm(artery_path) * 10.0)
        end
        println(io)
        println(io, "## Fixed XCAT Veins")
        println(io)
        for vein_path in vein_paths
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter\n",
                vein_path.surface, length(vein_path.points),
                path_length_cm(vein_path) * 10.0,
                minimum(vein_path.diameters_cm) * 10.0,
                maximum(vein_path.diameters_cm) * 10.0)
        end
        println(io, "- Fixed vein segments exported after arterial growth: $(vein_overlay_segments)")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added predicted branches: $(stats.added)")
        println(io, "- XCAT structural artery seed + fixed vein segments exported: $(xcat_segments)")
        println(io, "- Predicted/grown segments exported: $(grown_segments)")
        println(io, "- Total segments exported: $(length(tree.segment_start))")
        println(io, "- Terminal vertices exported: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum displayed diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum displayed diameter: $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
        println(io, "- Coverage p50 before fixed-vein export: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95 before fixed-vein export: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max before fixed-vein export: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "This run uses corrected raw-label XCAT vessel extraction. Retained arteries are split into active growth seeds and fixed structural exports, and every retained fixed vessel is marked into the route mask before graph construction. Fixed XCAT veins are exported for downstream use, but they do not seed arterial growth because the arterial and venous trees are physiologically separate.")
    end
    return path
end

function main_right_leg_xcat_50um_gpu()
    Random.seed!(42)
    started_at = Dates.now()
    terminal_um = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
    output_dir = length(ARGS) >= 2 ? ARGS[2] : RIGHT_LEG_50UM_DEFAULT_OUTPUT_DIR
    raw_path = length(ARGS) >= 3 ? ARGS[3] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "both_legs_act_1.raw")
    organ_ids_path = length(ARGS) >= 4 ? ARGS[4] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "organ_ids.txt")
    log_path = length(ARGS) >= 5 ? ARGS[5] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "both_legs_log")
    target_arg = length(ARGS) >= 6 ? ARGS[6] : "auto"
    frontier_batch = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : RIGHT_LEG_50UM_FRONTIER_BATCH
    graph_block_size = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : RIGHT_LEG_50UM_GRAPH_BLOCK_SIZE
    min_frontier_separation_cm = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : RIGHT_LEG_50UM_MIN_FRONTIER_SEPARATION_CM
    max_segment_length_cm = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : RIGHT_LEG_50UM_MAX_SEGMENT_LENGTH_CM
    graph_neighbors = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : RIGHT_LEG_50UM_GRAPH_NEIGHBORS
    max_path_nodes = length(ARGS) >= 12 ? parse(Int, ARGS[12]) : RIGHT_LEG_50UM_MAX_PATH_NODES
    graph_jitter_cm = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : RIGHT_LEG_50UM_GRAPH_JITTER_CM
    snap_terminal_to_target = length(ARGS) >= 14 ? parse_bool_arg(ARGS[14]) : RIGHT_LEG_50UM_SNAP_TERMINAL_TO_TARGET
    max_terminal_snap_cm = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : RIGHT_LEG_50UM_MAX_TERMINAL_SNAP_CM
    coverage_multiplier = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : RIGHT_LEG_50UM_COVERAGE_MULTIPLIER
    use_indexed_anchor = length(ARGS) >= 17 ? parse_bool_arg(ARGS[17]) : RIGHT_LEG_50UM_USE_INDEXED_ANCHOR
    use_astar_routing = length(ARGS) >= 18 ? parse_bool_arg(ARGS[18]) : RIGHT_LEG_50UM_USE_ASTAR_ROUTING
    frontier_candidate_factor = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : RIGHT_LEG_50UM_FRONTIER_CANDIDATE_FACTOR
    fixed_vein_exports = length(ARGS) >= 20 ? parse(Int, ARGS[20]) : RIGHT_LEG_50UM_MAIN_VESSEL_OVERLAYS
    max_anchor_gap_cm = length(ARGS) >= 21 ? parse(Float64, ARGS[21]) : RIGHT_LEG_50UM_MAX_ANCHOR_GAP_CM
    requested_growth_artery_seeds = length(ARGS) >= 22 ? parse(Int, ARGS[22]) : RIGHT_LEG_50UM_GROWTH_ARTERY_SEEDS
    growth_artery_min_length_cm = length(ARGS) >= 23 ? parse(Float64, ARGS[23]) : RIGHT_LEG_50UM_GROWTH_ARTERY_MIN_LENGTH_CM
    fixed_artery_min_length_cm = length(ARGS) >= 24 ? parse(Float64, ARGS[24]) : RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM
    fixed_vein_min_length_cm = length(ARGS) >= 25 ? parse(Float64, ARGS[25]) : RIGHT_LEG_50UM_FIXED_VEIN_MIN_LENGTH_CM
    terminal_cm = terminal_um / 1e4

    mkpath(output_dir)
    xcat_info = parse_xcat_log(log_path)
    name_to_labels, _ = parse_organ_ids(organ_ids_path)
    raw_labels = load_xcat_uint16_raw(raw_path, xcat_info.dims)
    crop_pad_voxels = vessel_aware_crop_pad_voxels(xcat_info.spacing_cm;
        vessel_radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)

    right_leg_labels = labels_for(name_to_labels, ["leg_right"])
    keep_x, lower_count, upper_count, keep_desc = choose_right_half(raw_labels, xcat_info.dims, right_leg_labels)
    muscle_labels = right_leg_muscle_labels(name_to_labels)
    full_muscle_mask, muscle_counts = build_right_leg_muscle_mask(raw_labels, xcat_info.dims, muscle_labels, keep_x)
    soft_target_labels = right_leg_soft_target_labels(name_to_labels)
    full_target_mask, target_counts = build_right_leg_label_mask(raw_labels, xcat_info.dims, soft_target_labels, keep_x;
        description="Right-leg soft target")
    target_mask, origin_cm, crop_lo, crop_hi = crop_mask(full_target_mask, xcat_info.spacing_cm; pad=crop_pad_voxels)
    mask = crop_mask_to_bounds(full_muscle_mask, crop_lo, crop_hi)
    nhdr_path, mask_raw_path = write_right_leg_mask_artifacts(output_dir, mask, origin_cm, xcat_info.spacing_cm)
    target_nhdr_path, target_mask_raw_path = write_right_leg_target_mask_artifacts(output_dir, target_mask, origin_cm, xcat_info.spacing_cm)
    domain, mask_info = build_domain_from_mask(target_mask, origin_cm, xcat_info.spacing_cm)
    route_labels = right_leg_route_tissue_labels(name_to_labels)
    route_mask_base = build_right_leg_route_tissue_mask(raw_labels, xcat_info.dims, route_labels, keep_x, crop_lo, crop_hi)
    route_mask = dilate_mask_box(route_mask_base, RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS)

    artery_labels = labels_for(name_to_labels, ["arteries_rleg"])
    artery_paths_all = raw_label_centerline_paths(raw_labels, xcat_info.dims, artery_labels, keep_x,
        xcat_info.spacing_cm; label_prefix="arteries_rleg_raw_")
    artery_paths_near = clip_paths_to_mask_proximity(artery_paths_all, target_mask, origin_cm, xcat_info.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    oriented_artery_paths_near = [orient_right_leg_root_path(path) for path in artery_paths_near]
    fixed_artery_paths, fixed_artery_audit = select_vessel_paths(oriented_artery_paths_near;
        max_paths=0,
        min_length_cm=fixed_artery_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_artery")
    artery_paths, growth_artery_audit = select_vessel_paths(fixed_artery_paths;
        max_paths=requested_growth_artery_seeds,
        min_length_cm=growth_artery_min_length_cm,
        score=:length_mean_diameter,
        role="growth_artery")
    isempty(artery_paths) && error("No usable raw-label artery seed paths were selected")
    growth_artery_count = length(artery_paths)
    growth_artery_paths = artery_paths[1:growth_artery_count]
    growth_surfaces = Set(path.surface for path in growth_artery_paths)
    artery_overlay_paths = [path for path in fixed_artery_paths if !(path.surface in growth_surfaces)]

    vein_labels = labels_for(name_to_labels, ["veins_rleg"])
    vein_paths_all = raw_label_centerline_paths(raw_labels, xcat_info.dims, vein_labels, keep_x,
        xcat_info.spacing_cm; label_prefix="vein_raw_")
    vein_paths_near = clip_paths_to_mask_proximity(vein_paths_all, target_mask, origin_cm, xcat_info.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    vein_paths, vein_audit = select_vessel_paths(vein_paths_near;
        max_paths=fixed_vein_exports,
        min_length_cm=fixed_vein_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_vein")

    selected_fixed_paths = vcat(fixed_artery_paths, vein_paths)
    mark_paths_in_mask!(route_mask, selected_fixed_paths, origin_cm, xcat_info.spacing_cm;
        radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)
    route_mask .|= target_mask
    route_nhdr_path, route_mask_raw_path = write_right_leg_route_mask_artifacts(output_dir, route_mask, origin_cm, xcat_info.spacing_cm)
    route_domain, route_mask_info = build_domain_from_mask(route_mask, origin_cm, xcat_info.spacing_cm)

    trees, growth_tree_names = make_right_leg_growth_trees(growth_artery_paths; terminal_diameter_cm=terminal_cm)
    root_diameters = [trees[name].root_diameter_cm for name in growth_tree_names]
    capacity_weights = Dict(name => artery_weight(path) for (name, path) in zip(growth_tree_names, growth_artery_paths))
    territory_distance_weights = Dict(name => proximal_root_diameter_cm(path) for (name, path) in zip(growth_tree_names, growth_artery_paths))
    target_branches = lowercase(target_arg) == "auto" ?
        ceil(Int, (maximum(root_diameters) / terminal_cm)^3) :
        parse(Int, target_arg)
    branch_caps = branch_caps_from_weights(growth_tree_names,
        [capacity_weights[name] for name in growth_tree_names], target_branches)
    coverage_count = max(target_branches, ceil(Int, target_branches * coverage_multiplier))

    println("=" ^ 78)
    println("XCAT corrected right-leg GPU run")
    println("Started: $(started_at)")
    println("Raw: $(raw_path)")
    println("Output: $(output_dir)")
    println("Terminal diameter: $(terminal_um) um")
    requested_growth_artery_seeds <= 0 &&
        println("Growth seed count set to all eligible XCAT artery paths.")
    println("Growth seeds: " * join([path.surface for path in growth_artery_paths], ", "))
    println("Seed proximal capacity diameters: " * join(["$(round(trees[name].root_diameter_cm * 10; digits=3)) mm" for name in growth_tree_names], ", "))
    println("Target added branches: $(target_branches)")
    println("Per-seed branch caps: " * join(["$(name)=$(branch_caps[name])" for name in growth_tree_names], ", "))
    println("Coverage points: $(coverage_count)")
    println("Graph block size: $(graph_block_size)")
    println("Frontier batch: $(frontier_batch)")
    println("Minimum frontier separation: $(round(min_frontier_separation_cm * 10; digits=3)) mm")
    println("Maximum segment length: $(round(max_segment_length_cm * 10; digits=3)) mm")
    println("Snap terminal to target: $(snap_terminal_to_target)")
    println("Maximum terminal snap: $(round(max_terminal_snap_cm * 10; digits=3)) mm")
    println("Maximum external anchor entry gap: $(round(max_anchor_gap_cm * 10; digits=3)) mm")
    println("Indexed anchor lookup: $(use_indexed_anchor)")
    println("A* routing: $(use_astar_routing)")
    println("Frontier candidate factor: $(frontier_candidate_factor)")
    println("Growth artery seed min length: $(round(growth_artery_min_length_cm * 10; digits=3)) mm")
    println("Fixed artery export min length: $(round(fixed_artery_min_length_cm * 10; digits=3)) mm")
    println("Fixed vein export min length: $(round(fixed_vein_min_length_cm * 10; digits=3)) mm")
    println("Vessel-aware crop pad: $(crop_pad_voxels) voxel(s)")
    println("Kept half: $(keep_desc) lower=$(lower_count) upper=$(upper_count)")
    println("Cropped dims: $(size(target_mask)), muscle_foreground=$(count(mask)), muscle_labels=$(length(muscle_counts))")
    println("Soft target mask foreground=$(count(target_mask)) target_labels=$(length(target_counts))")
    println("Route tissue mask foreground=$(count(route_mask)) route_labels=$(length(route_labels)) dilation_voxels=$(RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS) fixed_vessel_radius_voxels=$(RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)")
    println("[artery paths] growth=$(length(growth_artery_paths)) fixed_non_growth=$(length(artery_overlay_paths)) fixed_total=$(length(fixed_artery_paths)) from $(length(artery_paths_all)) raw-space artery paths ($(length(artery_paths_near)) near-soft-target clipped paths)")
    println("[fixed veins] selected=$(length(vein_paths)) from $(length(vein_paths_all)) raw-space vein paths ($(length(vein_paths_near)) near-soft-target clipped paths)")
    println("[coverage] sampling $(coverage_count) random target points inside the soft target mask")
    flush(stdout)

    coverage_points = random_points_in_mask(domain, coverage_count; rng_seed=42)
    graph_points = coverage_target_points_blockwise(route_domain; block_size=graph_block_size)
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) frontier_batch=$(frontier_batch)")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, route_domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=terminal_cm,
        max_new_branches_per_tree=branch_caps,
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
        max_anchor_gap_cm=max_anchor_gap_cm,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        component_reachability=true,
        use_gpu=true,
        tree_weights=capacity_weights,
        territory_weights=territory_distance_weights)

    stats = growth_stats["__global__"]
    route_repair_stats = Dict{String, NamedTuple}()
    for name in growth_tree_names
        route_repair_stats[name] = repair_grown_segments_to_mask!(
            trees[name], route_mask, origin_cm, xcat_info.spacing_cm;
            max_radius_voxels=2,
            max_passes=3)
    end
    route_repair_total = sum((stat.repaired for stat in values(route_repair_stats)); init=0)
    route_repair_skipped = sum((stat.skipped for stat in values(route_repair_stats)); init=0)
    route_repair_max_shift_cm = isempty(route_repair_stats) ? 0.0 :
        maximum(stat.max_shift_cm for stat in values(route_repair_stats))
    println("[route repair] repaired=$(route_repair_total) skipped=$(route_repair_skipped) max_shift=$(round(route_repair_max_shift_cm * 10; digits=3)) mm")
    for (path, name) in zip(growth_artery_paths, growth_tree_names)
        restore_xcat_seed_diameters!(trees[name], [path])
    end
    tree = merge_growth_trees(RIGHT_LEG_50UM_TREE_NAME,
        [trees[name] for name in growth_tree_names]; terminal_diameter_cm=terminal_cm)
    artery_overlay_segments = append_fixed_overlays!(tree, artery_overlay_paths)
    vein_overlay_segments = append_vein_overlays!(tree, vein_paths)

    terminal_tag = terminal_tag_um(terminal_um)
    csv_path = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_segments.csv")
    xcat_fixed_csv = joinpath(output_dir, "xcat_right_leg_corrected_fixed_vessels.csv")
    raw_artery_csv = joinpath(output_dir, "xcat_right_leg_corrected_raw_artery_centerlines.csv")
    raw_vein_csv = joinpath(output_dir, "xcat_right_leg_corrected_raw_vein_centerlines.csv")
    growth_artery_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_growth_artery_path_audit.csv")
    fixed_artery_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_fixed_artery_path_audit.csv")
    vein_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_vein_path_audit.csv")
    summary_path = joinpath(output_dir, "summary.md")

    write_growth_csv(csv_path, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_xcat_seed_csv(xcat_fixed_csv, tree)
    write_xcat_paths_csv(raw_artery_csv, fixed_artery_paths)
    write_xcat_paths_csv(raw_vein_csv, vein_paths)
    write_vessel_path_audit_csv(growth_artery_audit_csv, growth_artery_audit;
        mask=route_mask, origin_cm=origin_cm, spacing_cm=xcat_info.spacing_cm,
        paths=fixed_artery_paths)
    write_vessel_path_audit_csv(fixed_artery_audit_csv, fixed_artery_audit;
        mask=route_mask, origin_cm=origin_cm, spacing_cm=xcat_info.spacing_cm,
        paths=oriented_artery_paths_near)
    write_vessel_path_audit_csv(vein_audit_csv, vein_audit;
        mask=route_mask, origin_cm=origin_cm, spacing_cm=xcat_info.spacing_cm,
        paths=vein_paths_near)
    write_right_leg_full_summary(summary_path, raw_path, organ_ids_path, log_path,
        xcat_info, keep_desc, (lower_count, upper_count), crop_lo, crop_hi,
        domain, mask_info, count(route_mask), RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS, growth_artery_paths, fixed_artery_paths, vein_paths, artery_overlay_segments,
        vein_overlay_segments, tree, stats;
        terminal_um=terminal_um,
        target_branches=target_branches,
        crop_pad_voxels=crop_pad_voxels,
        muscle_tissue_voxels=count(mask),
        muscle_label_count=length(muscle_counts),
        target_label_count=length(target_counts),
        growth_artery_count=growth_artery_count,
        growth_artery_min_length_cm=growth_artery_min_length_cm,
        fixed_artery_min_length_cm=fixed_artery_min_length_cm,
        fixed_vein_min_length_cm=fixed_vein_min_length_cm,
        fixed_vessel_radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS,
        route_repair_stats=route_repair_stats,
        branch_caps=branch_caps,
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
        max_anchor_gap_cm=max_anchor_gap_cm,
        coverage_multiplier=coverage_multiplier,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        started_at=started_at,
        finished_at=Dates.now())

    println()
    println("XCAT Corrected Right-Leg $(terminal_tag) Result")
    println("  added predicted branches = $(stats.added)")
    println("  xcat structural/fixed segments exported = $(count(tree.is_xcat))")
    println("  fixed non-growth artery segments = $(artery_overlay_segments)")
    println("  fixed vein segments       = $(vein_overlay_segments)")
    println("  total segments exported   = $(length(tree.segment_start))")
    println("  min diameter              = $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
    println("  max diameter              = $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
    println("  p95 coverage              = $(round(stats.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(nhdr_path)")
    println("  $(mask_raw_path)")
    println("  $(target_nhdr_path)")
    println("  $(target_mask_raw_path)")
    println("  $(csv_path)")
    println("  $(xcat_fixed_csv)")
    println("  $(raw_artery_csv)")
    println("  $(raw_vein_csv)")
    println("  $(growth_artery_audit_csv)")
    println("  $(fixed_artery_audit_csv)")
    println("  $(vein_audit_csv)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_right_leg_xcat_50um_gpu()
end
