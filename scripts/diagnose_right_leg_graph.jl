#!/usr/bin/env julia

using Dates
using Printf
using Statistics
using StaticArrays
using VascularTreeSim

include(joinpath(@__DIR__, "..", "examples", "right_leg_xcat_50um_gpu.jl"))

function _parse_int_list(value::AbstractString)
    parts = split(value, ',')
    out = Int[]
    for part in parts
        stripped = strip(part)
        isempty(stripped) && continue
        push!(out, parse(Int, stripped))
    end
    isempty(out) && error("Expected at least one integer block size")
    return out
end

function _component_summary(graph::DomainGraph)
    component, component_count = VascularTreeSim.graph_connected_components(graph)
    sizes = zeros(Int, component_count)
    for comp in component
        sizes[comp] += 1
    end
    largest = isempty(sizes) ? 0 : maximum(sizes)
    isolated = count(nbrs -> isempty(nbrs), graph.neighbors)
    edges = sum(length, graph.neighbors)
    return (
        components=component_count,
        largest_nodes=largest,
        largest_fraction=length(graph.points) == 0 ? 0.0 : largest / length(graph.points),
        isolated_nodes=isolated,
        directed_edges=edges,
        mean_degree=length(graph.points) == 0 ? 0.0 : edges / length(graph.points),
    )
end

function _build_right_leg_route_domain(nrb_path::AbstractString;
                                       nrb_voxel_spacing_cm::Float64,
                                       nrb_domain_samples::Tuple{Int, Int},
                                       route_dilation_voxels::Int,
                                       fixed_vein_exports::Int,
                                       fixed_artery_min_length_cm::Float64,
                                       fixed_vein_min_length_cm::Float64)
    surfaces = parse_xcat_grouped_nrb(nrb_path)
    soft_surfaces = right_leg_nrb_soft_surfaces(surfaces)
    route_domain = build_multipatch_voxel_domain_floodfill(soft_surfaces;
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE,
        voxel_spacing_cm=nrb_voxel_spacing_cm,
        outer_samples=nrb_domain_samples,
        dilation_radius=1)
    route_mask = copy(route_domain.mask)
    route_dilation_voxels > 0 && (route_mask = dilate_mask_box(route_mask, route_dilation_voxels))

    artery_paths_all = nrb_group_seed_paths_from_surfaces(surfaces, "arteries_rleg";
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE)
    artery_paths_near = clip_paths_to_mask_proximity(
        artery_paths_all, route_mask, route_domain.origin_cm, route_domain.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    oriented_artery_paths_near = [orient_right_leg_root_path(path) for path in artery_paths_near]
    fixed_artery_paths, _ = select_vessel_paths(oriented_artery_paths_near;
        max_paths=0,
        min_length_cm=fixed_artery_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_artery")

    vein_paths_all = nrb_group_seed_paths_from_surfaces(surfaces, "veins_rleg";
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE)
    vein_paths_near = clip_paths_to_mask_proximity(
        vein_paths_all, route_mask, route_domain.origin_cm, route_domain.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    vein_paths, _ = select_vessel_paths(vein_paths_near;
        max_paths=fixed_vein_exports,
        min_length_cm=fixed_vein_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_vein")

    mark_paths_in_mask!(route_mask, vcat(fixed_artery_paths, vein_paths),
        route_domain.origin_cm, route_domain.spacing_cm;
        radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)

    return VoxelShellDomain(route_mask, route_domain.origin_cm, route_domain.spacing_cm,
        route_domain.center_cm, route_domain.outer_surface_points,
        route_domain.outer_surface_normals, route_domain.cavity_surface_points,
        route_domain.cavity_surface_normals, route_domain.outer_query_grid,
        route_domain.cavity_query_grids)
end

function main()
    nrb_path = length(ARGS) >= 1 ? ARGS[1] :
        get(ENV, "VTS_XCAT_NRB_PATH", RIGHT_LEG_50UM_DEFAULT_NRB_PATH)
    output_dir = length(ARGS) >= 2 ? ARGS[2] : joinpath("output", "right_leg_graph_diagnostics")
    blocks = length(ARGS) >= 3 ? _parse_int_list(ARGS[3]) :
        _parse_int_list(get(ENV, "VTS_GRAPH_DIAGNOSTIC_BLOCKS", "12,8,6"))
    graph_neighbors = length(ARGS) >= 4 ? parse(Int, ARGS[4]) :
        parse(Int, get(ENV, "VTS_GRAPH_DIAGNOSTIC_K", "8"))
    nrb_voxel_spacing_cm = parse(Float64,
        get(ENV, "VTS_NRB_DOMAIN_VOXEL_SPACING_CM", string(RIGHT_LEG_50UM_NRB_DOMAIN_VOXEL_SPACING_CM)))
    nrb_domain_samples = parse_pair_arg(get(ENV, "VTS_NRB_DOMAIN_SAMPLES",
        "$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[1]),$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[2])"))
    route_dilation_voxels = parse(Int,
        get(ENV, "VTS_ROUTE_DILATION_VOXELS", string(RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS)))
    fixed_vein_exports = parse(Int,
        get(ENV, "VTS_GRAPH_DIAGNOSTIC_FIXED_VEINS", string(RIGHT_LEG_50UM_MAIN_VESSEL_OVERLAYS)))
    fixed_artery_min_length_cm = parse(Float64,
        get(ENV, "VTS_FIXED_ARTERY_MIN_LENGTH_CM", string(RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM)))
    fixed_vein_min_length_cm = parse(Float64,
        get(ENV, "VTS_FIXED_VEIN_MIN_LENGTH_CM", string(RIGHT_LEG_50UM_FIXED_VEIN_MIN_LENGTH_CM)))

    mkpath(output_dir)
    csv_path = joinpath(output_dir, "right_leg_graph_diagnostic.csv")
    println("[graph-diagnostic] started=$(Dates.now())")
    println("[graph-diagnostic] nrb=$(nrb_path)")
    println("[graph-diagnostic] blocks=$(join(blocks, ",")) k=$(graph_neighbors)")
    flush(stdout)

    t0 = time()
    route_domain = _build_right_leg_route_domain(nrb_path;
        nrb_voxel_spacing_cm=nrb_voxel_spacing_cm,
        nrb_domain_samples=nrb_domain_samples,
        route_dilation_voxels=route_dilation_voxels,
        fixed_vein_exports=fixed_vein_exports,
        fixed_artery_min_length_cm=fixed_artery_min_length_cm,
        fixed_vein_min_length_cm=fixed_vein_min_length_cm)
    println("[graph-diagnostic] route_mask dims=$(size(route_domain.mask)) foreground=$(count(route_domain.mask)) build_seconds=$(round(time() - t0; digits=2))")
    flush(stdout)

    open(csv_path, "w") do io
        println(io, "block_size,k,graph_points,build_seconds,components,largest_nodes,largest_fraction,isolated_nodes,directed_edges,mean_degree")
        flush(io)
        for block in blocks
            t_block = time()
            points = coverage_target_points_blockwise(route_domain; block_size=block)
            t_graph = time()
            graph = build_domain_graph(points, route_domain; k=graph_neighbors)
            graph_seconds = time() - t_graph
            summary = _component_summary(graph)
            total_seconds = time() - t_block
            @printf("[graph-diagnostic] block=%d points=%d graph_seconds=%.2f total_seconds=%.2f components=%d largest=%.2f%% isolated=%d mean_degree=%.2f\n",
                block, size(points, 1), graph_seconds, total_seconds, summary.components,
                100 * summary.largest_fraction, summary.isolated_nodes, summary.mean_degree)
            @printf(io, "%d,%d,%d,%.6f,%d,%d,%.8f,%d,%d,%.8f\n",
                block, graph_neighbors, size(points, 1), graph_seconds,
                summary.components, summary.largest_nodes, summary.largest_fraction,
                summary.isolated_nodes, summary.directed_edges, summary.mean_degree)
            flush(stdout)
            flush(io)
        end
    end
    println("[graph-diagnostic] wrote $(csv_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
