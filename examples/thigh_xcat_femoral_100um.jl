"""
    thigh_xcat_femoral_100um.jl

Legacy raw-label right-thigh femoral test helper. Direct execution is disabled:
current right-leg XCAT workflows are NRB-input-only. Helper functions remain
available to the NRB runner.

Usage:
    julia --project=. examples/right_leg_xcat_50um_gpu.jl [terminal_um] [output_dir] [nrb_path] ...
"""

include(joinpath(@__DIR__, "thigh_xcat_right_100um_test.jl"))

using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Statistics

const XCAT_FEMORAL_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_thigh_xcat_femoral_100um")
const DEFAULT_XCAT_FEMORAL_SEED_CSV = joinpath(dirname(@__DIR__), "output", "xcat_femoral_geometry", "xcat_femoral_trunk_candidates_thigh_raw_space.csv")
const DEFAULT_XCAT_FEMORAL_MAX_BRANCHES = 1000
const DEFAULT_XCAT_FEMORAL_TERMINAL_UM = 100.0

struct XCATSeedPath
    surface::String
    point_indices::Vector{Int}
    points::Vector{SVector{3, Float64}}
    diameters_cm::Vector{Float64}
end

function _csv_unquote(text::AbstractString)
    stripped = strip(String(text))
    if startswith(stripped, "\"") && endswith(stripped, "\"")
        return replace(stripped[2:end-1], "\"\"" => "\"")
    end
    return stripped
end

function load_xcat_seed_paths(path::AbstractString)
    isfile(path) || error("Missing XCAT femoral seed CSV: $path. Run examples/extract_xcat_femoral_geometry.jl first.")
    lines = readlines(path)
    length(lines) >= 2 || error("XCAT femoral seed CSV has no rows: $path")
    header = split(lines[1], ',')
    col = Dict(name => idx for (idx, name) in enumerate(header))
    required = ["surface", "point_index", "raw_x_cm", "raw_y_cm", "raw_z_cm", "diameter_cm"]
    for name in required
        haskey(col, name) || error("XCAT femoral seed CSV is missing column `$name`")
    end

    order = String[]
    indices = Dict{String, Vector{Int}}()
    points = Dict{String, Vector{SVector{3, Float64}}}()
    diameters = Dict{String, Vector{Float64}}()
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        surface = _csv_unquote(parts[col["surface"]])
        if !haskey(points, surface)
            push!(order, surface)
            indices[surface] = Int[]
            points[surface] = SVector{3, Float64}[]
            diameters[surface] = Float64[]
        end
        push!(indices[surface], parse(Int, parts[col["point_index"]]))
        push!(points[surface], SVector(
            parse(Float64, parts[col["raw_x_cm"]]),
            parse(Float64, parts[col["raw_y_cm"]]),
            parse(Float64, parts[col["raw_z_cm"]]),
        ))
        push!(diameters[surface], parse(Float64, parts[col["diameter_cm"]]))
    end

    paths = XCATSeedPath[]
    for surface in order
        perm = sortperm(indices[surface])
        sorted_points = points[surface][perm]
        sorted_diameters = diameters[surface][perm]
        keep = Int[]
        for idx in eachindex(sorted_points)
            if isempty(keep) || norm(sorted_points[idx] - sorted_points[keep[end]]) >= 1e-4
                push!(keep, idx)
            end
        end
        length(keep) >= 2 || continue
        push!(paths, XCATSeedPath(surface, indices[surface][perm][keep], sorted_points[keep], sorted_diameters[keep]))
    end
    !isempty(paths) || error("No usable XCAT femoral seed paths found in $path")
    return paths
end

function orient_root_path(path::XCATSeedPath)
    if last(path.diameters_cm) > first(path.diameters_cm)
        return XCATSeedPath(path.surface, reverse(path.point_indices), reverse(path.points), reverse(path.diameters_cm))
    end
    return path
end

function reverse_path(path::XCATSeedPath)
    XCATSeedPath(path.surface, reverse(path.point_indices), reverse(path.points), reverse(path.diameters_cm))
end

function nearest_vertex(vertices::Vector{SVector{3, Float64}}, point::SVector{3, Float64})
    best = 1
    best_d = Inf
    for (idx, p) in enumerate(vertices)
        d = norm(point - p)
        if d < best_d
            best = idx
            best_d = d
        end
    end
    return best, best_d
end

function _push_seed_segment!(vertices, parent_vertex, incoming_segment, children,
                             segment_start, segment_end, segment_diameter_cm,
                             segment_label, is_xcat, subtree_terminal_count,
                             parent::Int, point::SVector{3, Float64},
                             diameter_cm::Float64, label::String)
    if norm(point - vertices[parent]) < 1e-8
        return parent
    end
    push!(vertices, point)
    vid = length(vertices)
    push!(parent_vertex, parent)
    push!(incoming_segment, length(segment_start) + 1)
    push!(children, Int[])
    push!(subtree_terminal_count, 0)
    push!(segment_start, parent)
    push!(segment_end, vid)
    push!(segment_diameter_cm, diameter_cm)
    push!(segment_label, label)
    push!(is_xcat, true)
    push!(children[parent], vid)
    return vid
end

function append_seed_path!(tree_arrays, path::XCATSeedPath; max_connector_cm::Float64=2.0)
    vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm,
        segment_label, is_xcat, subtree_terminal_count = tree_arrays

    first_anchor, first_d = nearest_vertex(vertices, first(path.points))
    last_anchor, last_d = nearest_vertex(vertices, last(path.points))
    oriented = first_d <= last_d ? path : reverse_path(path)
    anchor, gap = first_d <= last_d ? (first_anchor, first_d) : (last_anchor, last_d)
    if gap > max_connector_cm
        @warn "Connecting XCAT patch with a larger gap than expected" surface=oriented.surface gap_cm=gap
    end

    prev = anchor
    start_idx = norm(first(oriented.points) - vertices[anchor]) < 1e-8 ? 2 : 1
    if start_idx == 1
        prev = _push_seed_segment!(vertices, parent_vertex, incoming_segment, children,
            segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
            subtree_terminal_count, anchor, first(oriented.points), first(oriented.diameters_cm), oriented.surface)
        start_idx = 2
    end
    for idx in start_idx:length(oriented.points)
        d_cm = 0.5 * (oriented.diameters_cm[idx - 1] + oriented.diameters_cm[idx])
        prev = _push_seed_segment!(vertices, parent_vertex, incoming_segment, children,
            segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
            subtree_terminal_count, prev, oriented.points[idx], d_cm, oriented.surface)
    end
    return gap
end

function make_xcat_femoral_seed_tree(paths::Vector{XCATSeedPath}; terminal_diameter_cm::Float64)
    root_idx = argmax([maximum(p.diameters_cm) for p in paths])
    root_path = orient_root_path(paths[root_idx])
    ordered = vcat([root_path], [paths[i] for i in eachindex(paths) if i != root_idx])

    vertices = SVector{3, Float64}[first(root_path.points)]
    parent_vertex = Int[0]
    incoming_segment = Int[0]
    children = [Int[]]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = Int[0]
    arrays = (vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat, subtree_terminal_count)

    prev = 1
    for idx in 2:length(root_path.points)
        d_cm = 0.5 * (root_path.diameters_cm[idx - 1] + root_path.diameters_cm[idx])
        prev = _push_seed_segment!(vertices, parent_vertex, incoming_segment, children,
            segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
            subtree_terminal_count, prev, root_path.points[idx], d_cm, root_path.surface)
    end

    connector_gaps = Dict{String, Float64}()
    for path in ordered[2:end]
        connector_gaps[path.surface] = append_seed_path!(arrays, path)
    end

    root_diameter_cm = maximum(segment_diameter_cm)
    tree = GrowthTree(
        "XCATFemoral100um",
        vertices,
        parent_vertex,
        incoming_segment,
        children,
        segment_start,
        segment_end,
        segment_diameter_cm,
        segment_label,
        is_xcat,
        subtree_terminal_count,
        terminal_diameter_cm,
        1,
        root_diameter_cm,
    )
    return tree, ordered, connector_gaps
end

function nearest_path_diameter_cm(path::XCATSeedPath, point::SVector{3, Float64})
    length(path.points) == 1 && return first(path.diameters_cm)
    best_d = Inf
    best_diameter = first(path.diameters_cm)
    for idx in 1:(length(path.points) - 1)
        a = path.points[idx]
        b = path.points[idx + 1]
        ab = b - a
        denom = dot(ab, ab)
        t = denom <= 1e-12 ? 0.0 : clamp(dot(point - a, ab) / denom, 0.0, 1.0)
        proj = a + t * ab
        d = norm(point - proj)
        if d < best_d
            best_d = d
            best_diameter = (1.0 - t) * path.diameters_cm[idx] + t * path.diameters_cm[idx + 1]
        end
    end
    return best_diameter
end

function restore_xcat_seed_diameters!(tree::GrowthTree, paths::Vector{XCATSeedPath})
    by_surface = Dict(path.surface => path for path in paths)
    for s in eachindex(tree.segment_start)
        tree.is_xcat[s] || continue
        path = get(by_surface, tree.segment_label[s], nothing)
        path === nothing && continue
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        # Keep the larger of the NRB-measured anatomy and the current (post-Murray)
        # diameter. An unconditional reset would shrink a distal XCAT segment that now
        # feeds a large grown subtree back below its Murray demand, recreating the
        # bottleneck that _recompute_all_murray!'s max() was meant to prevent (R ~ 1/r^4).
        tree.segment_diameter_cm[s] = max(tree.segment_diameter_cm[s],
            nearest_path_diameter_cm(path, 0.5 .* (a .+ b)))
    end
    return tree
end

function write_xcat_seed_csv(path::AbstractString, tree::GrowthTree)
    open(path, "w") do io
        println(io, "segment_id,surface,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,diameter_cm,diameter_mm")
        for s in eachindex(tree.segment_start)
            tree.is_xcat[s] || continue
            a = tree.vertices[tree.segment_start[s]]
            b = tree.vertices[tree.segment_end[s]]
            @printf(io, "%d,%s,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                s, tree.segment_label[s], a[1], a[2], a[3], b[1], b[2], b[3],
                tree.segment_diameter_cm[s], 10.0 * tree.segment_diameter_cm[s])
        end
    end
    return path
end

function write_xcat_femoral_summary(path::AbstractString, raw_path, organ_ids_path, log_path,
                                    seed_csv, xcat_info, keep_desc, kept_counts, crop_lo, crop_hi,
                                    domain, mask_info, seed_paths, connector_gaps, tree, stats;
                                    max_branches::Int, block_size::Int, terminal_um::Float64)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    fixed_segments = count(tree.is_xcat)
    open(path, "w") do io
        println(io, "# XCAT Femoral 100um Simulation")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- XCAT raw: $(raw_path)")
        println(io, "- Organ IDs: $(organ_ids_path)")
        println(io, "- XCAT log: $(log_path)")
        println(io, "- XCAT femoral seed CSV: $(seed_csv)")
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
        println(io)
        println(io, "The fixed seed path is sampled from the XCAT NRB-derived artery centerlines, transformed into the XCAT raw volume coordinate system, and clipped to the exported thigh slab.")
    end
    return path
end

function main_xcat_femoral_100um()
    Random.seed!(42)
    max_branches = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : DEFAULT_XCAT_FEMORAL_MAX_BRANCHES
    terminal_um = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : DEFAULT_XCAT_FEMORAL_TERMINAL_UM
    seed_csv = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_XCAT_FEMORAL_SEED_CSV
    output_dir = length(ARGS) >= 4 ? ARGS[4] : XCAT_FEMORAL_OUTPUT_DIR
    raw_path = length(ARGS) >= 5 ? ARGS[5] : DEFAULT_XCAT_RAW
    organ_ids_path = length(ARGS) >= 6 ? ARGS[6] : DEFAULT_XCAT_ORGAN_IDS
    log_path = length(ARGS) >= 7 ? ARGS[7] : DEFAULT_XCAT_LOG
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
    trees = Dict("XCATFemoral100um" => tree)

    coverage_points = coverage_target_points_blockwise(domain; block_size=DEFAULT_XCAT_BLOCK_SIZE)
    graph_points = coverage_target_points_blockwise(domain; block_size=DEFAULT_XCAT_BLOCK_SIZE)
    viewer_points = coverage_target_points_blockwise(domain; block_size=max(4, DEFAULT_XCAT_BLOCK_SIZE))

    println("=" ^ 78)
    println("XCAT femoral-seeded right-thigh 100um simulation")
    println("Raw: $(raw_path)")
    println("Seed CSV: $(seed_csv)")
    println("Output: $(output_dir)")
    println("Terminal diameter: $(terminal_um) um")
    println("Max branches: $(max_branches)")
    println("Cropped dims: $(size(mask)), foreground=$(count(mask))")
    surfaces_text = join([p.surface for p in ordered_seed_paths], ", ")
    println("[xcat seeds] surfaces=$(surfaces_text)")
    println("[xcat seeds] fixed_segments=$(length(tree.segment_start)) root_diameter=$(round(tree.root_diameter_cm * 10; digits=3)) mm")
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

    restore_xcat_seed_diameters!(tree, ordered_seed_paths)
    stats = growth_stats["XCATFemoral100um"]

    csv_path = joinpath(output_dir, "xcat_femoral_100um_segments.csv")
    seed_out_csv = joinpath(output_dir, "xcat_femoral_seed_segments.csv")
    domain_csv = joinpath(output_dir, "xcat_femoral_domain_points.csv")
    viewer_path = joinpath(output_dir, "xcat_femoral_100um_viewer.html")
    summary_path = joinpath(output_dir, "summary.md")

    write_growth_csv(csv_path, "XCATFemoral100um", tree)
    write_xcat_seed_csv(seed_out_csv, tree)
    write_points_csv(domain_csv, viewer_points)
    write_femoral_viewer(viewer_path, mask, origin_cm, xcat_info.spacing_cm, mask_info, viewer_points, tree)
    write_xcat_femoral_summary(summary_path, raw_path, organ_ids_path, log_path,
        seed_csv, xcat_info, keep_desc, (lower_count, upper_count), crop_lo, crop_hi,
        domain, mask_info, ordered_seed_paths, connector_gaps, tree, stats;
        max_branches=max_branches, block_size=DEFAULT_XCAT_BLOCK_SIZE, terminal_um=terminal_um)

    fixed_segments = count(tree.is_xcat)
    println()
    println("XCAT Femoral 100um Result")
    println("  added branches       = $(stats.added)")
    println("  xcat fixed segments  = $(fixed_segments)")
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
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    legacy_raw_entrypoint_error("examples/thigh_xcat_femoral_100um.jl")
end
