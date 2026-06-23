"""
    Main growth orchestration — competitive round-robin growth.

All tree names are config-driven (no hardcoded LAD/LCX/RCA).
Optimized with multi-threading and spatial segment indexing.
"""

_domain_points(domain::VoxelShellDomain) = coverage_target_points(domain; stride=1)

function _viewer_domain_points(domain::VoxelShellDomain; stride::Int=3)
    return coverage_target_points_blockwise(domain; block_size=max(stride, 1))
end

function _surface_sample_matrices(surface::XCATNurbsSurface; n_u::Int, n_v::Int, orient_outward::Bool)
    pts, nrms, _, _ = xcat_sample_surface(surface; n_u=n_u, n_v=n_v, orient_outward=orient_outward)
    n = length(pts)
    points = Matrix{Float64}(undef, n, 3)
    normals = Matrix{Float64}(undef, n, 3)
    k = 1
    for j in axes(pts, 1), i in axes(pts, 2)
        p = pts[j, i]
        nrm = nrms[j, i]
        points[k, 1] = p[1]; points[k, 2] = p[2]; points[k, 3] = p[3]
        normals[k, 1] = nrm[1]; normals[k, 2] = nrm[2]; normals[k, 3] = nrm[3]
        k += 1
    end
    return points, normals
end

# ── Territory assignment (competitive, THREADED) ──
#
# `global_min_dist` stores the EFFECTIVE distance (raw / territory_weight).
# A tree claims a point when its effective distance beats the incumbent's.
# Weight = 1.0 (default) reduces to pure-geometric ownership (original
# behavior); larger weight lets a tree claim more distant points, which
# produces weighted-Voronoi-like territory proportional to weight³ in 3D.
# Supply-radius and p95 thresholds are in raw distance, so callers must
# multiply by owner's weight when comparing to a physical threshold.

function _update_global_min_distances_threaded!(global_min_dist::Vector{Float64}, owner::Vector{Int},
                                                points_cm::Matrix{Float64}, seg_idx::SegmentSpatialIndex,
                                                tree_idx::Int; weight::Float64=1.0)
    n = size(points_cm, 1)
    inv_w = 1.0 / weight
    Threads.@threads for i in 1:n
        d_raw = _indexed_segment_distance(seg_idx, points_cm[i, 1], points_cm[i, 2], points_cm[i, 3])
        d_eff = d_raw * inv_w
        if d_eff < global_min_dist[i]
            global_min_dist[i] = d_eff
            owner[i] = tree_idx
        end
    end
end

# Fallback for incremental update with new segments only (SoA fast path)
function _update_global_min_distances_incremental!(global_min_dist::Vector{Float64}, owner::Vector{Int},
                                                    points_cm::Matrix{Float64}, seg_idx::SegmentSpatialIndex,
                                                    tree_idx::Int, seg_start::Int, seg_end::Int;
                                                    weight::Float64=1.0)
    n = size(points_cm, 1)
    inv_w = 1.0 / weight
    Threads.@threads for i in 1:n
        best_d = global_min_dist[i]
        best_owner = owner[i]
        for s in seg_start:seg_end
            d2 = _point_seg_dist2(seg_idx, s, points_cm[i, 1], points_cm[i, 2], points_cm[i, 3])
            d_raw = sqrt(d2)
            d_eff = d_raw * inv_w
            if d_eff < best_d
                best_d = d_eff
                best_owner = tree_idx
            end
        end
        global_min_dist[i] = best_d
        owner[i] = best_owner
    end
end

function _choose_competitive_frontiers(global_min_dist::Vector{Float64}, owner::AbstractVector{<:Integer},
                                        tree_idx::Int, points_cm::Matrix{Float64};
                                        max_targets::Int, min_separation_cm::Float64,
                                        effective_supply_radius_cm::Float64,
                                        weight::Float64=1.0,
                                        candidate_factor::Int=0)
    # global_min_dist is EFFECTIVE distance (raw * 1/weight_owner). Since all
    # points here have owner == tree_idx, multiply by `weight` to recover raw.
    # Supply-radius comparison is in raw cm.
    scored = Tuple{Float64, Int}[]
    return_limit = candidate_factor > 0 ? max(max_targets * candidate_factor, max_targets) : max_targets
    candidate_limit = candidate_factor > 0 ? return_limit : 0
    for i in eachindex(global_min_dist)
        owner[i] == tree_idx || continue
        d_raw = global_min_dist[i] * weight
        d_raw <= effective_supply_radius_cm && continue
        if candidate_limit == 0
            push!(scored, (d_raw, i))
        elseif length(scored) < candidate_limit
            _heap_push!(scored, (d_raw, i))
        elseif d_raw > scored[1][1]
            _heap_replace_min!(scored, (d_raw, i))
        end
    end
    sort!(scored, by=first, rev=true)
    chosen = Int[]
    sep2 = min_separation_cm * min_separation_cm
    chosen_grid = Dict{NTuple{3, Int}, Vector{SVector{3, Float64}}}()
    inv_sep = min_separation_cm > 0 ? 1.0 / min_separation_cm : Inf
    for (_, idx) in scored
        p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
        ok = true
        if min_separation_cm > 0
            key = (floor(Int, p[1] * inv_sep), floor(Int, p[2] * inv_sep), floor(Int, p[3] * inv_sep))
            for dz in -1:1, dy in -1:1, dx in -1:1
                cell = get(chosen_grid, (key[1] + dx, key[2] + dy, key[3] + dz), nothing)
                cell === nothing && continue
                for q in cell
                    if sum(abs2, p - q) < sep2
                        ok = false
                        break
                    end
                end
                ok || break
            end
            if ok
                push!(get!(chosen_grid, key, SVector{3, Float64}[]), p)
            end
        end
        if ok
            push!(chosen, idx)
            length(chosen) >= return_limit && break
        end
    end
    return chosen
end

function _choose_competitive_frontiers_by_tree(global_min_dist::Vector{Float64}, owner::AbstractVector{<:Integer},
                                               points_cm::Matrix{Float64};
                                               max_targets_by_tree::Vector{Int},
                                               min_separation_cm::Float64,
                                               effective_supply_radius_cm::Float64,
                                               weights::Vector{Float64},
                                               candidate_factor::Int=0)
    n_trees = length(max_targets_by_tree)
    scored_by_tree = [Tuple{Float64, Int}[] for _ in 1:n_trees]
    candidate_limits = [
        candidate_factor > 0 ? max(max_targets_by_tree[ti] * candidate_factor, max_targets_by_tree[ti]) : 0
        for ti in 1:n_trees
    ]

    @inbounds for i in eachindex(global_min_dist)
        ti = owner[i]
        (ti <= 0 || ti > n_trees) && continue
        max_targets = max_targets_by_tree[ti]
        max_targets <= 0 && continue
        d_raw = global_min_dist[i] * weights[ti]
        d_raw <= effective_supply_radius_cm && continue
        scored = scored_by_tree[ti]
        candidate_limit = candidate_limits[ti]
        if candidate_limit == 0
            push!(scored, (d_raw, i))
        elseif length(scored) < candidate_limit
            _heap_push!(scored, (d_raw, i))
        elseif d_raw > scored[1][1]
            _heap_replace_min!(scored, (d_raw, i))
        end
    end

    chosen_by_tree = [Int[] for _ in 1:n_trees]
    sep2 = min_separation_cm * min_separation_cm
    inv_sep = min_separation_cm > 0 ? 1.0 / min_separation_cm : Inf
    for ti in 1:n_trees
        max_targets = max_targets_by_tree[ti]
        max_targets <= 0 && continue
        scored = scored_by_tree[ti]
        sort!(scored, by=first, rev=true)
        chosen = chosen_by_tree[ti]
        return_limit = candidate_limits[ti] > 0 ? candidate_limits[ti] : max_targets
        chosen_grid = Dict{NTuple{3, Int}, Vector{SVector{3, Float64}}}()
        for (_, idx) in scored
            p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
            ok = true
            if min_separation_cm > 0
                key = (floor(Int, p[1] * inv_sep), floor(Int, p[2] * inv_sep), floor(Int, p[3] * inv_sep))
                for dz in -1:1, dy in -1:1, dx in -1:1
                    cell = get(chosen_grid, (key[1] + dx, key[2] + dy, key[3] + dz), nothing)
                    cell === nothing && continue
                    for q in cell
                        if sum(abs2, p - q) < sep2
                            ok = false
                            break
                        end
                    end
                    ok || break
                end
                if ok
                    push!(get!(chosen_grid, key, SVector{3, Float64}[]), p)
                end
            end
            if ok
                push!(chosen, idx)
                length(chosen) >= return_limit && break
            end
        end
    end
    return chosen_by_tree
end

function _coverage_graph_components(points_cm::Matrix{Float64}, sgrid::GraphSpatialGrid,
                                    graph_component::Vector{Int})
    point_component, _ = _coverage_graph_components_and_indices(points_cm, sgrid, graph_component)
    return point_component
end

function _coverage_graph_indices(points_cm::Matrix{Float64}, sgrid::GraphSpatialGrid)
    graph_indices = Vector{Int}(undef, size(points_cm, 1))
    Threads.@threads for i in axes(points_cm, 1)
        p = SVector(points_cm[i, 1], points_cm[i, 2], points_cm[i, 3])
        graph_idx, _ = _nearest_graph_index(sgrid, p)
        graph_indices[i] = graph_idx
    end
    return graph_indices
end

function _coverage_graph_components_and_indices(points_cm::Matrix{Float64}, sgrid::GraphSpatialGrid,
                                                graph_component::Vector{Int})
    point_component = Vector{Int}(undef, size(points_cm, 1))
    graph_indices = Vector{Int}(undef, size(points_cm, 1))
    Threads.@threads for i in axes(points_cm, 1)
        p = SVector(points_cm[i, 1], points_cm[i, 2], points_cm[i, 3])
        graph_idx, _ = _nearest_graph_index(sgrid, p)
        graph_indices[i] = graph_idx
        point_component[i] = graph_component[graph_idx]
    end
    return point_component, graph_indices
end

function _graph_component_diagnostics(graph::DomainGraph, component::Vector{Int}, component_count::Int)
    n_nodes = length(graph.points)
    sizes = zeros(Int, component_count)
    for comp in component
        sizes[comp] += 1
    end
    largest = isempty(sizes) ? 0 : maximum(sizes)
    isolated = count(isempty, graph.neighbors)
    edges = sum(length, graph.neighbors) ÷ 2
    return (
        components=component_count,
        largest_component_nodes=largest,
        largest_component_fraction=n_nodes == 0 ? NaN : largest / n_nodes,
        isolated_nodes=isolated,
        edges=edges,
    )
end

function _tree_graph_components(tree::GrowthTree, sgrid::GraphSpatialGrid,
                                graph_component::Vector{Int}, max_seed_gap_cm::Float64)
    comps = Set{Int}()
    best_idx = 1
    best_dist = Inf
    for p in tree.vertices
        graph_idx, dist = _nearest_graph_index(sgrid, p)
        if dist < best_dist
            best_dist = dist
            best_idx = graph_idx
        end
        dist <= max_seed_gap_cm && push!(comps, graph_component[graph_idx])
    end
    if isempty(comps)
        push!(comps, graph_component[best_idx])
    end
    return comps
end

function _enforce_component_reachability!(global_min_dist::Vector{Float64}, owner::AbstractVector{<:Integer},
                                          points_cm::Matrix{Float64}, point_components::Vector{Int},
                                          tree_components::Vector{Set{Int}},
                                          seg_indices::Dict{String, SegmentSpatialIndex},
                                          branch_names::Vector{String},
                                          territory_w_arr::Vector{Float64})
    corrected = 0
    unclaimed = 0
    n_trees = length(branch_names)
    @inbounds for i in eachindex(owner)
        current = owner[i]
        comp = point_components[i]
        if current > 0 && comp in tree_components[current]
            continue
        end

        best_eff = Inf
        best_owner = 0
        x = points_cm[i, 1]
        y = points_cm[i, 2]
        z = points_cm[i, 3]
        for ti in 1:n_trees
            comp in tree_components[ti] || continue
            d = _indexed_segment_distance(seg_indices[branch_names[ti]], x, y, z)
            d_eff = d / territory_w_arr[ti]
            if d_eff < best_eff
                best_eff = d_eff
                best_owner = ti
            end
        end

        if best_owner == 0
            global_min_dist[i] = Inf
            owner[i] = 0
            unclaimed += 1
        else
            global_min_dist[i] = best_eff
            owner[i] = best_owner
            corrected += 1
        end
    end
    return corrected, unclaimed
end

function _round_distance_stats!(raw_dists::Vector{Float64},
                                owner_counts::Vector{Int},
                                per_tree_dists::Vector{Vector{Float64}},
                                               global_min_dist::Vector{Float64},
                                               owner::AbstractVector{<:Integer},
                                               territory_weights::Vector{Float64})
    fill!(owner_counts, 0)
    for buf in per_tree_dists
        empty!(buf)
    end

    n_trees = length(territory_weights)
    current_max = -Inf
    valid_count = 0
    unclaimed = 0
    @inbounds for i in eachindex(global_min_dist)
        ti = owner[i]
        if !(1 <= ti <= n_trees)
            unclaimed += 1
            continue
        end
        d_raw = global_min_dist[i] * territory_weights[ti]
        valid_count += 1
        raw_dists[valid_count] = d_raw
        current_max = max(current_max, d_raw)
        owner_counts[ti] += 1
        push!(per_tree_dists[ti], d_raw)
    end

    if valid_count == 0
        global_q = [NaN, Inf]
        current_max = NaN
    else
        valid_dists = raw_dists[1:valid_count]
        global_q = Statistics.quantile!(valid_dists, [0.50, 0.95])
    end
    per_tree_p50 = Vector{Float64}(undef, n_trees)
    per_tree_p95 = Vector{Float64}(undef, n_trees)
    per_tree_max = Vector{Float64}(undef, n_trees)
    for ti in 1:n_trees
        buf = per_tree_dists[ti]
        if isempty(buf)
            per_tree_p50[ti] = NaN
            per_tree_p95[ti] = Inf
            per_tree_max[ti] = NaN
        else
            per_tree_max[ti] = maximum(buf)
            q = Statistics.quantile!(buf, [0.50, 0.95])
            per_tree_p50[ti] = q[1]
            per_tree_p95[ti] = q[2]
        end
    end
    return (
        p50=global_q[1],
        p95=global_q[2],
        max=current_max,
        per_tree_p50=per_tree_p50,
        per_tree_p95=per_tree_p95,
        per_tree_max=per_tree_max,
        unclaimed=unclaimed,
    )
end

function _segment_stays_in_domain(domain::VoxelShellDomain,
                                  a::SVector{3, Float64},
                                  b::SVector{3, Float64})
    point_in_domain(domain, a) || return false
    point_in_domain(domain, b) || return false
    dist = norm(b - a)
    step = max(minimum(domain.spacing_cm) / 3, 1e-6)
    n = max(1, ceil(Int, dist / step))
    for i in 1:(n - 1)
        t = i / n
        p = (1.0 - t) .* a .+ t .* b
        point_in_domain(domain, p) || return false
    end
    return true
end

# ── Main growth loop ──

function _path_length_from_anchor(anchor::SVector{3, Float64},
                                  points::Vector{SVector{3, Float64}})
    total = 0.0
    prev = anchor
    for p in points
        total += norm(p - prev)
        prev = p
    end
    return total
end

function _growth_poiseuille_resistance(length_cm::Float64, diameter_cm::Float64;
                                       viscosity_poise::Float64=0.035)
    (length_cm <= 0.0 || diameter_cm <= 0.0) && return 0.0
    radius_cm = 0.5 * diameter_cm
    return 8.0 * viscosity_poise * length_cm / (pi * radius_cm^4)
end

function _growth_path_resistance_to_vertex(tree::GrowthTree, vertex::Int;
                                           viscosity_poise::Float64=0.035)
    resistance = 0.0
    v = vertex
    while v != 0
        seg = tree.incoming_segment[v]
        seg == 0 && break
        a = tree.vertices[tree.segment_start[seg]]
        b = tree.vertices[tree.segment_end[seg]]
        resistance += _growth_poiseuille_resistance(norm(b - a), tree.segment_diameter_cm[seg];
            viscosity_poise=viscosity_poise)
        v = tree.parent_vertex[v]
    end
    return resistance
end

function _growth_vertex_path_resistances(tree::GrowthTree;
                                         viscosity_poise::Float64=0.035)
    path_resistance = zeros(Float64, length(tree.vertices))
    roots = [v for v in eachindex(tree.vertices) if tree.parent_vertex[v] == 0]
    for root in roots
        stack = copy(tree.children[root])
        while !isempty(stack)
            v = pop!(stack)
            seg = tree.incoming_segment[v]
            if seg != 0
                parent = tree.segment_start[seg]
                a = tree.vertices[parent]
                b = tree.vertices[v]
                path_resistance[v] = path_resistance[parent] +
                    _growth_poiseuille_resistance(norm(b - a), tree.segment_diameter_cm[seg];
                        viscosity_poise=viscosity_poise)
            end
            append!(stack, tree.children[v])
        end
    end
    return path_resistance
end

function _candidate_anchor_info(tree::GrowthTree, seg_idx::SegmentSpatialIndex,
                                point::SVector{3, Float64},
                                vertex_path_resistance::Vector{Float64};
                                use_indexed_anchor::Bool=false,
                                split_range=(0.2, 0.8),
                                viscosity_poise::Float64=0.035)
    if isempty(tree.segment_start)
        root = tree.root_vertex
        return tree.vertices[root], vertex_path_resistance[root]
    end

    seg_id, t, proj, _ = use_indexed_anchor ?
        _indexed_nearest_segment_projection(seg_idx, point) :
        _nearest_tree_segment_projection(tree, point)
    if seg_id == 0
        root = tree.root_vertex
        return tree.vertices[root], vertex_path_resistance[root]
    end

    start_v = tree.segment_start[seg_id]
    end_v = tree.segment_end[seg_id]
    if split_range[1] <= t <= split_range[2]
        a = tree.vertices[start_v]
        partial_len = norm(proj - a)
        partial_r = _growth_poiseuille_resistance(partial_len, tree.segment_diameter_cm[seg_id];
            viscosity_poise=viscosity_poise)
        return proj, vertex_path_resistance[start_v] + partial_r
    end

    ds = norm(point - tree.vertices[start_v])
    de = norm(point - tree.vertices[end_v])
    vid = ds <= de ? start_v : end_v
    return tree.vertices[vid], vertex_path_resistance[vid]
end

function _rank_frontiers_by_hydraulic_score(frontiers::Vector{Int}, tree::GrowthTree,
                                            seg_idx::SegmentSpatialIndex,
                                            points_cm::Matrix{Float64},
                                            global_min_dist::Vector{Float64},
                                            weight::Float64;
                                            batch::Int,
                                            rank_candidate_factor::Int,
                                            effective_supply_radius_cm::Float64,
                                            hydraulic_cost_weight::Float64,
                                            hydraulic_existing_path_weight::Float64,
                                            hydraulic_reference_resistance::Float64,
                                            blood_viscosity_poise::Float64,
                                            use_indexed_anchor::Bool)
    rank_candidate_factor <= 0 && return frontiers
    hydraulic_cost_weight <= 0.0 && return frontiers
    hydraulic_reference_resistance <= 0.0 && return frontiers
    isempty(frontiers) && return frontiers

    pool_n = min(length(frontiers), max(batch * rank_candidate_factor, batch))
    vertex_r = _growth_vertex_path_resistances(tree; viscosity_poise=blood_viscosity_poise)
    scored = Vector{Tuple{Float64, Float64, Int}}()
    sizehint!(scored, pool_n)
    @inbounds for pos in 1:pool_n
        idx = frontiers[pos]
        p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
        anchor_point, anchor_r = _candidate_anchor_info(tree, seg_idx, p, vertex_r;
            use_indexed_anchor=use_indexed_anchor,
            viscosity_poise=blood_viscosity_poise)
        route_len = max(norm(p - anchor_point), global_min_dist[idx] * weight)
        new_branch_r = _growth_poiseuille_resistance(route_len, max(tree.terminal_diameter_cm, eps(Float64));
            viscosity_poise=blood_viscosity_poise)
        new_branch_rel = new_branch_r / hydraulic_reference_resistance
        anchor_rel = anchor_r / hydraulic_reference_resistance
        weighted_cost_rel = new_branch_rel + hydraulic_existing_path_weight * anchor_rel
        coverage_gain_cm = max(global_min_dist[idx] * weight - effective_supply_radius_cm, 0.0)
        score = coverage_gain_cm / (1.0 + hydraulic_cost_weight * log1p(max(weighted_cost_rel, 0.0)))
        push!(scored, (score, coverage_gain_cm, idx))
    end
    sort!(scored, by=x -> (x[1], x[2]), rev=true)
    ranked = [idx for (_, _, idx) in scored]
    if pool_n < length(frontiers)
        append!(ranked, @view frontiers[(pool_n + 1):end])
    end
    return ranked
end

function _p95_or_nan(values::Vector{Float64})
    isempty(values) && return NaN
    return Statistics.quantile(copy(values), 0.95)
end

function _prune_growth_checkpoints!(checkpoint_dir::AbstractString, keep_last::Int)
    keep_last <= 0 && return
    files = filter(readdir(checkpoint_dir)) do name
        startswith(name, "growth_checkpoint_round") && endswith(name, ".jls")
    end
    sort!(files)
    excess = length(files) - keep_last
    excess <= 0 && return
    for name in files[1:excess]
        rm(joinpath(checkpoint_dir, name); force=true)
    end
end

function _write_growth_checkpoint!(checkpoint_dir::AbstractString,
                                   trees::Dict{String, GrowthTree};
                                   round_num::Int,
                                   total_added::Dict{String, Int},
                                   branch_caps::Dict{String, Int},
                                   current_p95::Float64,
                                   current_max::Float64,
                                   tree_best_p95::Dict{String, Float64},
                                   tree_stall::Dict{String, Int},
                                   tree_saturated::Dict{String, Bool},
                                   stall_rounds::Int,
                                   keep_last::Int)
    mkpath(checkpoint_dir)
    stamp = Dates.format(Dates.now(), "yyyymmdd_HHMMSS")
    base = "growth_checkpoint_round$(lpad(string(round_num), 6, '0'))_$(stamp).jls"
    tmp_path = joinpath(checkpoint_dir, base * ".tmp")
    final_path = joinpath(checkpoint_dir, base)
    payload = Dict{String, Any}(
        "schema_version" => 1,
        "created_at" => string(Dates.now()),
        "round" => round_num,
        "total_added" => copy(total_added),
        "branch_caps" => copy(branch_caps),
        "current_p95_cm" => current_p95,
        "current_max_cm" => current_max,
        "tree_best_p95" => copy(tree_best_p95),
        "tree_stall" => copy(tree_stall),
        "tree_saturated" => copy(tree_saturated),
        "stall_rounds" => stall_rounds,
        "tree_names" => sort(collect(keys(trees))),
        "total_segments" => sum(length(tree.segment_start) for tree in values(trees)),
        "total_vertices" => sum(length(tree.vertices) for tree in values(trees)),
        "trees" => trees,
    )
    open(tmp_path, "w") do io
        Serialization.serialize(io, payload)
    end
    mv(tmp_path, final_path; force=true)
    open(joinpath(checkpoint_dir, "latest_checkpoint.txt"), "w") do io
        println(io, final_path)
    end
    _prune_growth_checkpoints!(checkpoint_dir, keep_last)
    return final_path
end

function grow_trees_mcp!(trees::Dict{String, GrowthTree}, domain;
        effective_supply_radius_cm::Float64=1.25e-3,
        capillary_diameter_cm::Float64=8e-4,
        max_new_branches_per_tree::Union{Int, Dict{String, Int}}=120,
        graph_neighbors::Int=10,
        gamma::Float64=3.0,
        min_frontier_separation_cm::Float64=0.12,
        max_path_nodes::Int=20,
        target_p95_distance_cm::Float64=Inf,
        target_max_distance_cm::Float64=Inf,
        frontier_batch::Int=8,
        smooth_passes::Int=20,
        spline_density::Int=5,
        max_segment_length_cm::Float64=0.1,
        max_branch_length_cm::Float64=Inf,
        max_route_tortuosity::Float64=Inf,
        coverage_points_cm::Union{Nothing, Matrix{Float64}}=nothing,
        graph_points_cm::Union{Nothing, Matrix{Float64}}=nothing,
        use_gpu::Bool=gpu_available(),
        turn_penalty::Float64=0.5,
        graph_jitter_cm::Float64=-1.0,
        snap_terminal_to_target::Bool=false,
        max_terminal_snap_cm::Float64=Inf,
        max_anchor_gap_cm::Float64=-1.0,
        use_indexed_anchor::Bool=false,
        use_astar_routing::Bool=false,
        frontier_candidate_factor::Int=0,
        component_reachability::Bool=true,
        tree_weights::Union{Nothing, Dict{String, Float64}}=nothing,
        territory_weights::Union{Nothing, Dict{String, Float64}}=nothing,
        hydraulic_cost_weight::Float64=0.0,
        hydraulic_rank_candidate_factor::Int=0,
        hydraulic_reference_length_cm::Float64=1.0,
        hydraulic_existing_path_weight::Float64=0.0,
        min_hydraulic_score_cm::Float64=0.0,
        max_new_branch_resistance_rel::Float64=Inf,
        max_terminal_path_resistance_rel::Float64=Inf,
        blood_viscosity_poise::Float64=0.035,
        precompute_target_graph_indices::Bool=false,
        progress_csv_path::Union{Nothing, AbstractString}=nothing,
        checkpoint_dir::Union{Nothing, AbstractString}=nothing,
        checkpoint_interval_seconds::Real=Inf,
        checkpoint_keep::Int=2)

    points_cm = coverage_points_cm === nothing ? _domain_points(domain) : coverage_points_cm
    route_points_cm = graph_points_cm === nothing ? points_cm : graph_points_cm
    # Auto-jitter graph points to break grid alignment (default: 1/3 of domain spacing)
    if graph_jitter_cm < 0.0
        graph_jitter_cm = domain.spacing_cm[1] * 0.33
    end
    if graph_jitter_cm > 0.0
        route_points_cm = _jitter_points_in_domain(route_points_cm, domain; max_jitter_cm=graph_jitter_cm)
    end
    graph = build_domain_graph(route_points_cm, domain; k=graph_neighbors)
    sgrid = _build_graph_spatial_grid(graph)
    route_workspace = use_astar_routing ? RouteWorkspace(length(graph.points)) : nothing
    nt = Threads.nthreads()
    backend_str = use_gpu ? "GPU (CUDA)" : "CPU ($(nt) threads)"
    println("[growth] graph spatial grid ready — backend: $(backend_str)")
    flush(stdout)

    branch_names = sort(collect(keys(trees)))
    n_trees = length(branch_names)
    n_points = size(points_cm, 1)
    progress_csv = progress_csv_path === nothing ? nothing : String(progress_csv_path)
    checkpoint_dir_str = checkpoint_dir === nothing ? nothing : String(checkpoint_dir)
    checkpoint_interval = Float64(checkpoint_interval_seconds)
    checkpoint_enabled = checkpoint_dir_str !== nothing && isfinite(checkpoint_interval) && checkpoint_interval > 0.0
    checkpoint_keep = max(checkpoint_keep, 1)
    last_checkpoint_at = time()
    csv_cell(text) = "\"" * replace(String(text), "\"" => "\"\"") * "\""
    if progress_csv !== nothing
        dir = dirname(progress_csv)
        !isempty(dir) && mkpath(dir)
        open(progress_csv, "w") do io
            println(io, "round,round_seconds,total_added,global_p95_cm,global_max_cm,unclaimed_targets,per_tree_added,per_tree_p95,territory_counts,stall_rounds,hydraulic_rejected,new_branch_resistance_rel_p95,terminal_path_resistance_rel_p95")
        end
    end
    branch_caps = Dict{String, Int}()
    for name in branch_names
        cap = max_new_branches_per_tree isa Dict ?
            get(max_new_branches_per_tree, name, 0) :
            max_new_branches_per_tree
        branch_caps[name] = max(0, cap)
    end

    # Normalize per-tree growth weights. Each tree's per-round batch is
    # scaled by its normalized weight so that the total number of frontiers
    # considered per round stays ≈ n_trees × frontier_batch (total throughput
    # unchanged) but the split across trees reflects anatomical prior such
    # as target flow (Q ∝ d^3 ∝ territory volume → bigger tree deserves more
    # per-round growth).
    raw_weights = Float64[]
    for name in branch_names
        w = tree_weights === nothing ? 1.0 : get(tree_weights, name, 1.0)
        push!(raw_weights, max(w, 0.0))
    end
    total_raw_w = sum(raw_weights)
    if total_raw_w <= 0.0
        norm_weights = fill(1.0, n_trees)
    else
        # Normalize so weights sum to n_trees → per-tree factor ~1 on average
        norm_weights = raw_weights .* (n_trees / total_raw_w)
    end
    per_tree_batch = [max(1, round(Int, frontier_batch * w)) for w in norm_weights]
    tree_to_batch = Dict(branch_names[i] => per_tree_batch[i] for i in 1:n_trees)
    if tree_weights !== nothing
        println("[growth] tree_weights (normalized): " *
                join(["$(branch_names[i])=$(round(norm_weights[i]; digits=3)) → batch=$(per_tree_batch[i])" for i in 1:n_trees], " "))
        flush(stdout)
    end

    # Per-tree territory weights drive weighted-Voronoi ownership: a tree
    # with weight w claims points up to w times farther (in raw distance)
    # than a unit-weight competitor, yielding territory volume ∝ w^3.
    # Without territory_weights, all w=1 (geometric Voronoi = original).
    territory_w_arr = Float64[
        territory_weights === nothing ? 1.0 :
            max(get(territory_weights, name, 1.0), 1e-9)
        for name in branch_names
    ]
    # Normalize so max weight = 1, keeping effective distances in a sane range.
    max_w = maximum(territory_w_arr)
    if max_w > 0
        territory_w_arr ./= max_w
    end
    if territory_weights !== nothing
        println("[growth] territory_weights (normalized, max=1): " *
                join(["$(branch_names[i])=$(round(territory_w_arr[i]; digits=3))" for i in 1:n_trees], " "))
        flush(stdout)
    end

    hydraulic_cost_weight = max(hydraulic_cost_weight, 0.0)
    hydraulic_rank_candidate_factor = max(hydraulic_rank_candidate_factor, 0)
    hydraulic_existing_path_weight = max(hydraulic_existing_path_weight, 0.0)
    min_hydraulic_score_cm = max(min_hydraulic_score_cm, 0.0)
    hydraulic_reference_length_cm > 0.0 || error("hydraulic_reference_length_cm must be positive")
    blood_viscosity_poise > 0.0 || error("blood_viscosity_poise must be positive")
    hydraulic_reference_resistance = _growth_poiseuille_resistance(
        hydraulic_reference_length_cm, capillary_diameter_cm;
        viscosity_poise=blood_viscosity_poise)
    hydraulic_guards_enabled =
        hydraulic_cost_weight > 0.0 ||
        min_hydraulic_score_cm > 0.0 ||
        isfinite(max_new_branch_resistance_rel) ||
        isfinite(max_terminal_path_resistance_rel)
    if hydraulic_guards_enabled
        println("[growth] hydraulic guards: weight=$(hydraulic_cost_weight) rank_factor=$(hydraulic_rank_candidate_factor) reference_length_cm=$(hydraulic_reference_length_cm) existing_path_weight=$(hydraulic_existing_path_weight) min_score_cm=$(min_hydraulic_score_cm) max_new_rel=$(max_new_branch_resistance_rel) max_terminal_rel=$(max_terminal_path_resistance_rel)")
        flush(stdout)
    end
    hydraulic_ranking_enabled = hydraulic_cost_weight > 0.0 && hydraulic_rank_candidate_factor > 0
    hydraulic_ranking_enabled && frontier_candidate_factor <= 1 &&
        @warn "Hydraulic candidate ranking works best with frontier_candidate_factor > 1" frontier_candidate_factor hydraulic_rank_candidate_factor

    # Build segment spatial indices for each tree
    seg_indices = Dict{String, SegmentSpatialIndex}()
    for name in branch_names
        seg_indices[name] = build_segment_index(trees[name])
    end

    point_components = Int[]
    target_graph_indices = Int[]
    tree_components = Set{Int}[]
    graph_component_count = 0
    graph_diag = (
        components=1,
        largest_component_nodes=length(graph.points),
        largest_component_fraction=1.0,
        isolated_nodes=count(isempty, graph.neighbors),
        edges=sum(length, graph.neighbors) ÷ 2,
    )
    if component_reachability
        t_comp = time()
        graph_component, graph_component_count = graph_connected_components(graph)
        graph_diag = _graph_component_diagnostics(graph, graph_component, graph_component_count)
        println("[growth] graph connectivity: components=$(graph_diag.components) largest=$(graph_diag.largest_component_nodes)/$(length(graph.points)) ($(round(100 * graph_diag.largest_component_fraction; digits=2))%) isolated=$(graph_diag.isolated_nodes) edges=$(graph_diag.edges)")
        if graph_component_count <= 1
            component_reachability = false
            println("[growth] graph components=1 reachability filter not needed ($(round(time()-t_comp; digits=2))s)")
        else
            point_components, target_graph_indices = _coverage_graph_components_and_indices(points_cm, sgrid, graph_component)
            effective_seed_gap_cm = max_anchor_gap_cm < 0.0 ? max(4.0 * max_segment_length_cm, 0.25) : max_anchor_gap_cm
            tree_components = [
                _tree_graph_components(trees[name], sgrid, graph_component, effective_seed_gap_cm)
                for name in branch_names
            ]
            comp_summary = join([
                "$(branch_names[i])=$(length(tree_components[i]))"
                for i in eachindex(branch_names)
            ], " ")
            println("[growth] graph components=$(graph_component_count) reachability=$(round(time()-t_comp; digits=2))s tree_components=[$(comp_summary)]")
        end
        flush(stdout)
    end

    # ── Initialize global min distances ──
    if precompute_target_graph_indices && isempty(target_graph_indices)
        t_targets = time()
        target_graph_indices = _coverage_graph_indices(points_cm, sgrid)
        println("[growth] target graph indices precomputed ($(round(time()-t_targets; digits=2))s)")
        flush(stdout)
    elseif !isempty(target_graph_indices)
        println("[growth] target graph indices reused from reachability precompute")
        flush(stdout)
    end

    t_init = time()
    gpu_state = nothing
    gpu_component_reachability = false

    if use_gpu
        # GPU path: upload points once, run distance kernels on device
        gpu_state = _gpu_init_distance_state(points_cm)
        if component_reachability
            _gpu_set_component_reachability!(gpu_state, point_components, tree_components, graph_component_count)
            gpu_component_reachability = true
        end
        for (ti, name) in enumerate(branch_names)
            tree = trees[name]
            w_ti = territory_w_arr[ti]
            if isempty(tree.segment_start)
                _gpu_seed_distance!(gpu_state, tree.vertices[tree.root_vertex], ti; weight=w_ti)
            else
                _gpu_full_distance_scan!(gpu_state, seg_indices[name], ti; weight=w_ti)
            end
        end
        global_min_dist, owner = _gpu_download_distances(gpu_state)
    else
        # CPU path: threaded distance computation
        global_min_dist = fill(Inf, n_points)
        owner = fill(0, n_points)
        for (ti, name) in enumerate(branch_names)
            tree = trees[name]
            w_ti = territory_w_arr[ti]
            inv_w = 1.0 / w_ti
            if isempty(tree.segment_start)
                rv = tree.vertices[tree.root_vertex]
                Threads.@threads for i in 1:n_points
                    d_raw = sqrt((points_cm[i,1]-rv[1])^2 + (points_cm[i,2]-rv[2])^2 + (points_cm[i,3]-rv[3])^2)
                    d_eff = d_raw * inv_w
                    if d_eff < global_min_dist[i]
                        global_min_dist[i] = d_eff
                        owner[i] = ti
                    end
                end
            else
                _update_global_min_distances_threaded!(global_min_dist, owner, points_cm, seg_indices[name], ti; weight=w_ti)
            end
        end
    end

    if component_reachability && gpu_component_reachability
        println("[growth] component reachability enforced in GPU kernels unclaimed=$(count(==(0), owner))")
        flush(stdout)
    elseif component_reachability
        corrected, unclaimed = _enforce_component_reachability!(
            global_min_dist, owner, points_cm, point_components, tree_components,
            seg_indices, branch_names, territory_w_arr)
        println("[growth] component reachability corrected initial owners=$(corrected) unclaimed=$(unclaimed)")
        flush(stdout)
    end

    # Points with owner=0 (unclaimed — shouldn't happen after init) use raw=Inf.

    println("[growth] initial global distance scan: $(round(time()-t_init; digits=2))s  points=$(n_points)")
    flush(stdout)
    initial_owner_counts = zeros(Int, n_trees)
    for (ti, name) in enumerate(branch_names)
        initial_owner_counts[ti] = count(==(ti), owner)
        println("[growth] $(name) initial territory: $(initial_owner_counts[ti]) points")
    end
    flush(stdout)

    # ── Competitive round-robin growth ──
    total_added = Dict(name => 0 for name in branch_names)
    round_num = 0
    best_p95 = Inf
    stall_rounds = 0
    max_stall_rounds = 20   # stop if p95 hasn't improved in this many rounds
    raw_dists = Vector{Float64}(undef, n_points)
    owner_counts = zeros(Int, n_trees)
    per_tree_dist_buffers = [Float64[] for _ in 1:n_trees]
    for buf in per_tree_dist_buffers
        sizehint!(buf, max(1, cld(n_points, max(n_trees, 1))))
    end
    current_p50 = Inf
    current_p95 = Inf
    current_max = Inf
    current_unclaimed = n_points
    last_tree_p50 = fill(NaN, n_trees)
    last_tree_p95 = fill(Inf, n_trees)
    last_tree_max = fill(NaN, n_trees)

    # Per-tree saturation tracking. Each tree's p95 is computed over its own
    # owned points (raw distance). When a tree's p95 stalls for max_stall_rounds
    # rounds, that tree is marked saturated and skipped in subsequent rounds.
    # This prevents fast-territory trees (small Voronoi cell) from "catching
    # up" by stretching into already-filled regions while slow-territory trees
    # (big Voronoi cell) are still genuinely improving — the observed effect
    # that previously equalized terminal counts despite weighted-Voronoi
    # initial territory differences.
    tree_best_p95 = Dict(name => Inf for name in branch_names)
    tree_stall = Dict(name => 0 for name in branch_names)
    tree_saturated = Dict(name => false for name in branch_names)

    while true
        round_started_at = time()
        round_num += 1
        round_progress = false
        round_hydraulic_rejected = 0
        round_new_branch_resistance_rel = Float64[]
        round_terminal_path_resistance_rel = Float64[]
        round_batches = zeros(Int, n_trees)
        for (ti, name) in enumerate(branch_names)
            tree_saturated[name] && continue
            total_added[name] >= branch_caps[name] && continue
            remaining = branch_caps[name] - total_added[name]
            round_batches[ti] = min(tree_to_batch[name], remaining)
        end
        round_frontiers = _choose_competitive_frontiers_by_tree(
            global_min_dist, owner, points_cm;
            max_targets_by_tree=round_batches,
            min_separation_cm=min_frontier_separation_cm,
            effective_supply_radius_cm=effective_supply_radius_cm,
            weights=territory_w_arr,
            candidate_factor=frontier_candidate_factor)

        for (ti, name) in enumerate(branch_names)
            tree_saturated[name] && continue
            total_added[name] >= branch_caps[name] && continue
            tree = trees[name]
            batch = round_batches[ti]
            batch <= 0 && continue

            w_ti = territory_w_arr[ti]
            frontiers = round_frontiers[ti]
            isempty(frontiers) && continue
            if hydraulic_ranking_enabled && length(frontiers) > 1
                frontiers = _rank_frontiers_by_hydraulic_score(frontiers, tree, seg_indices[name],
                    points_cm, global_min_dist, w_ti;
                    batch=batch,
                    rank_candidate_factor=hydraulic_rank_candidate_factor,
                    effective_supply_radius_cm=effective_supply_radius_cm,
                    hydraulic_cost_weight=hydraulic_cost_weight,
                    hydraulic_existing_path_weight=hydraulic_existing_path_weight,
                    hydraulic_reference_resistance=hydraulic_reference_resistance,
                    blood_viscosity_poise=blood_viscosity_poise,
                    use_indexed_anchor=use_indexed_anchor)
            end

            seg_before = length(tree.segment_start)
            local_added = 0
            route_empty = 0
            prepared_empty = 0
            route_geometry_rejected = 0
            hydraulic_rejected = 0
            add_rejected = 0
            for idx in frontiers
                owner[idx] == ti || continue
                # global_min_dist is effective; compare raw against supply radius
                (global_min_dist[idx] * w_ti) <= effective_supply_radius_cm && continue
                p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
                if use_indexed_anchor
                    anchor_vertex, anchor_point = _choose_anchor_vertex_indexed!(tree, seg_indices[name], p)
                else
                    anchor_vertex, anchor_point = _choose_anchor_vertex(tree, p)
                end
                path_seg_before = length(tree.segment_start)
                source_idx, _ = _nearest_graph_index(sgrid, anchor_point)
                target_idx = isempty(target_graph_indices) ? first(_nearest_graph_index(sgrid, p)) : target_graph_indices[idx]
                path_ids = use_astar_routing ?
                    _shortest_path_astar!(route_workspace, graph, source_idx, target_idx; turn_penalty=turn_penalty) :
                    _shortest_path(graph, source_idx, target_idx; turn_penalty=turn_penalty)
                if isempty(path_ids)
                    route_empty += 1
                    continue
                end
                path_points = _prepare_branch_path([graph.points[i] for i in path_ids], domain;
                    max_nodes=max_path_nodes, smooth_passes=smooth_passes, spline_density=spline_density)
                if isempty(path_points)
                    prepared_empty += 1
                    continue
                end
                if snap_terminal_to_target && !isempty(path_points)
                    snap_distance = norm(p - path_points[end])
                    if snap_distance > 1e-8 &&
                            snap_distance <= max_terminal_snap_cm &&
                            _segment_stays_in_domain(domain, path_points[end], p)
                        push!(path_points, p)
                    end
                end
                route_length = _path_length_from_anchor(anchor_point, path_points)
                chord_length = norm(path_points[end] - anchor_point)
                route_tortuosity = chord_length > 1e-8 ? route_length / chord_length : 1.0
                if (isfinite(max_branch_length_cm) && route_length > max_branch_length_cm) ||
                        (isfinite(max_route_tortuosity) && route_tortuosity > max_route_tortuosity)
                    route_geometry_rejected += 1
                    continue
                end
                if hydraulic_guards_enabled && hydraulic_reference_resistance > 0.0
                    new_branch_resistance = _growth_poiseuille_resistance(
                        route_length, max(tree.terminal_diameter_cm, eps(Float64));
                        viscosity_poise=blood_viscosity_poise)
                    new_branch_rel = new_branch_resistance / hydraulic_reference_resistance
                    need_anchor_resistance = hydraulic_existing_path_weight > 0.0 ||
                        isfinite(max_terminal_path_resistance_rel)
                    anchor_rel = need_anchor_resistance ?
                        _growth_path_resistance_to_vertex(tree, anchor_vertex;
                            viscosity_poise=blood_viscosity_poise) / hydraulic_reference_resistance :
                        0.0
                    terminal_path_rel = anchor_rel + new_branch_rel
                    weighted_cost_rel = new_branch_rel + hydraulic_existing_path_weight * anchor_rel
                    coverage_gain_cm = max(global_min_dist[idx] * w_ti - effective_supply_radius_cm, 0.0)
                    hydraulic_score_cm = coverage_gain_cm /
                        (1.0 + hydraulic_cost_weight * log1p(max(weighted_cost_rel, 0.0)))
                    push!(round_new_branch_resistance_rel, new_branch_rel)
                    push!(round_terminal_path_resistance_rel, terminal_path_rel)
                    if new_branch_rel > max_new_branch_resistance_rel ||
                            terminal_path_rel > max_terminal_path_resistance_rel ||
                            hydraulic_score_cm < min_hydraulic_score_cm
                        hydraulic_rejected += 1
                        round_hydraulic_rejected += 1
                        continue
                    end
                end
                if _add_branch_path!(tree, anchor_vertex, path_points;
                        gamma=gamma,
                        max_branch_length_cm=max_branch_length_cm,
                        max_segment_length_cm=max_segment_length_cm,
                        domain=domain, max_anchor_gap_cm=max_anchor_gap_cm)
                    if use_indexed_anchor && length(tree.segment_start) > path_seg_before
                        seg_indices[name] = update_segment_index!(seg_indices[name], tree, path_seg_before + 1)
                    end
                    total_added[name] += 1
                    local_added += 1
                    round_progress = true
                    local_added >= batch && break
                    total_added[name] >= branch_caps[name] && break
                elseif use_indexed_anchor && length(tree.segment_start) > path_seg_before
                    seg_indices[name] = update_segment_index!(seg_indices[name], tree, path_seg_before + 1)
                else
                    add_rejected += 1
                end
            end
            if local_added == 0
                println("[growth-debug] $(name) round=$(round_num) frontiers=$(length(frontiers)) route_empty=$(route_empty) prepared_empty=$(prepared_empty) route_geometry_rejected=$(route_geometry_rejected) hydraulic_rejected=$(hydraulic_rejected) add_rejected=$(add_rejected)")
                flush(stdout)
            end

            # Incremental distance update
            seg_after = length(tree.segment_start)
            if seg_after > seg_before
                if !use_indexed_anchor
                    seg_indices[name] = update_segment_index!(seg_indices[name], tree, seg_before + 1)
                end
                if use_gpu && gpu_state !== nothing
                    # GPU incremental: scan only new segments on device
                    _gpu_incremental_scan!(gpu_state, seg_indices[name], ti, seg_before + 1, seg_after; weight=w_ti)
                    global_min_dist, owner = _gpu_download_distances(gpu_state)
                    if component_reachability && !gpu_component_reachability
                        _enforce_component_reachability!(
                            global_min_dist, owner, points_cm, point_components, tree_components,
                            seg_indices, branch_names, territory_w_arr)
                    end
                else
                    # CPU incremental: threaded scan of new segments
                    _update_global_min_distances_incremental!(global_min_dist, owner, points_cm,
                        seg_indices[name], ti, seg_before + 1, seg_after; weight=w_ti)
                    if component_reachability
                        _enforce_component_reachability!(
                            global_min_dist, owner, points_cm, point_components, tree_components,
                            seg_indices, branch_names, territory_w_arr)
                    end
                end
            end
        end

        # Status report (convert effective → raw for p95/max so physical
        # thresholds and stall detection remain in cm).
        dist_stats = _round_distance_stats!(raw_dists, owner_counts, per_tree_dist_buffers,
            global_min_dist, owner, territory_w_arr)
        current_p50 = dist_stats.p50
        current_p95 = dist_stats.p95
        current_max = dist_stats.max
        current_unclaimed = dist_stats.unclaimed
        last_tree_p50 = dist_stats.per_tree_p50
        last_tree_p95 = dist_stats.per_tree_p95
        last_tree_max = dist_stats.per_tree_max

        # Per-tree p95 + saturation update. Compute p95 over owned points
        # only, advance per-tree stall counter, mark tree saturated when its
        # local stall reaches the threshold.
        if n_trees == 1
            name = branch_names[1]
            tree_p95 = tree_saturated[name] ? tree_best_p95[name] : last_tree_p95[1]
            last_tree_p95[1] = tree_p95
            if !tree_saturated[name]
                if tree_p95 < tree_best_p95[name] - 1e-6
                    tree_best_p95[name] = tree_p95
                    tree_stall[name] = 0
                else
                    tree_stall[name] += 1
                end
                if round_num > 10 && tree_stall[name] >= max_stall_rounds
                    tree_saturated[name] = true
                    println("[growth] $(name) saturated: p95 stalled at $(round(tree_best_p95[name]; digits=5)) cm after $(total_added[name]) branches")
                    flush(stdout)
                end
            end
        else
            for (ti, name) in enumerate(branch_names)
                if tree_saturated[name]
                    last_tree_p95[ti] = tree_best_p95[name]
                    continue
                end
                tree_p95 = last_tree_p95[ti]
                if tree_p95 < tree_best_p95[name] - 1e-6
                    tree_best_p95[name] = tree_p95
                    tree_stall[name] = 0
                else
                    tree_stall[name] += 1
                end
                if round_num > 10 && tree_stall[name] >= max_stall_rounds
                    tree_saturated[name] = true
                    println("[growth] $(name) saturated: p95 stalled at $(round(tree_best_p95[name]; digits=5)) cm after $(total_added[name]) branches")
                    flush(stdout)
                end
            end
        end

        if round_num <= 3 || round_num % 5 == 0
            round_seconds = round(time() - round_started_at; digits=2)
            territory_counts = join(["$(branch_names[ti])=$(owner_counts[ti])" for ti in 1:n_trees], " ")
            added_str = join(["$(name)=$(total_added[name])" for name in branch_names], " ")
            tree_p95_str = join(["$(branch_names[ti])=$(round(last_tree_p95[ti]; digits=4))" for ti in 1:n_trees], " ")
            new_branch_rel_p95 = _p95_or_nan(round_new_branch_resistance_rel)
            terminal_path_rel_p95 = _p95_or_nan(round_terminal_path_resistance_rel)
            hydraulic_msg = hydraulic_guards_enabled ?
                " hydraulic_rejected=$(round_hydraulic_rejected) newR95=$(round(new_branch_rel_p95; digits=3)) pathR95=$(round(terminal_path_rel_p95; digits=3))" :
                ""
            println("[growth] round=$(round_num) seconds=$(round_seconds) added=[$(added_str)] global_p95=$(round(current_p95; digits=5)) tree_p95=[$(tree_p95_str)] territory=[$(territory_counts)]$(hydraulic_msg)")
            if progress_csv !== nothing
                open(progress_csv, "a") do io
                    println(io, join((
                        round_num,
                        round_seconds,
                        sum(values(total_added)),
                        current_p95,
                        current_max,
                        current_unclaimed,
                        csv_cell(added_str),
                        csv_cell(tree_p95_str),
                        csv_cell(territory_counts),
                        stall_rounds,
                        round_hydraulic_rejected,
                        new_branch_rel_p95,
                        terminal_path_rel_p95,
                    ), ","))
                end
            end
            flush(stdout)
        end

        # Stopping criteria
        all_maxed = all(total_added[name] >= branch_caps[name] for name in branch_names)
        p95_ok = isfinite(target_p95_distance_cm) && current_p95 <= target_p95_distance_cm
        max_ok = isfinite(target_max_distance_cm) && current_max <= target_max_distance_cm
        all_saturated = all(values(tree_saturated))

        # Global stall detection (fallback for unweighted runs / sanity check)
        if current_p95 < best_p95 - 1e-6
            best_p95 = current_p95
            stall_rounds = 0
        else
            stall_rounds += 1
        end
        coverage_stalled = stall_rounds >= max_stall_rounds

        if all_saturated
            println("[growth] STOP: all trees saturated (per-tree p95 stalled)")
            flush(stdout)
        elseif coverage_stalled && round_num > 10
            println("[growth] STOP: global p95 stalled at $(round(best_p95; digits=5)) cm for $(stall_rounds) rounds")
            flush(stdout)
        end

        if checkpoint_enabled && (time() - last_checkpoint_at) >= checkpoint_interval
            try
                checkpoint_path = _write_growth_checkpoint!(checkpoint_dir_str::String, trees;
                    round_num=round_num,
                    total_added=total_added,
                    branch_caps=branch_caps,
                    current_p95=current_p95,
                    current_max=current_max,
                    tree_best_p95=tree_best_p95,
                    tree_stall=tree_stall,
                    tree_saturated=tree_saturated,
                    stall_rounds=stall_rounds,
                    keep_last=checkpoint_keep)
                println("[checkpoint] wrote $(checkpoint_path) round=$(round_num) added=$(sum(values(total_added)))")
                flush(stdout)
            catch err
                println(stderr, "[checkpoint-warning] failed at round $(round_num):")
                showerror(stderr, err)
                println(stderr)
                flush(stderr)
            finally
                last_checkpoint_at = time()
            end
        end

        (all_saturated || all_maxed || (p95_ok && max_ok) || !round_progress || (coverage_stalled && round_num > 10)) && break
    end

    # Free GPU resources if used
    if gpu_state !== nothing
        _gpu_free!(gpu_state)
    end

    # Note: XCAT segment diameters are Murray-derived during growth (same as grown
    # segments). Only the root (ostium) segment retains its original XCAT diameter.
    # No post-processing junction smoothing needed.

    # Final per-tree stats (THREADED distance computation)
    territories = Dict(name => Int[] for name in branch_names)
    for i in 1:n_points
        ti = owner[i]
        ti > 0 && push!(territories[branch_names[ti]], i)
    end

    stats = Dict{String, NamedTuple}()
    total_branch_cap = sum(values(branch_caps))
    territory_prior = territory_w_arr .^ 3
    territory_prior_sum = sum(territory_prior)
    stats["__global__"] = (
        terminals=sum(length(_branch_terminals(trees[name])) for name in branch_names),
        p50=current_p50,
        p95=current_p95,
        max=current_max,
        added=sum(values(total_added)),
        graph_components=graph_diag.components,
        graph_largest_component_nodes=graph_diag.largest_component_nodes,
        graph_largest_component_fraction=graph_diag.largest_component_fraction,
        graph_isolated_nodes=graph_diag.isolated_nodes,
        graph_edges=graph_diag.edges,
        unclaimed_targets=current_unclaimed,
        coverage_points=n_points,
        branch_cap_total=total_branch_cap,
    )
    for (ti, name) in enumerate(branch_names)
        idxs = territories[name]
        println("[growth] $(name) finished added=$(total_added[name]) territory=$(length(idxs)) terminals=$(length(_branch_terminals(trees[name])))")
        flush(stdout)
        territory_points = length(idxs)
        stats[name] = (
            terminals=length(_branch_terminals(trees[name])),
            p50=last_tree_p50[ti],
            p95=last_tree_p95[ti],
            max=last_tree_max[ti],
            added=total_added[name],
            initial_territory_points=initial_owner_counts[ti],
            initial_territory_fraction=n_points > 0 ? initial_owner_counts[ti] / n_points : NaN,
            territory_points=territory_points,
            territory_fraction=n_points > 0 ? territory_points / n_points : NaN,
            branch_cap=branch_caps[name],
            branch_cap_fraction=total_branch_cap > 0 ? branch_caps[name] / total_branch_cap : NaN,
            tree_weight_raw=raw_weights[ti],
            tree_weight_normalized=norm_weights[ti],
            territory_weight_normalized=territory_w_arr[ti],
            territory_prior_fraction=territory_prior_sum > 0.0 ? territory_prior[ti] / territory_prior_sum : NaN,
            saturated=tree_saturated[name],
            stall_rounds=tree_stall[name],
        )
    end
    return graph, territories, stats
end

# ── Top-level run function ──

"""
    run_growth(config::OrganConfig; output_dir::String="output")

End-to-end growth pipeline:
1. Parse NRB surfaces
2. Build voxel shell domain from config
3. Build or seed vessel trees (based on config.growth_mode)
4. Run competitive growth
5. Export CSVs + viewer HTML
"""
function run_growth(config::OrganConfig; output_dir::String="output")
    mkpath(output_dir)

    println("[run_growth] parsing NRB: $(config.nrb_path)")
    flush(stdout)
    surfaces = parse_xcat_nrb(config.nrb_path)
    obj = xcat_object_dict(surfaces)

    println("[run_growth] building domain...")
    flush(stdout)
    outer_surface = obj[config.outer_surface]
    cavity_surface_list = [obj[name] for name in config.cavity_surfaces]
    domain = build_voxel_shell_domain_floodfill(outer_surface, cavity_surface_list;
        coordinate_scale=config.coordinate_scale,
        voxel_spacing_cm=config.voxel_spacing_cm,
        outer_samples=config.outer_samples,
        cavity_samples=config.cavity_samples,
        dilation_radius=config.dilation_radius,
        coarse_seed_cm=config.coarse_seed_cm)

    println("[run_growth] initializing vessel trees (mode=$(config.growth_mode))...")
    flush(stdout)
    growth_trees = Dict{String, GrowthTree}()

    if config.growth_mode == :continue_from_xcat
        all_vessel_surface_names = String[]
        !isempty(config.reference_surface) && push!(all_vessel_surface_names, config.reference_surface)
        for spec in config.vessel_trees
            append!(all_vessel_surface_names, spec.surface_names)
        end
        unique!(all_vessel_surface_names)
        vessel_surfaces = [obj[name] for name in all_vessel_surface_names if haskey(obj, name)]
        centerlines = [xcat_centerline_from_surface(s) for s in vessel_surfaces]
        xcat_trees = build_vessel_trees(centerlines, config)
        for spec in config.vessel_trees
            haskey(xcat_trees, spec.name) || continue
            growth_trees[spec.name] = growth_tree_from_xcat(spec.name, xcat_trees[spec.name];
                terminal_diameter_cm=config.terminal_diameter_cm)
        end
    elseif config.growth_mode == :seed_point
        for spec in config.vessel_trees
            haskey(config.seed_points, spec.name) && (growth_trees[spec.name] = growth_tree_from_seed(spec.name, config.seed_points[spec.name];
                terminal_diameter_cm=config.terminal_diameter_cm))
        end
    else
        error("Unknown growth mode: $(config.growth_mode)")
    end
    isempty(growth_trees) && error("No vessel trees were initialized. Check config.")

    coverage_block = max(config.coverage_stride, 1)
    route_block = config.graph_stride <= 0 ? coverage_block : config.graph_stride
    coverage_points = coverage_target_points_blockwise(domain; block_size=coverage_block)
    graph_points = coverage_target_points_blockwise(domain; block_size=route_block)
    graph_points = _jitter_points_in_domain(graph_points, domain; max_jitter_cm=config.graph_jitter_cm)
    println("[run_growth] coverage=$(size(coverage_points,1)) graph=$(size(graph_points,1)) points")
    flush(stdout)

    graph, territories, stats = grow_trees_mcp!(growth_trees, domain;
        coverage_points_cm=coverage_points, graph_points_cm=graph_points,
        effective_supply_radius_cm=config.effective_supply_radius_cm,
        capillary_diameter_cm=config.capillary_diameter_cm,
        max_new_branches_per_tree=config.max_new_branches_per_tree,
        graph_neighbors=config.graph_neighbors,
        min_frontier_separation_cm=config.min_frontier_separation_cm,
        max_path_nodes=config.max_path_nodes,
        frontier_batch=config.frontier_batch,
        gamma=config.murray_gamma,
        smooth_passes=config.smooth_passes,
        spline_density=config.spline_density,
        max_segment_length_cm=config.max_segment_length_cm,
        target_p95_distance_cm=config.target_p95_distance_cm,
        target_max_distance_cm=config.target_max_distance_cm,
        hydraulic_cost_weight=config.hydraulic_cost_weight,
        hydraulic_rank_candidate_factor=config.hydraulic_rank_candidate_factor,
        hydraulic_reference_length_cm=config.hydraulic_reference_length_cm,
        hydraulic_existing_path_weight=config.hydraulic_existing_path_weight,
        min_hydraulic_score_cm=config.min_hydraulic_score_cm,
        max_new_branch_resistance_rel=config.max_new_branch_resistance_rel,
        max_terminal_path_resistance_rel=config.max_terminal_path_resistance_rel,
        blood_viscosity_poise=config.blood_viscosity_poise,
        turn_penalty=config.turn_penalty,
        graph_jitter_cm=0.0)  # jitter already applied above from config

    println("[run_growth] exporting CSVs...")
    flush(stdout)
    color_map = Dict(spec.name => spec.color for spec in config.vessel_trees)
    for (name, tree) in growth_trees
        write_growth_csv(joinpath(output_dir, lowercase(name) * "_grown_segments.csv"), name, tree)
    end

    html_path = joinpath(output_dir, "index.html")
    growth_viewer_html(html_path, domain, growth_trees, stats, color_map)
    println("[run_growth] done. Viewer: $(html_path)")
    flush(stdout)

    return (html_path=html_path, domain=domain, coverage_points=coverage_points,
            trees=growth_trees, territories=territories, stats=stats)
end
