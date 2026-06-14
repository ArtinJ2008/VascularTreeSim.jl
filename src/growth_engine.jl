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

function _choose_competitive_frontiers(global_min_dist::Vector{Float64}, owner::Vector{Int},
                                        tree_idx::Int, points_cm::Matrix{Float64};
                                        max_targets::Int, min_separation_cm::Float64,
                                        effective_supply_radius_cm::Float64,
                                        weight::Float64=1.0)
    # global_min_dist is EFFECTIVE distance (raw * 1/weight_owner). Since all
    # points here have owner == tree_idx, multiply by `weight` to recover raw.
    # Supply-radius comparison is in raw cm.
    scored = Tuple{Float64, Int}[]
    for i in eachindex(global_min_dist)
        owner[i] == tree_idx || continue
        d_raw = global_min_dist[i] * weight
        d_raw <= effective_supply_radius_cm && continue
        push!(scored, (d_raw, i))
    end
    sort!(scored, by=first, rev=true)
    chosen = Int[]
    chosen_points = SVector{3, Float64}[]
    for (_, idx) in scored
        p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
        if all(norm(p - q) >= min_separation_cm for q in chosen_points)
            push!(chosen, idx)
            push!(chosen_points, p)
            length(chosen) >= max_targets && break
        end
    end
    return chosen
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

function grow_trees_mcp!(trees::Dict{String, GrowthTree}, domain;
        effective_supply_radius_cm::Float64=1.25e-3,
        capillary_diameter_cm::Float64=8e-4,
        max_new_branches_per_tree::Int=120,
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
        coverage_points_cm::Union{Nothing, Matrix{Float64}}=nothing,
        graph_points_cm::Union{Nothing, Matrix{Float64}}=nothing,
        use_gpu::Bool=gpu_available(),
        turn_penalty::Float64=0.5,
        graph_jitter_cm::Float64=-1.0,
        snap_terminal_to_target::Bool=false,
        max_terminal_snap_cm::Float64=Inf,
        tree_weights::Union{Nothing, Dict{String, Float64}}=nothing,
        territory_weights::Union{Nothing, Dict{String, Float64}}=nothing)

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
    nt = Threads.nthreads()
    backend_str = use_gpu ? "GPU (CUDA)" : "CPU ($(nt) threads)"
    println("[growth] graph spatial grid ready — backend: $(backend_str)")
    flush(stdout)

    branch_names = sort(collect(keys(trees)))
    n_trees = length(branch_names)
    n_points = size(points_cm, 1)

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

    # Build segment spatial indices for each tree
    seg_indices = Dict{String, SegmentSpatialIndex}()
    for name in branch_names
        seg_indices[name] = build_segment_index(trees[name])
    end

    # ── Initialize global min distances ──
    t_init = time()
    gpu_state = nothing

    if use_gpu
        # GPU path: upload points once, run distance kernels on device
        gpu_state = _gpu_init_distance_state(points_cm)
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

    # Helper: raw distance at point i (undoing the effective-distance storage).
    # Points with owner=0 (unclaimed — shouldn't happen after init) use raw=Inf.
    raw_dist_at(i) = owner[i] > 0 ? global_min_dist[i] * territory_w_arr[owner[i]] : Inf

    println("[growth] initial global distance scan: $(round(time()-t_init; digits=2))s  points=$(n_points)")
    flush(stdout)
    for (ti, name) in enumerate(branch_names)
        println("[growth] $(name) initial territory: $(count(==(ti), owner)) points")
    end
    flush(stdout)

    # ── Competitive round-robin growth ──
    total_added = Dict(name => 0 for name in branch_names)
    round_num = 0
    best_p95 = Inf
    stall_rounds = 0
    max_stall_rounds = 20   # stop if p95 hasn't improved in this many rounds

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
        round_num += 1
        round_progress = false

        for (ti, name) in enumerate(branch_names)
            tree_saturated[name] && continue
            total_added[name] >= max_new_branches_per_tree && continue
            tree = trees[name]
            remaining = max_new_branches_per_tree - total_added[name]
            batch = min(tree_to_batch[name], remaining)

            w_ti = territory_w_arr[ti]
            frontiers = _choose_competitive_frontiers(
                global_min_dist, owner, ti, points_cm;
                max_targets=batch, min_separation_cm=min_frontier_separation_cm,
                effective_supply_radius_cm=effective_supply_radius_cm,
                weight=w_ti)
            isempty(frontiers) && continue

            seg_before = length(tree.segment_start)
            local_added = 0
            for idx in frontiers
                # global_min_dist is effective; compare raw against supply radius
                (global_min_dist[idx] * w_ti) <= effective_supply_radius_cm && continue
                p = SVector(points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
                anchor_vertex, anchor_point = _choose_anchor_vertex(tree, p)
                source_idx, _ = _nearest_graph_index(sgrid, anchor_point)
                target_idx, _ = _nearest_graph_index(sgrid, p)
                path_ids = _shortest_path(graph, source_idx, target_idx; turn_penalty=turn_penalty)
                path_points = _prepare_branch_path([graph.points[i] for i in path_ids], domain;
                    max_nodes=max_path_nodes, smooth_passes=smooth_passes, spline_density=spline_density)
                if snap_terminal_to_target && !isempty(path_points)
                    snap_distance = norm(p - path_points[end])
                    if snap_distance > 1e-8 &&
                            snap_distance <= max_terminal_snap_cm &&
                            _segment_stays_in_domain(domain, path_points[end], p)
                        push!(path_points, p)
                    end
                end
                if _add_branch_path!(tree, anchor_vertex, path_points;
                        gamma=gamma,
                        max_branch_length_cm=Inf, max_segment_length_cm=max_segment_length_cm)
                    total_added[name] += 1
                    local_added += 1
                    round_progress = true
                    total_added[name] >= max_new_branches_per_tree && break
                end
            end

            # Incremental distance update
            seg_after = length(tree.segment_start)
            if seg_after > seg_before
                seg_indices[name] = update_segment_index!(seg_indices[name], tree, seg_before + 1)
                if use_gpu && gpu_state !== nothing
                    # GPU incremental: scan only new segments on device
                    _gpu_incremental_scan!(gpu_state, seg_indices[name], ti, seg_before + 1, seg_after; weight=w_ti)
                    global_min_dist, owner = _gpu_download_distances(gpu_state)
                else
                    # CPU incremental: threaded scan of new segments
                    _update_global_min_distances_incremental!(global_min_dist, owner, points_cm,
                        seg_indices[name], ti, seg_before + 1, seg_after; weight=w_ti)
                end
            end
        end

        # Status report (convert effective → raw for p95/max so physical
        # thresholds and stall detection remain in cm).
        raw_dists = [raw_dist_at(i) for i in 1:n_points]
        current_p95 = quantile(raw_dists, 0.95)
        current_max = maximum(raw_dists)

        # Per-tree p95 + saturation update. Compute p95 over owned points
        # only, advance per-tree stall counter, mark tree saturated when its
        # local stall reaches the threshold.
        per_tree_p95 = Float64[]
        for (ti, name) in enumerate(branch_names)
            tree_saturated[name] && (push!(per_tree_p95, tree_best_p95[name]); continue)
            owned_dists = Float64[]
            for i in 1:n_points
                owner[i] == ti && push!(owned_dists, raw_dists[i])
            end
            tree_p95 = isempty(owned_dists) ? Inf : quantile(owned_dists, 0.95)
            push!(per_tree_p95, tree_p95)
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

        if round_num <= 3 || round_num % 5 == 0
            territory_counts = join(["$(branch_names[ti])=$(count(==(ti), owner))" for ti in 1:n_trees], " ")
            added_str = join(["$(name)=$(total_added[name])" for name in branch_names], " ")
            tree_p95_str = join(["$(branch_names[ti])=$(round(per_tree_p95[ti]; digits=4))" for ti in 1:n_trees], " ")
            println("[growth] round=$(round_num) added=[$(added_str)] global_p95=$(round(current_p95; digits=5)) tree_p95=[$(tree_p95_str)] territory=[$(territory_counts)]")
            flush(stdout)
        end

        # Stopping criteria
        all_maxed = all(total_added[name] >= max_new_branches_per_tree for name in branch_names)
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
    for (ti, name) in enumerate(branch_names)
        idxs = territories[name]
        dists = Vector{Float64}(undef, length(idxs))
        idx_arr = seg_indices[name]
        Threads.@threads for j in eachindex(idxs)
            idx = idxs[j]
            dists[j] = _indexed_segment_distance(idx_arr, points_cm[idx, 1], points_cm[idx, 2], points_cm[idx, 3])
        end
        println("[growth] $(name) finished added=$(total_added[name]) territory=$(length(idxs)) terminals=$(length(_branch_terminals(trees[name])))")
        flush(stdout)
        stats[name] = (
            terminals=length(_branch_terminals(trees[name])),
            p50=isempty(dists) ? NaN : quantile(dists, 0.50),
            p95=isempty(dists) ? NaN : quantile(dists, 0.95),
            max=isempty(dists) ? NaN : maximum(dists),
            added=total_added[name],
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
