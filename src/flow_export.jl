"""
    Flow-oriented exports and audits.

The geometry CSV is useful for visualization, but flow solvers need a stricter
arterial network and diagnostics for path depth, resistance, and topology.
These helpers do not change the grown geometry. They create separate flow-ready
artifacts so distal beds can be lumped/pruned without hiding what the generator
actually produced.
"""

const DEFAULT_BLOOD_VISCOSITY_POISE = 0.035  # 3.5 cP
const DEFAULT_BLOOD_DENSITY_G_PER_CM3 = 1.06
const STANDARD_GRAVITY_CM_PER_S2 = 980.665
const MMHG_TO_DYN_PER_CM2 = 1333.22

_segment_length_cm(tree::GrowthTree, s::Int) =
    norm(tree.vertices[tree.segment_end[s]] - tree.vertices[tree.segment_start[s]])

function _segment_parent_ids(tree::GrowthTree)
    parent_ids = zeros(Int, length(tree.segment_start))
    for s in eachindex(tree.segment_start)
        start_v = tree.segment_start[s]
        parent_ids[s] = tree.incoming_segment[start_v]
    end
    return parent_ids
end

function _segment_role(tree::GrowthTree, s::Int)
    label = lowercase(tree.segment_label[s])
    if occursin("vein", label) || occursin("ven", label)
        return "vein"
    elseif tree.is_xcat[s]
        return "fixed_artery"
    else
        return "grown_artery"
    end
end

_is_vein_segment(tree::GrowthTree, s::Int) = _segment_role(tree, s) == "vein"

function _is_arterial_segment(tree::GrowthTree, s::Int;
                              include_fixed_arteries::Bool=true,
                              include_grown::Bool=true)
    _is_vein_segment(tree, s) && return false
    tree.is_xcat[s] ? include_fixed_arteries : include_grown
end

function _poiseuille_resistance(length_cm::Float64, diameter_cm::Float64;
                                viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    diameter_cm <= 0.0 && return Inf
    radius_cm = 0.5 * diameter_cm
    return 8.0 * viscosity_poise * length_cm / (pi * radius_cm^4)
end

function _children_segments(tree::GrowthTree; arterial_only::Bool=false)
    children = [Int[] for _ in eachindex(tree.vertices)]
    for s in eachindex(tree.segment_start)
        arterial_only && !_is_arterial_segment(tree, s) && continue
        push!(children[tree.segment_start[s]], s)
    end
    return children
end

function _tree_topology_metrics(tree::GrowthTree; arterial_only::Bool=false,
                                viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    nseg = length(tree.segment_start)
    nv = length(tree.vertices)
    parent_ids = _segment_parent_ids(tree)
    child_segments = _children_segments(tree; arterial_only=arterial_only)
    generation = zeros(Int, nseg)
    branchpoint_generation = zeros(Int, nseg)
    root_vertex = zeros(Int, nv)
    path_length_cm = zeros(Float64, nv)
    path_resistance = zeros(Float64, nv)

    root_vertices = [v for v in eachindex(tree.vertices) if tree.parent_vertex[v] == 0]
    stack = Tuple{Int, Int, Int}[]
    for root in root_vertices
        root_vertex[root] = root
        for s in child_segments[root]
            push!(stack, (s, 1, root))
        end
        while !isempty(stack)
            s, gen, root = pop!(stack)
            start_v = tree.segment_start[s]
            end_v = tree.segment_end[s]
            generation[s] = gen
            parent_seg = parent_ids[s]
            if parent_seg == 0
                branchpoint_generation[s] = 1
            else
                parent_branch_gen = branchpoint_generation[parent_seg]
                parent_branch_gen == 0 && (parent_branch_gen = 1)
                # Degree-2 vertices are polyline samples along the same vessel,
                # not new physiological generations. Increment only when a
                # segment leaves a true branchpoint.
                branchpoint_generation[s] = parent_branch_gen +
                    (length(child_segments[start_v]) == 1 ? 0 : 1)
            end
            root_vertex[end_v] = root
            len_cm = _segment_length_cm(tree, s)
            path_length_cm[end_v] = path_length_cm[start_v] + len_cm
            path_resistance[end_v] = path_resistance[start_v] +
                _poiseuille_resistance(len_cm, tree.segment_diameter_cm[s];
                    viscosity_poise=viscosity_poise)
            for child_s in child_segments[end_v]
                push!(stack, (child_s, gen + 1, root))
            end
        end
    end

    vertex_order = ones(Int, nv)
    visited = falses(nv)
    function strahler_vertex(v::Int)
        visited[v] && return vertex_order[v]
        orders = Int[]
        for s in child_segments[v]
            push!(orders, strahler_vertex(tree.segment_end[s]))
        end
        if isempty(orders)
            vertex_order[v] = 1
        else
            max_order = maximum(orders)
            vertex_order[v] = count(==(max_order), orders) >= 2 ? max_order + 1 : max_order
        end
        visited[v] = true
        return vertex_order[v]
    end
    for root in root_vertices
        strahler_vertex(root)
    end
    segment_order = [vertex_order[tree.segment_end[s]] for s in 1:nseg]

    return (
        parent_ids=parent_ids,
        generation=generation,
        branchpoint_generation=branchpoint_generation,
        strahler_order=segment_order,
        root_vertex=root_vertex,
        path_length_cm=path_length_cm,
        path_resistance=path_resistance,
        children_segments=child_segments,
    )
end

function _csv_text(value)
    text = string(value)
    if occursin(",", text) || occursin("\"", text) || occursin("\n", text)
        return "\"" * replace(text, "\"" => "\"\"") * "\""
    end
    return text
end

function _path_chain_stats(tree::GrowthTree, path_segments::Vector{Int}, metrics)
    isempty(path_segments) && return (0, 0, 0)
    degree2_count = 0
    current_run = 0
    max_run = 0
    for s in path_segments
        current_run += 1
        # `path_segments` is root-to-terminal. A segment remains in the same
        # unbranched chain when its end vertex has exactly one arterial child.
        end_v = tree.segment_end[s]
        if length(metrics.children_segments[end_v]) == 1
            degree2_count += 1
        else
            max_run = max(max_run, current_run)
            current_run = 0
        end
    end
    max_run = max(max_run, current_run)
    return (length(path_segments), degree2_count, max_run)
end

function _write_segment_row(io, branch::String, tree::GrowthTree, s::Int, parent_seg_id::Int)
    a = tree.vertices[tree.segment_start[s]]
    b = tree.vertices[tree.segment_end[s]]
    length_cm = norm(b - a)
    diameter_cm = tree.segment_diameter_cm[s]
    println(io, join((
        branch, s, parent_seg_id,
        a[1], a[2], a[3],
        b[1], b[2], b[3],
        (a[1] + b[1]) / 2,
        (a[2] + b[2]) / 2,
        (a[3] + b[3]) / 2,
        10.0 * length_cm,
        1.0e4 * diameter_cm,
        tree.segment_label[s],
    ), ","))
end

function write_arterial_growth_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                   include_fixed_arteries::Bool=true,
                                   include_grown::Bool=true)
    parent_ids = _segment_parent_ids(tree)
    open(path, "w") do io
        println(io, "branch,segment_id,parent_segment_id,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,xmid_cm,ymid_cm,zmid_cm,length_mm,diameter_um,label")
        for s in eachindex(tree.segment_start)
            _is_arterial_segment(tree, s;
                include_fixed_arteries=include_fixed_arteries,
                include_grown=include_grown) || continue
            parent = parent_ids[s]
            parent > 0 && !_is_arterial_segment(tree, parent;
                include_fixed_arteries=include_fixed_arteries,
                include_grown=include_grown) && (parent = 0)
            _write_segment_row(io, branch, tree, s, parent)
        end
    end
    return path
end

function write_flow_topology_audit_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                       arterial_only::Bool=true,
                                       min_diameter_um::Float64=0.0,
                                       viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    metrics = _tree_topology_metrics(tree;
        arterial_only=arterial_only,
        viscosity_poise=viscosity_poise)
    min_diameter_cm = max(0.0, min_diameter_um / 1.0e4)
    open(path, "w") do io
        println(io, "branch,segment_id,parent_segment_id,start_vertex,end_vertex,root_vertex,role,label,is_xcat,generation,branchpoint_generation,strahler_order,child_segments,subtree_terminals,length_mm,diameter_um,length_to_diameter,path_length_mm,path_resistance_abs,orphan_grown_root")
        for s in eachindex(tree.segment_start)
            arterial_only && !_is_arterial_segment(tree, s) && continue
            start_v = tree.segment_start[s]
            end_v = tree.segment_end[s]
            len_cm = _segment_length_cm(tree, s)
            d_cm = tree.segment_diameter_cm[s]
            min_diameter_cm > 0.0 && !tree.is_xcat[s] && d_cm < min_diameter_cm && continue
            parent = metrics.parent_ids[s]
            role = _segment_role(tree, s)
            orphan_grown_root = !tree.is_xcat[s] && parent == 0
            println(io, join((
                branch,
                s,
                parent,
                start_v,
                end_v,
                metrics.root_vertex[end_v],
                role,
                _csv_text(tree.segment_label[s]),
                tree.is_xcat[s],
                metrics.generation[s],
                metrics.branchpoint_generation[s],
                metrics.strahler_order[s],
                length(metrics.children_segments[end_v]),
                tree.subtree_terminal_count[end_v],
                10.0 * len_cm,
                1.0e4 * d_cm,
                d_cm > 0 ? len_cm / d_cm : Inf,
                10.0 * metrics.path_length_cm[end_v],
                metrics.path_resistance[end_v],
                orphan_grown_root,
            ), ","))
        end
    end
    return path
end

function write_terminal_path_audit_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                       arterial_only::Bool=true,
                                       max_rows::Union{Nothing, Int}=nothing,
                                       include_path_segments::Bool=true,
                                       viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    if max_rows !== nothing && max_rows <= 0
        open(path, "w") do io
            println(io, "branch,terminal_vertex,root_vertex,incoming_segment,role,label,generation,branchpoint_generation,strahler_order,path_length_mm,path_resistance_abs,min_path_diameter_um,max_path_diameter_um,path_segment_count,degree2_segment_count,max_degree2_chain_segments,path_segments")
        end
        return path
    end

    metrics = _tree_topology_metrics(tree;
        arterial_only=arterial_only,
        viscosity_poise=viscosity_poise)
    total_terminals = 0
    for v in eachindex(tree.vertices)
        isempty(metrics.children_segments[v]) || continue
        incoming = tree.incoming_segment[v]
        incoming == 0 && continue
        arterial_only && !_is_arterial_segment(tree, incoming) && continue
        total_terminals += 1
    end
    sample_stride = max_rows === nothing ? 1 :
        max(1, ceil(Int, total_terminals / max(max_rows, 1)))
    terminal_index = 0
    written = 0

    open(path, "w") do io
        println(io, "branch,terminal_vertex,root_vertex,incoming_segment,role,label,generation,branchpoint_generation,strahler_order,path_length_mm,path_resistance_abs,min_path_diameter_um,max_path_diameter_um,path_segment_count,degree2_segment_count,max_degree2_chain_segments,path_segments")
        for v in eachindex(tree.vertices)
            isempty(metrics.children_segments[v]) || continue
            incoming = tree.incoming_segment[v]
            incoming == 0 && continue
            arterial_only && !_is_arterial_segment(tree, incoming) && continue
            terminal_index += 1
            ((terminal_index - 1) % sample_stride == 0) || continue
            if max_rows !== nothing && written >= max_rows
                break
            end
            path_segments = Int[]
            min_d = Inf
            max_d = 0.0
            cursor = incoming
            while cursor != 0
                if !arterial_only || _is_arterial_segment(tree, cursor)
                    push!(path_segments, cursor)
                    d_um = 1.0e4 * tree.segment_diameter_cm[cursor]
                    min_d = min(min_d, d_um)
                    max_d = max(max_d, d_um)
                end
                cursor = metrics.parent_ids[cursor]
            end
            reverse!(path_segments)
            path_segment_count, degree2_segment_count, max_degree2_chain_segments =
                _path_chain_stats(tree, path_segments, metrics)
            println(io, join((
                branch,
                v,
                metrics.root_vertex[v],
                incoming,
                _segment_role(tree, incoming),
                _csv_text(tree.segment_label[incoming]),
                metrics.generation[incoming],
                metrics.branchpoint_generation[incoming],
                metrics.strahler_order[incoming],
                10.0 * metrics.path_length_cm[v],
                metrics.path_resistance[v],
                min_d,
                max_d,
                path_segment_count,
                degree2_segment_count,
                max_degree2_chain_segments,
                include_path_segments ? _csv_text(join(path_segments, ";")) : "",
            ), ","))
            written += 1
        end
    end
    return path
end

function write_root_territory_audit_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                        arterial_only::Bool=true)
    metrics = _tree_topology_metrics(tree; arterial_only=arterial_only)
    roots = sort(unique(v for v in metrics.root_vertex if v > 0))
    open(path, "w") do io
        println(io, "branch,root_vertex,root_segments,segment_count,terminal_count,max_generation,max_branchpoint_generation,max_path_length_mm,max_path_resistance_abs,root_max_diameter_mm,roles")
        for root in roots
            segs = [s for s in eachindex(tree.segment_start)
                if metrics.root_vertex[tree.segment_end[s]] == root &&
                   (!arterial_only || _is_arterial_segment(tree, s))]
            isempty(segs) && continue
            terminals = [tree.segment_end[s] for s in segs if isempty(metrics.children_segments[tree.segment_end[s]])]
            roles = sort(unique(_segment_role(tree, s) for s in segs))
            root_children = metrics.children_segments[root]
            println(io, join((
                branch,
                root,
                _csv_text(join(root_children, ";")),
                length(segs),
                length(terminals),
                maximum(metrics.generation[s] for s in segs),
                maximum(metrics.branchpoint_generation[s] for s in segs),
                maximum(10.0 * metrics.path_length_cm[tree.segment_end[s]] for s in segs),
                maximum(metrics.path_resistance[tree.segment_end[s]] for s in segs),
                maximum(10.0 * tree.segment_diameter_cm[s] for s in segs),
                _csv_text(join(roles, ";")),
            ), ","))
        end
    end
    return path
end

function _quantile_triplet(values::Vector{Float64})
    isempty(values) && return (NaN, NaN, NaN)
    q = Statistics.quantile(copy(values), [0.25, 0.50, 0.75])
    return (q[1], q[2], q[3])
end

mutable struct DiameterOrderAccumulator
    segment_count::Int
    grown_count::Int
    fixed_artery_count::Int
    diameters_um::Vector{Float64}
    lengths_mm::Vector{Float64}
    max_length_to_diameter::Float64
    max_generation::Int
    max_branchpoint_generation::Int
    max_path_length_mm::Float64
    max_path_resistance_rel::Float64
end

DiameterOrderAccumulator() = DiameterOrderAccumulator(
    0,
    0,
    0,
    Float64[],
    Float64[],
    -Inf,
    0,
    0,
    -Inf,
    -Inf,
)

function write_diameter_order_audit_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                        arterial_only::Bool=true,
                                        include_fixed_arteries::Bool=true,
                                        include_grown::Bool=true,
                                        viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    metrics = _tree_topology_metrics(tree;
        arterial_only=arterial_only,
        viscosity_poise=viscosity_poise)
    segment_filter(s::Int) = (!arterial_only || _is_arterial_segment(tree, s;
            include_fixed_arteries=include_fixed_arteries,
            include_grown=include_grown))
    accumulators = Dict{Int, DiameterOrderAccumulator}()
    for s in eachindex(tree.segment_start)
        segment_filter(s) || continue
        order = metrics.strahler_order[s]
        acc = get!(accumulators, order, DiameterOrderAccumulator())
        len_cm = _segment_length_cm(tree, s)
        d_cm = tree.segment_diameter_cm[s]
        len_mm = 10.0 * len_cm
        diameter_um = 1.0e4 * d_cm
        acc.segment_count += 1
        tree.is_xcat[s] ? (acc.fixed_artery_count += 1) : (acc.grown_count += 1)
        push!(acc.diameters_um, diameter_um)
        push!(acc.lengths_mm, len_mm)
        acc.max_length_to_diameter = max(acc.max_length_to_diameter,
            d_cm > 0.0 ? len_cm / d_cm : Inf)
        acc.max_generation = max(acc.max_generation, metrics.generation[s])
        acc.max_branchpoint_generation = max(acc.max_branchpoint_generation,
            metrics.branchpoint_generation[s])
        end_v = tree.segment_end[s]
        acc.max_path_length_mm = max(acc.max_path_length_mm,
            10.0 * metrics.path_length_cm[end_v])
        acc.max_path_resistance_rel = max(acc.max_path_resistance_rel,
            metrics.path_resistance[end_v])
    end
    orders = sort(collect(keys(accumulators)))

    open(path, "w") do io
        println(io, "branch,strahler_order,segment_count,grown_count,fixed_artery_count,min_diameter_um,p25_diameter_um,median_diameter_um,p75_diameter_um,max_diameter_um,mean_length_mm,median_length_mm,max_length_to_diameter,max_generation,max_branchpoint_generation,max_path_length_mm,max_path_resistance_abs")
        for order in orders
            acc = accumulators[order]
            p25, median_d, p75 = _quantile_triplet(acc.diameters_um)
            _, median_len, _ = _quantile_triplet(acc.lengths_mm)
            println(io, join((
                branch,
                order,
                acc.segment_count,
                acc.grown_count,
                acc.fixed_artery_count,
                minimum(acc.diameters_um),
                p25,
                median_d,
                p75,
                maximum(acc.diameters_um),
                sum(acc.lengths_mm) / length(acc.lengths_mm),
                median_len,
                acc.max_length_to_diameter,
                acc.max_generation,
                acc.max_branchpoint_generation,
                acc.max_path_length_mm,
                acc.max_path_resistance_rel,
            ), ","))
        end
    end
    return path
end

function _equiv_diameter_for_resistance(total_length_cm::Float64, resistance_sum::Float64;
                                        viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    (total_length_cm <= 0.0 || !isfinite(resistance_sum) || resistance_sum <= 0.0) && return 0.0
    radius_cm = (8.0 * viscosity_poise * total_length_cm / (pi * resistance_sum))^(0.25)
    return 2.0 * radius_cm
end

function write_hemodynamic_tree_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                    min_explicit_diameter_um::Float64=50.0,
                                    include_fixed_arteries::Bool=true,
                                    include_grown::Bool=true,
                                    collapse_degree2::Bool=true,
                                    add_virtual_inlets::Bool=true,
                                    viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE,
                                    blood_density_g_per_cm3::Float64=DEFAULT_BLOOD_DENSITY_G_PER_CM3,
                                    gravity_cm_per_s2::Float64=STANDARD_GRAVITY_CM_PER_S2)
    min_explicit_cm = min_explicit_diameter_um / 1.0e4
    metrics = _tree_topology_metrics(tree;
        arterial_only=true,
        viscosity_poise=viscosity_poise)
    child_segments = metrics.children_segments
    exportable(s::Int) = _is_arterial_segment(tree, s;
            include_fixed_arteries=include_fixed_arteries,
            include_grown=include_grown) &&
        (tree.is_xcat[s] || tree.segment_diameter_cm[s] >= min_explicit_cm)

    flow_id = 0
    rows = Vector{NamedTuple}()

    function add_virtual_inlet(root::Int, child_ids::Vector{Int})
        flow_id += 1
        this_id = flow_id
        max_d = maximum(tree.segment_diameter_cm[s] for s in child_ids)
        total_terms = sum(tree.subtree_terminal_count[tree.segment_end[s]] for s in child_ids)
        push!(rows, (
            flow_segment_id=this_id,
            parent_flow_segment_id=0,
            source_segment_ids="virtual_inlet_root$(root)",
            start_vertex=root,
            end_vertex=root,
            x1=tree.vertices[root][1],
            y1=tree.vertices[root][2],
            z1=tree.vertices[root][3],
            x2=tree.vertices[root][1],
            y2=tree.vertices[root][2],
            z2=tree.vertices[root][3],
            path_length_mm=0.0,
            chord_length_mm=0.0,
            tortuosity=1.0,
            equiv_diameter_um=1.0e4 * max_d,
            min_source_diameter_um=1.0e4 * max_d,
            max_source_diameter_um=1.0e4 * max_d,
            resistance_abs=0.0,
            delta_z_cm=0.0,
            gravity_head_mmHg=0.0,
            generation=0,
            strahler_order=isempty(child_ids) ? 1 : maximum(metrics.strahler_order[s] for s in child_ids),
            subtree_terminals=total_terms,
            exported_child_count=length(child_ids),
            pruned_child_count=0,
            terminal_bed=false,
            xcat_segment_count=0,
            grown_segment_count=0,
            labels="virtual_inlet",
        ))
        return this_id
    end

    function emit_chain(first_seg::Int, parent_flow_id::Int, flow_generation::Int)
        exportable(first_seg) || return
        source_ids = Int[]
        labels = String[]
        start_v = tree.segment_start[first_seg]
        end_v = tree.segment_end[first_seg]
        total_len = 0.0
        resistance = 0.0
        min_d = Inf
        max_d = 0.0
        xcat_count = 0
        grown_count = 0
        orders = Int[]
        current = first_seg
        pruned_children = Int[]   # sub-threshold beds lumped at EVERY vertex along the conduit

        while true
            push!(source_ids, current)
            push!(labels, tree.segment_label[current])
            len_cm = _segment_length_cm(tree, current)
            d_cm = tree.segment_diameter_cm[current]
            total_len += len_cm
            resistance += _poiseuille_resistance(len_cm, d_cm; viscosity_poise=viscosity_poise)
            min_d = min(min_d, d_cm)
            max_d = max(max_d, d_cm)
            tree.is_xcat[current] ? (xcat_count += 1) : (grown_count += 1)
            push!(orders, metrics.strahler_order[current])
            end_v = tree.segment_end[current]

            # Lump sub-threshold (pruned) arterial children at THIS vertex into the
            # conduit — including interior vertices we collapse through — so their
            # terminals are not silently dropped from the flow accounting.
            for s in child_segments[end_v]
                (_is_arterial_segment(tree, s) && !exportable(s)) && push!(pruned_children, s)
            end

            next_children = filter(exportable, child_segments[end_v])
            if !(collapse_degree2 && length(next_children) == 1)
                break
            end
            current = only(next_children)
        end

        flow_id += 1
        this_id = flow_id
        exported_children = filter(exportable, child_segments[end_v])
        chord = norm(tree.vertices[end_v] - tree.vertices[start_v])
        equiv_d = _equiv_diameter_for_resistance(total_len, resistance; viscosity_poise=viscosity_poise)
        delta_z_cm = tree.vertices[start_v][3] - tree.vertices[end_v][3]
        gravity_head_mmHg = blood_density_g_per_cm3 * gravity_cm_per_s2 * delta_z_cm / MMHG_TO_DYN_PER_CM2
        push!(rows, (
            flow_segment_id=this_id,
            parent_flow_segment_id=parent_flow_id,
            source_segment_ids=join(source_ids, ";"),
            start_vertex=start_v,
            end_vertex=end_v,
            x1=tree.vertices[start_v][1],
            y1=tree.vertices[start_v][2],
            z1=tree.vertices[start_v][3],
            x2=tree.vertices[end_v][1],
            y2=tree.vertices[end_v][2],
            z2=tree.vertices[end_v][3],
            path_length_mm=10.0 * total_len,
            chord_length_mm=10.0 * chord,
            tortuosity=chord > 1e-12 ? total_len / chord : 1.0,
            equiv_diameter_um=1.0e4 * equiv_d,
            min_source_diameter_um=1.0e4 * min_d,
            max_source_diameter_um=1.0e4 * max_d,
            resistance_abs=resistance,
            delta_z_cm=delta_z_cm,
            gravity_head_mmHg=gravity_head_mmHg,
            generation=flow_generation,
            strahler_order=isempty(orders) ? 1 : maximum(orders),
            subtree_terminals=tree.subtree_terminal_count[tree.segment_end[first_seg]],
            exported_child_count=length(exported_children),
            pruned_child_count=length(pruned_children),
            terminal_bed=isempty(exported_children),
            xcat_segment_count=xcat_count,
            grown_segment_count=grown_count,
            labels=join(sort(unique(labels)), "|"),
        ))
        for child in exported_children
            emit_chain(child, this_id, flow_generation + 1)
        end
    end

    root_vertices = [v for v in eachindex(tree.vertices) if tree.parent_vertex[v] == 0]
    for root in root_vertices
        root_children = filter(exportable, child_segments[root])
        parent = add_virtual_inlets && !isempty(root_children) ?
            add_virtual_inlet(root, root_children) : 0
        for s in root_children
            emit_chain(s, parent, 1)
        end
    end

    open(path, "w") do io
        println(io, "branch,flow_segment_id,parent_flow_segment_id,source_segment_ids,start_vertex,end_vertex,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,path_length_mm,chord_length_mm,tortuosity,equiv_diameter_um,min_source_diameter_um,max_source_diameter_um,resistance_abs,delta_z_cm,gravity_head_mmHg,generation,strahler_order,subtree_terminals,exported_child_count,pruned_child_count,terminal_bed,xcat_segment_count,grown_segment_count,labels")
        for r in rows
            println(io, join((
                branch,
                r.flow_segment_id,
                r.parent_flow_segment_id,
                _csv_text(r.source_segment_ids),
                r.start_vertex,
                r.end_vertex,
                r.x1, r.y1, r.z1,
                r.x2, r.y2, r.z2,
                r.path_length_mm,
                r.chord_length_mm,
                r.tortuosity,
                r.equiv_diameter_um,
                r.min_source_diameter_um,
                r.max_source_diameter_um,
                r.resistance_abs,
                r.delta_z_cm,
                r.gravity_head_mmHg,
                r.generation,
                r.strahler_order,
                r.subtree_terminals,
                r.exported_child_count,
                r.pruned_child_count,
                r.terminal_bed,
                r.xcat_segment_count,
                r.grown_segment_count,
                _csv_text(r.labels),
            ), ","))
        end
    end
    return path
end

function write_terminal_bed_audit_csv(path::AbstractString, branch::String, tree::GrowthTree;
                                      bed_terminal_diameter_um::Float64=8.0,
                                      min_explicit_diameter_um::Float64=50.0,
                                      terminal_bed_length_cm::Float64=0.05,
                                      gamma::Float64=3.0,
                                      proximal_gamma::Float64=gamma,
                                      transition_diameter_cm::Float64=Inf,
                                      include_fixed_arteries::Bool=true,
                                      include_grown::Bool=true,
                                      viscosity_poise::Float64=DEFAULT_BLOOD_VISCOSITY_POISE)
    bed_terminal_diameter_um > 0.0 || error("bed_terminal_diameter_um must be positive")
    min_explicit_cm = min_explicit_diameter_um / 1.0e4
    bed_terminal_cm = bed_terminal_diameter_um / 1.0e4
    terminal_bed_length_cm > 0.0 || error("terminal_bed_length_cm must be positive")
    metrics = _tree_topology_metrics(tree;
        arterial_only=true,
        viscosity_poise=viscosity_poise)
    child_segments = metrics.children_segments
    exportable(s::Int) = _is_arterial_segment(tree, s;
            include_fixed_arteries=include_fixed_arteries,
            include_grown=include_grown) &&
        (tree.is_xcat[s] || tree.segment_diameter_cm[s] >= min_explicit_cm)

    open(path, "w") do io
        println(io, "branch,bed_id,bed_root_vertex,incoming_segment,root_vertex,x_cm,y_cm,z_cm,entry_diameter_um,bed_terminal_diameter_um,estimated_parallel_terminals,tree_subtree_terminals,pruned_child_count,explicit_min_diameter_um,terminal_bed_length_mm,single_terminal_resistance_rel,lumped_parallel_resistance_rel,path_length_mm,path_resistance_rel,generation,branchpoint_generation,strahler_order,role,label")
        bed_id = 0
        for s in eachindex(tree.segment_start)
            exportable(s) || continue
            end_v = tree.segment_end[s]
            exported_children = filter(exportable, child_segments[end_v])
            isempty(exported_children) || continue
            pruned_children = [child for child in child_segments[end_v]
                if _is_arterial_segment(tree, child) && !exportable(child)]
            entry_diameter_um = 1.0e4 * tree.segment_diameter_cm[s]
            estimated_terms = max(1, ceil(Int, murray_terminal_capacity(
                max(tree.segment_diameter_cm[s], bed_terminal_cm),
                bed_terminal_cm;
                gamma=gamma,
                proximal_gamma=proximal_gamma,
                transition_diameter_cm=transition_diameter_cm)))
            subtree_terms = max(tree.subtree_terminal_count[end_v], estimated_terms)
            single_r = _poiseuille_resistance(terminal_bed_length_cm, bed_terminal_cm;
                viscosity_poise=viscosity_poise)
            lumped_r = single_r / max(subtree_terms, 1)
            p = tree.vertices[end_v]
            bed_id += 1
            println(io, join((
                branch,
                bed_id,
                end_v,
                s,
                metrics.root_vertex[end_v],
                p[1], p[2], p[3],
                entry_diameter_um,
                bed_terminal_diameter_um,
                estimated_terms,
                tree.subtree_terminal_count[end_v],
                length(pruned_children),
                min_explicit_diameter_um,
                10.0 * terminal_bed_length_cm,
                single_r,
                lumped_r,
                10.0 * metrics.path_length_cm[end_v],
                metrics.path_resistance[end_v],
                metrics.generation[s],
                metrics.branchpoint_generation[s],
                metrics.strahler_order[s],
                _segment_role(tree, s),
                _csv_text(tree.segment_label[s]),
            ), ","))
        end
    end
    return path
end
