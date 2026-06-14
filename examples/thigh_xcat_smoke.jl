"""
    thigh_xcat_smoke.jl

Smoke-test vessel growth through a grouped XCAT NRB muscle/leg object.

Usage:
    julia --project=. examples/thigh_xcat_smoke.jl [nrb_path] [target_object] [max_branches]

If target_object is omitted or "auto", the script chooses the largest
`musc####` object whose centroid falls inside a leg_left/leg_right bounding
box. The object inventory is always written so the target can be changed after
inspection.
"""

using VascularTreeSim
using StaticArrays
using Random
using Printf
using Dates
using Statistics

const DEFAULT_NRB = raw"Z:\Artin\XCAT\thigh_xcat_1.nrb"
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "thigh_xcat_smoke")
const COORDINATE_SCALE = 0.1       # XCAT mm -> cm
const VOXEL_SPACING_CM = 0.30      # coarse by design: this is a smoke test
const SAMPLE_GRID = (28, 28)
const TERMINAL_DIAMETER_CM = 0.06

function csv_cell(x)
    s = string(x)
    return "\"" * replace(s, "\"" => "\"\"") * "\""
end

function group_surfaces(surfaces)
    groups = Dict{String, Vector{XCATNurbsSurface}}()
    for surface in surfaces
        push!(get!(groups, xcat_group_name(surface), XCATNurbsSurface[]), surface)
    end
    return groups
end

function group_bounds(surfaces::AbstractVector{XCATNurbsSurface})
    lo = SVector(Inf, Inf, Inf)
    hi = SVector(-Inf, -Inf, -Inf)
    control_points = 0
    for surface in surfaces
        slo, shi = xcat_bounds(surface)
        lo = min.(lo, slo)
        hi = max.(hi, shi)
        control_points += length(surface.control_points)
    end
    return lo, hi, control_points
end

function bbox_volume(lo, hi)
    ext = max.(hi .- lo, 0.0)
    return ext[1] * ext[2] * ext[3]
end

function inside_bounds(p, lo, hi; pad=0.0)
    return all((lo .- pad) .<= p .<= (hi .+ pad))
end

function inventory_rows(groups)
    rows = NamedTuple[]
    leg_bounds = Tuple{SVector{3, Float64}, SVector{3, Float64}}[]
    for leg_name in ("leg_left", "leg_right")
        if haskey(groups, leg_name)
            lo, hi, _ = group_bounds(groups[leg_name])
            push!(leg_bounds, (lo, hi))
        end
    end

    for name in sort(collect(keys(groups)))
        lo, hi, control_points = group_bounds(groups[name])
        center = 0.5 .* (lo .+ hi)
        ext = hi .- lo
        in_leg = any(bounds -> inside_bounds(center, bounds[1], bounds[2]; pad=25.0), leg_bounds)
        push!(rows, (
            name=name,
            patch_count=length(groups[name]),
            control_points=control_points,
            is_muscle=startswith(lowercase(name), "musc"),
            center=center,
            extent=ext,
            min=lo,
            max=hi,
            bbox_volume_mm3=bbox_volume(lo, hi),
            center_in_leg_bbox=in_leg,
        ))
    end
    return rows
end

function write_inventory_csv(path, rows)
    open(path, "w") do io
        println(io, join([
            "object", "patch_count", "control_points", "is_muscle", "center_in_leg_bbox",
            "center_x_mm", "center_y_mm", "center_z_mm",
            "extent_x_mm", "extent_y_mm", "extent_z_mm",
            "min_x_mm", "min_y_mm", "min_z_mm",
            "max_x_mm", "max_y_mm", "max_z_mm",
            "bbox_volume_mm3",
        ], ","))
        for row in rows
            println(io, join((
                csv_cell(row.name), row.patch_count, row.control_points, row.is_muscle,
                row.center_in_leg_bbox,
                row.center[1], row.center[2], row.center[3],
                row.extent[1], row.extent[2], row.extent[3],
                row.min[1], row.min[2], row.min[3],
                row.max[1], row.max[2], row.max[3],
                row.bbox_volume_mm3,
            ), ","))
        end
    end
    return path
end

function write_candidate_csv(path, rows; limit=80)
    candidates = [row for row in rows if row.is_muscle]
    sort!(candidates, by = row -> (row.center_in_leg_bbox ? 1 : 0, row.bbox_volume_mm3), rev=true)
    open(path, "w") do io
        println(io, "rank,object,patch_count,center_in_leg_bbox,center_x_cm,center_y_cm,center_z_cm,extent_x_cm,extent_y_cm,extent_z_cm,bbox_volume_cm3")
        for (rank, row) in enumerate(candidates[1:min(limit, length(candidates))])
            c = COORDINATE_SCALE .* row.center
            e = COORDINATE_SCALE .* row.extent
            @printf(io, "%d,%s,%d,%s,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                rank,
                csv_cell(row.name),
                row.patch_count,
                string(row.center_in_leg_bbox),
                c[1], c[2], c[3],
                e[1], e[2], e[3],
                row.bbox_volume_mm3 * COORDINATE_SCALE^3)
        end
    end
    return candidates
end

function choose_target(groups, rows, requested::AbstractString)
    requested_clean = strip(requested)
    if !isempty(requested_clean) && lowercase(requested_clean) != "auto"
        haskey(groups, requested_clean) && return requested_clean, "requested"
        sample_names = join(first(sort(collect(keys(groups))), min(12, length(groups))), ", ")
        error("Target object `$requested_clean` not found. Inventory examples: $sample_names")
    end

    muscles = [row for row in rows if row.is_muscle]
    sort!(muscles, by = row -> (row.center_in_leg_bbox ? 1 : 0, row.bbox_volume_mm3), rev=true)
    if !isempty(muscles)
        reason = muscles[1].center_in_leg_bbox ? "largest muscle-like object inside leg bbox" : "largest muscle-like object"
        return muscles[1].name, reason
    end

    for fallback in ("leg_left", "leg_right")
        haskey(groups, fallback) && return fallback, "fallback leg surface"
    end

    largest = maximum(rows, by = row -> row.bbox_volume_mm3)
    return largest.name, "largest object fallback"
end

function write_points_csv(path, points::Matrix{Float64})
    open(path, "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for i in axes(points, 1)
            @printf(io, "%.5f,%.5f,%.5f\n", points[i, 1], points[i, 2], points[i, 3])
        end
    end
    return path
end

function sample_rows(points::Matrix{Float64}, max_points::Int; seed=11)
    size(points, 1) <= max_points && return points
    rng = MersenneTwister(seed)
    keep = sort(randperm(rng, size(points, 1))[1:max_points])
    return points[keep, :]
end

function nearest_domain_seed(domain)
    points = coverage_target_points_blockwise(domain; block_size=1)
    size(points, 1) > 0 || error("Cannot choose a seed from an empty domain")
    center = domain.center_cm
    best_idx = 1
    best_d2 = Inf
    for i in axes(points, 1)
        dx = points[i, 1] - center[1]
        dy = points[i, 2] - center[2]
        dz = points[i, 3] - center[3]
        d2 = dx * dx + dy * dy + dz * dz
        if d2 < best_d2
            best_d2 = d2
            best_idx = i
        end
    end
    return SVector(points[best_idx, 1], points[best_idx, 2], points[best_idx, 3])
end

function js_number_array(values)
    return "[" * join((@sprintf("%.4f", v) for v in values), ",") * "]"
end

function tree_line_arrays(tree::GrowthTree)
    xs = String[]
    ys = String[]
    zs = String[]
    for s in eachindex(tree.segment_start)
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        push!(xs, @sprintf("%.4f", a[1]), @sprintf("%.4f", b[1]), "null")
        push!(ys, @sprintf("%.4f", a[2]), @sprintf("%.4f", b[2]), "null")
        push!(zs, @sprintf("%.4f", a[3]), @sprintf("%.4f", b[3]), "null")
    end
    return "[" * join(xs, ",") * "]", "[" * join(ys, ",") * "]", "[" * join(zs, ",") * "]"
end

function write_viewer(path, target_name, domain_points, surface_points, tree::GrowthTree)
    display_domain = sample_rows(domain_points, 9000; seed=17)
    display_surface = sample_rows(surface_points, 9000; seed=19)
    vessel_x, vessel_y, vessel_z = tree_line_arrays(tree)
    seed = tree.vertices[tree.root_vertex]

    dx = js_number_array(display_domain[:, 1])
    dy = js_number_array(display_domain[:, 2])
    dz = js_number_array(display_domain[:, 3])
    sx = js_number_array(display_surface[:, 1])
    sy = js_number_array(display_surface[:, 2])
    sz = js_number_array(display_surface[:, 3])
    safe_title = replace(target_name, "\\" => "\\\\", "\"" => "\\\"")

    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>XCAT Thigh Vessel Smoke Test</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { margin:0; height:100%; background:#101217; color:#e7e7e7; font-family:Arial,sans-serif; }
    #plot { width:100vw; height:100vh; }
    #info { position:fixed; left:12px; top:12px; padding:10px 12px; background:rgba(0,0,0,.62); border-radius:4px; font-size:13px; line-height:1.45; }
  </style>
</head>
<body>
<div id="plot"></div>
<div id="info">
  <b>XCAT target:</b> $safe_title<br>
  solid multi-patch flood-fill domain<br>
  drag to rotate, scroll to zoom
</div>
<script>
const traces = [
  {
    type: "scatter3d", mode: "markers", name: "Voxel domain sample",
    x: $dx, y: $dy, z: $dz,
    marker: { size: 1.6, color: "#9aa3b2", opacity: 0.18 },
    hoverinfo: "skip"
  },
  {
    type: "scatter3d", mode: "markers", name: "XCAT surface sample",
    x: $sx, y: $sy, z: $sz,
    marker: { size: 1.4, color: "#ffb703", opacity: 0.28 },
    hoverinfo: "skip"
  },
  {
    type: "scatter3d", mode: "lines", name: "Predicted vessel tree",
    x: $vessel_x, y: $vessel_y, z: $vessel_z,
    line: { color: "#2a9d8f", width: 4 },
    hoverinfo: "name"
  },
  {
    type: "scatter3d", mode: "markers", name: "Seed",
    x: [$(seed[1])], y: [$(seed[2])], z: [$(seed[3])],
    marker: { size: 5, color: "#e63946" },
    hoverinfo: "name"
  }
];
Plotly.newPlot("plot", traces, {
  paper_bgcolor:"#101217",
  plot_bgcolor:"#101217",
  margin:{l:0,r:0,t:0,b:0},
  scene:{
    bgcolor:"#101217",
    xaxis:{title:"x (cm)", color:"#c8c8c8", gridcolor:"#2a2e39"},
    yaxis:{title:"y (cm)", color:"#c8c8c8", gridcolor:"#2a2e39"},
    zaxis:{title:"z (cm)", color:"#c8c8c8", gridcolor:"#2a2e39"},
    aspectmode:"data"
  },
  legend:{x:0.99, xanchor:"right", y:0.98, font:{color:"#ddd"}, bgcolor:"rgba(0,0,0,.55)"}
}, {displaylogo:false, responsive:true});
</script>
</body>
</html>
"""
    open(path, "w") do io
        write(io, html)
    end
    return path
end

function write_summary(path, nrb_path, target_name, target_reason, selected_surfaces, domain, coverage_points, graph_points, tree, stats)
    lo, hi, _ = group_bounds(selected_surfaces)
    ext = hi .- lo
    open(path, "w") do io
        println(io, "# XCAT Thigh Smoke Test")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- NRB: $(nrb_path)")
        println(io, "- Target object: $(target_name) ($(target_reason))")
        println(io, "- Target patches: $(length(selected_surfaces))")
        println(io, "- Target bounds: x $(round(lo[1]; digits=2))-$(round(hi[1]; digits=2)) mm, y $(round(lo[2]; digits=2))-$(round(hi[2]; digits=2)) mm, z $(round(lo[3]; digits=2))-$(round(hi[3]; digits=2)) mm")
        println(io, "- Target extent: $(round(ext[1] * COORDINATE_SCALE; digits=2)) x $(round(ext[2] * COORDINATE_SCALE; digits=2)) x $(round(ext[3] * COORDINATE_SCALE; digits=2)) cm")
        println(io, "- Voxel spacing: $(VOXEL_SPACING_CM) cm")
        println(io, "- Domain voxels: $(count(domain.mask))")
        println(io, "- Coverage points: $(size(coverage_points, 1))")
        println(io, "- Graph points: $(size(graph_points, 1))")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- Total segments: $(length(tree.segment_start))")
        println(io, "- Terminal vertices: $(count(v -> isempty(v), tree.children))")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "## Notes")
        println(io)
        println(io, "- The NRB uses coded muscle names such as `musc####`; this script does not know their human anatomical labels.")
        println(io, "- This is a bounded smoke test, not a production capillary-resolution run.")
    end
    return path
end

function main()
    Random.seed!(42)
    nrb_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NRB
    requested_target = length(ARGS) >= 2 ? ARGS[2] : "auto"
    max_branches = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 120

    mkpath(OUTPUT_DIR)
    inventory_path = joinpath(OUTPUT_DIR, "xcat_object_inventory.csv")
    candidates_path = joinpath(OUTPUT_DIR, "candidate_muscles.csv")
    csv_path = joinpath(OUTPUT_DIR, "thigh_xcat_segments.csv")
    domain_csv = joinpath(OUTPUT_DIR, "thigh_xcat_domain_points.csv")
    surface_csv = joinpath(OUTPUT_DIR, "thigh_xcat_surface_points.csv")
    viewer_path = joinpath(OUTPUT_DIR, "thigh_xcat_viewer.html")
    summary_path = joinpath(OUTPUT_DIR, "summary.md")

    println("=" ^ 64)
    println("XCAT thigh grouped-NRB smoke test")
    println("NRB: $(nrb_path)")
    println("Output: $(OUTPUT_DIR)")
    println("=" ^ 64)
    flush(stdout)

    surfaces = parse_xcat_grouped_nrb(nrb_path)
    groups = group_surfaces(surfaces)
    rows = inventory_rows(groups)
    write_inventory_csv(inventory_path, rows)
    write_candidate_csv(candidates_path, rows)

    target_name, target_reason = choose_target(groups, rows, requested_target)
    selected_surfaces = groups[target_name]
    println("[xcat] parsed surfaces=$(length(surfaces)) objects=$(length(groups))")
    println("[xcat] selected target=$(target_name) ($(target_reason)), patches=$(length(selected_surfaces))")
    flush(stdout)

    domain = build_multipatch_voxel_domain_floodfill(selected_surfaces;
        coordinate_scale=COORDINATE_SCALE,
        voxel_spacing_cm=VOXEL_SPACING_CM,
        outer_samples=SAMPLE_GRID,
        dilation_radius=1)

    count(domain.mask) > 20 || error("The selected object produced an empty/tiny voxel domain. Pick a different target from $(candidates_path).")

    coverage_points = coverage_target_points_blockwise(domain; block_size=2)
    graph_points = coverage_target_points_blockwise(domain; block_size=2)
    all_domain_points = coverage_target_points_blockwise(domain; block_size=1)
    seed = nearest_domain_seed(domain)
    tree = growth_tree_from_seed("ThighSeed", seed; terminal_diameter_cm=TERMINAL_DIAMETER_CM)
    trees = Dict("ThighSeed" => tree)

    println("[growth] seed=$(seed) coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) max_branches=$(max_branches)")
    flush(stdout)

    _, _, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.35,
        capillary_diameter_cm=0.02,
        max_new_branches_per_tree=max_branches,
        graph_neighbors=12,
        min_frontier_separation_cm=0.60,
        max_path_nodes=18,
        frontier_batch=8,
        gamma=3.0,
        smooth_passes=8,
        spline_density=3,
        max_segment_length_cm=0.30,
        turn_penalty=0.35,
        graph_jitter_cm=0.04,
        use_gpu=false)

    st = stats["ThighSeed"]
    write_growth_csv(csv_path, "ThighSeed", tree)
    write_points_csv(domain_csv, all_domain_points)
    write_points_csv(surface_csv, domain.outer_surface_points)
    write_viewer(viewer_path, target_name, all_domain_points, domain.outer_surface_points, tree)
    write_summary(summary_path, nrb_path, target_name, target_reason, selected_surfaces, domain, coverage_points, graph_points, tree, st)

    println()
    println("Result")
    println("  target              = $(target_name) ($(target_reason))")
    println("  domain voxels       = $(count(domain.mask))")
    println("  added branches      = $(st.added)")
    println("  total segments      = $(length(tree.segment_start))")
    println("  p95 coverage        = $(round(st.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(inventory_path)")
    println("  $(candidates_path)")
    println("  $(csv_path)")
    println("  $(domain_csv)")
    println("  $(surface_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

main()
