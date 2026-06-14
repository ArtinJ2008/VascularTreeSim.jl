"""
    cube_side_tube.jl

Synthetic no-XCAT vessel-growth test:

* 6 x 6 x 6 cm solid cube tissue domain
* one fixed side tube running along the y side of the cube
* side-tube radius = 4 mm = 0.4 cm
* competitive growth predicts branches from the tube into the cube

Usage:
    julia --project=. examples/cube_side_tube.jl
"""

using VascularTreeSim
using StaticArrays
using Random
using Printf
using Dates
using Statistics

import VascularTreeSim: shell_midwall_cost

const CUBE_SIDE_CM = 6.0
const VOXEL_SPACING_CM = 0.20
const TUBE_RADIUS_CM = 0.40
const TUBE_DIAMETER_CM = 2TUBE_RADIUS_CM
const TUBE_CENTER_X_CM = TUBE_RADIUS_CM
const TUBE_CENTER_Z_CM = CUBE_SIDE_CM / 2
const TUBE_Y_MIN_CM = 0.10
const TUBE_Y_MAX_CM = CUBE_SIDE_CM - 0.10
const TUBE_VERTICES = 17
const TERMINAL_DIAMETER_CM = 0.08

const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "cube_side_tube")

struct CubeDomain
    mask::BitArray{3}
    origin_cm::SVector{3, Float64}
    spacing_cm::SVector{3, Float64}
    center_cm::SVector{3, Float64}
end

# A solid cube has no cavity midwall preference. Use uniform routing cost.
shell_midwall_cost(::CubeDomain, point) = 1.0

function voxel_center(domain::CubeDomain, i::Int, j::Int, k::Int)
    return domain.origin_cm + SVector(
        (i - 0.5) * domain.spacing_cm[1],
        (j - 0.5) * domain.spacing_cm[2],
        (k - 0.5) * domain.spacing_cm[3],
    )
end

function build_cube_domain()
    n = round(Int, CUBE_SIDE_CM / VOXEL_SPACING_CM)
    mask = trues(n, n, n)
    origin = SVector(0.0, 0.0, 0.0)
    spacing = SVector(VOXEL_SPACING_CM, VOXEL_SPACING_CM, VOXEL_SPACING_CM)
    center = SVector(CUBE_SIDE_CM / 2, CUBE_SIDE_CM / 2, CUBE_SIDE_CM / 2)
    return CubeDomain(mask, origin, spacing, center)
end

function cube_points(domain::CubeDomain; block_size::Int=1)
    dims = size(domain.mask)
    pts = Float64[]
    for k0 in 1:block_size:dims[3], j0 in 1:block_size:dims[2], i0 in 1:block_size:dims[1]
        i = min(i0 + block_size ÷ 2, dims[1])
        j = min(j0 + block_size ÷ 2, dims[2])
        k = min(k0 + block_size ÷ 2, dims[3])
        p = voxel_center(domain, i, j, k)
        push!(pts, p[1], p[2], p[3])
    end
    n = length(pts) ÷ 3
    return copy(reshape(pts, 3, n)')
end

function make_side_tube_tree()
    ys = collect(range(TUBE_Y_MIN_CM, TUBE_Y_MAX_CM; length=TUBE_VERTICES))
    vertices = [SVector(TUBE_CENTER_X_CM, y, TUBE_CENTER_Z_CM) for y in ys]
    parent_vertex = [i == 1 ? 0 : i - 1 for i in 1:TUBE_VERTICES]
    incoming_segment = [i == 1 ? 0 : i - 1 for i in 1:TUBE_VERTICES]
    children = [Int[] for _ in 1:TUBE_VERTICES]
    for i in 1:(TUBE_VERTICES - 1)
        push!(children[i], i + 1)
    end

    nseg = TUBE_VERTICES - 1
    segment_start = collect(1:nseg)
    segment_end = collect(2:TUBE_VERTICES)
    segment_diameter_cm = fill(TUBE_DIAMETER_CM, nseg)
    segment_label = fill("side_tube_r4mm", nseg)
    is_xcat = fill(true, nseg)
    subtree_terminal_count = zeros(Int, TUBE_VERTICES)

    return GrowthTree(
        "CubeTube",
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
        TERMINAL_DIAMETER_CM,
        1,
        TUBE_DIAMETER_CM,
    )
end

function preserve_side_tube_diameter!(tree::GrowthTree)
    for s in eachindex(tree.segment_label)
        if startswith(tree.segment_label[s], "side_tube")
            tree.segment_diameter_cm[s] = TUBE_DIAMETER_CM
        end
    end
    return tree
end

function write_points_csv(path::AbstractString, points::Matrix{Float64})
    open(path, "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for i in axes(points, 1)
            @printf(io, "%.4f,%.4f,%.4f\n", points[i, 1], points[i, 2], points[i, 3])
        end
    end
    return path
end

function write_tube_axis_csv(path::AbstractString, tree::GrowthTree)
    tube_vertex_ids = Set{Int}()
    for s in eachindex(tree.segment_label)
        startswith(tree.segment_label[s], "side_tube") || continue
        push!(tube_vertex_ids, tree.segment_start[s])
        push!(tube_vertex_ids, tree.segment_end[s])
    end
    ordered_vertices = sort(collect(tube_vertex_ids), by = v -> tree.vertices[v][2])

    open(path, "w") do io
        println(io, "x_cm,y_cm,z_cm,radius_cm")
        for v_id in ordered_vertices
            v = tree.vertices[v_id]
            @printf(io, "%.4f,%.4f,%.4f,%.4f\n", v[1], v[2], v[3], TUBE_RADIUS_CM)
        end
    end
    return path
end

function js_number_array(values)
    return "[" * join((@sprintf("%.4f", v) for v in values), ",") * "]"
end

function line_arrays(tree::GrowthTree; tube::Bool)
    xs = String[]
    ys = String[]
    zs = String[]
    for s in eachindex(tree.segment_start)
        is_tube = startswith(tree.segment_label[s], "side_tube")
        is_tube == tube || continue
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        push!(xs, @sprintf("%.4f", a[1]), @sprintf("%.4f", b[1]), "null")
        push!(ys, @sprintf("%.4f", a[2]), @sprintf("%.4f", b[2]), "null")
        push!(zs, @sprintf("%.4f", a[3]), @sprintf("%.4f", b[3]), "null")
    end
    return "[" * join(xs, ",") * "]", "[" * join(ys, ",") * "]", "[" * join(zs, ",") * "]"
end

function write_viewer(path::AbstractString, domain_points::Matrix{Float64}, tree::GrowthTree)
    max_domain = min(size(domain_points, 1), 7000)
    rng = MersenneTwister(7)
    keep = sort(randperm(rng, size(domain_points, 1))[1:max_domain])
    display_points = domain_points[keep, :]

    tube_x, tube_y, tube_z = line_arrays(tree; tube=true)
    pred_x, pred_y, pred_z = line_arrays(tree; tube=false)
    dx = js_number_array(display_points[:, 1])
    dy = js_number_array(display_points[:, 2])
    dz = js_number_array(display_points[:, 3])

    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Cube Side-Tube Vessel Prediction</title>
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
  <b>6 x 6 x 6 cm cube</b><br>
  side tube radius: 4 mm<br>
  drag to rotate, scroll to zoom
</div>
<script>
const traces = [
  {
    type: "scatter3d", mode: "markers", name: "Cube domain sample",
    x: $dx, y: $dy, z: $dz,
    marker: { size: 1.4, color: "#8d99ae", opacity: 0.16 },
    hoverinfo: "skip"
  },
  {
    type: "scatter3d", mode: "lines", name: "Fixed side tube (r=4 mm)",
    x: $tube_x, y: $tube_y, z: $tube_z,
    line: { color: "#ffb703", width: 8 },
    hoverinfo: "name"
  },
  {
    type: "scatter3d", mode: "lines", name: "Predicted branches",
    x: $pred_x, y: $pred_y, z: $pred_z,
    line: { color: "#2a9d8f", width: 3 },
    hoverinfo: "name"
  }
];
Plotly.newPlot("plot", traces, {
  paper_bgcolor:"#101217",
  plot_bgcolor:"#101217",
  margin:{l:0,r:0,t:0,b:0},
  scene:{
    bgcolor:"#101217",
    xaxis:{title:"x (cm)", color:"#c8c8c8", gridcolor:"#2a2e39", range:[0,6]},
    yaxis:{title:"y (cm)", color:"#c8c8c8", gridcolor:"#2a2e39", range:[0,6]},
    zaxis:{title:"z (cm)", color:"#c8c8c8", gridcolor:"#2a2e39", range:[0,6]},
    aspectmode:"cube"
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

function write_summary(path::AbstractString, tree::GrowthTree, stats)
    side_tube_segments = count(label -> startswith(label, "side_tube"), tree.segment_label)
    total_segments = length(tree.segment_start)
    predicted_segments = total_segments - side_tube_segments
    terminals = count(v -> isempty(v), tree.children)
    min_d = minimum(tree.segment_diameter_cm) * 10.0
    max_d = maximum(tree.segment_diameter_cm) * 10.0

    open(path, "w") do io
        println(io, "# Cube Side-Tube Vessel Prediction")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- Domain: $(CUBE_SIDE_CM) x $(CUBE_SIDE_CM) x $(CUBE_SIDE_CM) cm cube")
        println(io, "- Voxel spacing: $(VOXEL_SPACING_CM) cm")
        println(io, "- Domain voxels: $(count(build_cube_domain().mask))")
        println(io, "- Side tube centerline: x=$(TUBE_CENTER_X_CM) cm, z=$(TUBE_CENTER_Z_CM) cm, y=$(TUBE_Y_MIN_CM)-$(TUBE_Y_MAX_CM) cm")
        println(io, "- Side tube radius: $(TUBE_RADIUS_CM) cm = 4 mm")
        println(io, "- Side tube diameter: $(TUBE_DIAMETER_CM) cm = 8 mm")
        println(io, "- Terminal diameter used for this gross synthetic test: $(TERMINAL_DIAMETER_CM) cm")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- Total segments: $(total_segments)")
        println(io, "- Fixed side-tube segments: $(side_tube_segments)")
        println(io, "- Predicted/grown segments: $(predicted_segments)")
        println(io, "- Terminal vertices: $(terminals)")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
        println(io, "- Diameter range: $(round(min_d; digits=2))-$(round(max_d; digits=2)) mm")
    end
    return path
end

function main()
    Random.seed!(42)
    mkpath(OUTPUT_DIR)

    domain = build_cube_domain()
    coverage_points = cube_points(domain; block_size=2)
    graph_points = cube_points(domain; block_size=2)
    full_domain_points = cube_points(domain; block_size=1)

    tree = make_side_tube_tree()
    trees = Dict("CubeTube" => tree)

    println("=" ^ 60)
    println("Cube side-tube vessel prediction")
    println("Domain: $(CUBE_SIDE_CM)x$(CUBE_SIDE_CM)x$(CUBE_SIDE_CM) cm, spacing=$(VOXEL_SPACING_CM) cm")
    println("Tube: radius=$(TUBE_RADIUS_CM) cm (4 mm), diameter=$(TUBE_DIAMETER_CM) cm")
    println("Coverage points: $(size(coverage_points, 1))")
    println("Output: $(OUTPUT_DIR)")
    println("=" ^ 60)
    flush(stdout)

    _, _, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.18,
        capillary_diameter_cm=0.02,
        max_new_branches_per_tree=160,
        graph_neighbors=14,
        min_frontier_separation_cm=0.32,
        max_path_nodes=18,
        frontier_batch=10,
        gamma=3.0,
        smooth_passes=8,
        spline_density=4,
        max_segment_length_cm=0.18,
        turn_penalty=0.35,
        graph_jitter_cm=0.03,
        use_gpu=false)

    preserve_side_tube_diameter!(tree)
    st = stats["CubeTube"]

    csv_path = joinpath(OUTPUT_DIR, "cube_side_tube_segments.csv")
    domain_csv = joinpath(OUTPUT_DIR, "cube_domain_points.csv")
    tube_csv = joinpath(OUTPUT_DIR, "side_tube_axis_points.csv")
    viewer_path = joinpath(OUTPUT_DIR, "cube_side_tube_viewer.html")
    summary_path = joinpath(OUTPUT_DIR, "summary.md")

    write_growth_csv(csv_path, "CubeTube", tree)
    write_points_csv(domain_csv, full_domain_points)
    write_tube_axis_csv(tube_csv, tree)
    write_viewer(viewer_path, full_domain_points, tree)
    write_summary(summary_path, tree, st)

    side_tube_segments = count(label -> startswith(label, "side_tube"), tree.segment_label)
    predicted_segments = length(tree.segment_start) - side_tube_segments

    println()
    println("Result")
    println("  added branches       = $(st.added)")
    println("  total segments       = $(length(tree.segment_start))")
    println("  side-tube segments   = $(side_tube_segments)")
    println("  predicted segments   = $(predicted_segments)")
    println("  p95 coverage         = $(round(st.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(csv_path)")
    println("  $(domain_csv)")
    println("  $(tube_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

main()
