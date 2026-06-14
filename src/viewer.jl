"""
    HTML viewer generation.

Two modes:
  1. domain_check_html — domain-only viewer for user confirmation (before growth)
  2. growth_viewer_html — full viewer with trees (delegates to build_viewer.py)
"""

function _sample_points_for_viewer(pts::Matrix{Float64}; max_points::Int=500_000)
    n = size(pts, 1)
    if n <= max_points
        return pts
    end
    # Random subsample for display
    rng = Random.MersenneTwister(42)
    indices = sort(Random.randperm(rng, n)[1:max_points])
    return pts[indices, :]
end

"""
    domain_check_html(path, domain; max_display=500_000)

Generate a domain-only HTML viewer for user confirmation before growth.
Shows myocardium, epicardial surface, and endocardial surfaces.
"""
function domain_check_html(path::AbstractString, domain::VoxelShellDomain;
        max_display::Int=500_000)

    # Collect domain points
    all_pts = voxel_mask_points(domain)
    display_pts = _sample_points_for_viewer(all_pts; max_points=max_display)

    # Build JSON traces manually (simple enough for domain-only)
    traces = String[]

    # Domain (myocardium) trace
    n = size(display_pts, 1)
    dx = join([@sprintf("%.3f", display_pts[i,1]) for i in 1:n], ",")
    dy = join([@sprintf("%.3f", display_pts[i,2]) for i in 1:n], ",")
    dz = join([@sprintf("%.3f", display_pts[i,3]) for i in 1:n], ",")
    push!(traces, """{
        "type":"scatter3d","mode":"markers","name":"Myocardium ($(size(all_pts,1)) pts, showing $(n))",
        "x":[$dx],"y":[$dy],"z":[$dz],
        "marker":{"size":1.5,"color":"#6b7280","opacity":0.15},"hoverinfo":"skip"
    }""")

    # Epicardial surface
    epi = domain.outer_surface_points
    ne = size(epi, 1)
    ex = join([@sprintf("%.3f", epi[i,1]) for i in 1:ne], ",")
    ey = join([@sprintf("%.3f", epi[i,2]) for i in 1:ne], ",")
    ez = join([@sprintf("%.3f", epi[i,3]) for i in 1:ne], ",")
    push!(traces, """{
        "type":"scatter3d","mode":"markers","name":"Epicardium ($(ne) pts)",
        "x":[$ex],"y":[$ey],"z":[$ez],
        "marker":{"size":1.5,"color":"#ef4444","opacity":0.25},"hoverinfo":"skip","visible":false
    }""")

    # Endocardial surfaces
    colors = ["#3b82f6","#8b5cf6","#06b6d4","#10b981","#f59e0b","#ec4899","#6366f1","#14b8a6"]
    for (idx, cavity_pts) in enumerate(domain.cavity_surface_points)
        nc = size(cavity_pts, 1)
        color = colors[mod1(idx, length(colors))]
        cx = join([@sprintf("%.3f", cavity_pts[i,1]) for i in 1:nc], ",")
        cy = join([@sprintf("%.3f", cavity_pts[i,2]) for i in 1:nc], ",")
        cz = join([@sprintf("%.3f", cavity_pts[i,3]) for i in 1:nc], ",")
        push!(traces, """{
            "type":"scatter3d","mode":"markers","name":"Cavity $(idx) ($(nc) pts)",
            "x":[$cx],"y":[$cy],"z":[$cz],
            "marker":{"size":1.5,"color":"$(color)","opacity":0.20},"hoverinfo":"skip","visible":false
        }""")
    end

    traces_json = join(traces, ",\n")
    n_traces = length(traces)

    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Domain Check</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script></head>
<body style="margin:0;font-family:Arial,sans-serif">
<div style="padding:10px 14px">
  <h2 style="margin:0 0 6px">Domain Confirmation</h2>
  <div style="color:#555">Check: no holes in myocardium surface, cavities properly excluded</div>
</div>
<div style="padding:0 14px 8px;display:flex;gap:8px">
  <button onclick="toggle(0)">Toggle Myocardium</button>
  <button onclick="toggle(1)">Toggle Epicardium</button>
  <button onclick="for(let i=2;i<$(n_traces);i++) toggle(i)">Toggle Cavities</button>
  <button onclick="Plotly.restyle('plot',{visible:true},Array.from({length:$(n_traces)},(_,i)=>i))">Show All</button>
</div>
<div id="plot" style="width:100vw;height:85vh"></div>
<script>
const traces = [$traces_json];
Plotly.newPlot('plot', traces, {
  scene:{xaxis:{title:'X (cm)'},yaxis:{title:'Y (cm)'},zaxis:{title:'Z (cm)'},aspectmode:'data'},
  margin:{l:0,r:0,b:0,t:0}
},{displaylogo:false,responsive:true});
function toggle(i){
  const p=document.getElementById('plot');
  Plotly.restyle('plot',{visible:p.data[i].visible!==false?false:true},[i]);
}
</script></body></html>"""

    open(path, "w") do io
        write(io, html)
    end
    @info "Domain check viewer: $path ($(round(filesize(path)/1e6; digits=1)) MB)"
    return path
end

function _viewer_number_array(values)
    return "[" * join((@sprintf("%.4f", v) for v in values), ",") * "]"
end

function _viewer_tree_line_arrays(tree::GrowthTree)
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

function _write_fallback_growth_viewer(path::AbstractString, domain, trees::Dict{String, GrowthTree},
        stats::Dict{String, NamedTuple}, color_map::Dict{String, String};
        domain_stride::Int=1, max_domain_points::Int=40_000)

    domain_points = _sample_points_for_viewer(coverage_target_points_blockwise(domain; block_size=max(domain_stride, 1));
        max_points=max_domain_points)
    dx = _viewer_number_array(domain_points[:, 1])
    dy = _viewer_number_array(domain_points[:, 2])
    dz = _viewer_number_array(domain_points[:, 3])

    traces = String[]
    push!(traces, """{
      "type":"scatter3d","mode":"markers","name":"Domain sample",
      "x":$dx,"y":$dy,"z":$dz,
      "marker":{"size":1.5,"color":"#8d99ae","opacity":0.15},
      "hoverinfo":"skip"
    }""")

    fallback_colors = ["#1f77ff", "#e3342f", "#22aa44", "#ff7f0e", "#9467bd"]
    for (idx, name) in enumerate(sort(collect(keys(trees))))
        tree = trees[name]
        tx, ty, tz = _viewer_tree_line_arrays(tree)
        color = get(color_map, name, fallback_colors[mod1(idx, length(fallback_colors))])
        st = get(stats, name, nothing)
        label = st === nothing ? name : "$(name): $(length(tree.segment_start)) segments, p95=$(round(st.p95 * 10; digits=2)) mm"
        safe_label = replace(label, "\\" => "\\\\", "\"" => "\\\"")
        push!(traces, """{
          "type":"scatter3d","mode":"lines","name":"$safe_label",
          "x":$tx,"y":$ty,"z":$tz,
          "line":{"color":"$color","width":3},
          "hoverinfo":"name"
        }""")
    end

    traces_json = join(traces, ",\n")
    html = """<!doctype html>
<html><head><meta charset="utf-8"><title>Vascular Growth Viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body { margin:0; height:100%; background:#101217; color:#e7e7e7; font-family:Arial,sans-serif; }
  #plot { width:100vw; height:100vh; }
  #info { position:fixed; left:12px; top:12px; padding:10px 12px; background:rgba(0,0,0,.62); border-radius:4px; font-size:13px; line-height:1.45; }
</style></head>
<body><div id="plot"></div>
<div id="info"><b>Vascular growth viewer</b><br>drag to rotate, scroll to zoom</div>
<script>
const traces = [$traces_json];
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
</script></body></html>
"""

    open(path, "w") do io
        write(io, html)
    end
    @info "Fallback viewer HTML generated: $path"
    return path
end

function growth_viewer_html(path::AbstractString, domain, trees::Dict{String, GrowthTree},
        stats::Dict{String, NamedTuple}, color_map::Dict{String, String};
        domain_stride::Int=1, surface_stride::Int=4)

    output_dir = dirname(path)

    # Write tree CSVs (the Python script reads them)
    for (branch, tree) in trees
        csv_path = joinpath(output_dir, lowercase(branch) * "_segments.csv")
        write_growth_csv(csv_path, branch, tree)
    end

    # Run the Python builder script
    script_path = joinpath(output_dir, "build_viewer.py")
    if !isfile(script_path)
        @info "build_viewer.py not found at $script_path; using fallback Julia viewer"
        return _write_fallback_growth_viewer(path, domain, trees, stats, color_map; domain_stride=domain_stride)
    end

    cmd = `python3 $script_path`
    try
        run(cmd)
        @info "Viewer HTML generated: $path"
    catch e
        @warn "Failed to generate viewer HTML" exception=e
        return _write_fallback_growth_viewer(path, domain, trees, stats, color_map; domain_stride=domain_stride)
    end

    isfile(path) || return _write_fallback_growth_viewer(path, domain, trees, stats, color_map; domain_stride=domain_stride)
    return path
end
