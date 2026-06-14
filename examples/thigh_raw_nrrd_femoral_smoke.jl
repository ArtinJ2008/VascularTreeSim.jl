"""
    thigh_raw_nrrd_femoral_smoke.jl

Smoke-test vessel growth in a binary thigh mask using anatomy-informed fixed
arterial seeds. The seeds are snapped into the foreground mask slice-by-slice:

- femoral artery through the femoral triangle, adductor canal, and hiatus
- profunda femoris branch in the deep posterior thigh
- medial and lateral circumflex branches in the proximal thigh
- three perforator-style branches toward the posterior compartment
- distal descending-genicular-style branch near the adductor canal

Usage:
    julia --project=. examples/thigh_raw_nrrd_femoral_smoke.jl [nhdr_path] [max_branches]
"""

include(joinpath(@__DIR__, "thigh_raw_nrrd_smoke.jl"))

using Base64

const FEMORAL_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "thigh_raw_nrrd_femoral_smoke")
const FIXED_BRANCH_ORDER = [
    "femoral_trunk",
    "profunda_femoris",
    "lateral_circumflex_descending",
    "medial_circumflex",
    "perforator_1",
    "perforator_2",
    "perforator_3",
    "descending_genicular",
]
const FIXED_BRANCH_DIAMETER_CM = Dict(
    "femoral_trunk" => 0.60,
    "profunda_femoris" => 0.42,
    "lateral_circumflex_descending" => 0.28,
    "medial_circumflex" => 0.30,
    "perforator_1" => 0.22,
    "perforator_2" => 0.20,
    "perforator_3" => 0.18,
    "descending_genicular" => 0.18,
)
const FIXED_BRANCH_COLOR = Dict(
    "femoral_trunk" => "#ffb703",
    "profunda_femoris" => "#f77f00",
    "lateral_circumflex_descending" => "#ffd166",
    "medial_circumflex" => "#ef476f",
    "perforator_1" => "#f4a261",
    "perforator_2" => "#e76f51",
    "perforator_3" => "#d62828",
    "descending_genicular" => "#fcbf49",
)
const GROWN_BRANCH_COLOR = "#2a9d8f"
const XCAT_BRANCH_PALETTE = [
    "#ffb703", "#f77f00", "#ef476f", "#06d6a0", "#118ab2",
    "#ffd166", "#f4a261", "#8ecae6", "#e76f51", "#c77dff",
]

function fixed_diameter_cm(label::AbstractString)
    return get(FIXED_BRANCH_DIAMETER_CM, String(label), TERMINAL_DIAMETER_CM)
end

function fixed_branch_color(label::AbstractString)
    label_s = String(label)
    if haskey(FIXED_BRANCH_COLOR, label_s)
        return FIXED_BRANCH_COLOR[label_s]
    end
    seed = 0
    for (idx, byte) in enumerate(codeunits(label_s))
        seed += idx * Int(byte)
    end
    return XCAT_BRANCH_PALETTE[mod(seed, length(XCAT_BRANCH_PALETTE)) + 1]
end

function slice_summary(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                       spacing_cm::SVector{3, Float64}, k::Int)
    dims = size(mask)
    xs = Int[]
    ys = Int[]
    for j in 1:dims[2], i in 1:dims[1]
        mask[i, j, k] || continue
        push!(xs, i)
        push!(ys, j)
    end
    isempty(xs) && return nothing

    min_i, max_i = minimum(xs), maximum(xs)
    min_j, max_j = minimum(ys), maximum(ys)
    mean_i = mean(xs)
    mean_j = mean(ys)
    return (
        min_i=min_i,
        max_i=max_i,
        min_j=min_j,
        max_j=max_j,
        center_i=mean_i,
        center_j=mean_j,
        min_x=origin_cm[1] + (min_i - 0.5) * spacing_cm[1],
        max_x=origin_cm[1] + (max_i - 0.5) * spacing_cm[1],
        min_y=origin_cm[2] + (min_j - 0.5) * spacing_cm[2],
        max_y=origin_cm[2] + (max_j - 0.5) * spacing_cm[2],
        center_x=origin_cm[1] + (mean_i - 0.5) * spacing_cm[1],
        center_y=origin_cm[2] + (mean_j - 0.5) * spacing_cm[2],
        z=origin_cm[3] + (k - 0.5) * spacing_cm[3],
    )
end

function target_offsets_for_fraction(zfrac::Float64)
    # `space: left-posterior-superior`: low y is anterior, high y is posterior.
    # We assume the lower-x side is medial for this cropped thigh smoke test.
    if zfrac >= 0.82
        return -0.12, 0.22
    elseif zfrac >= 0.66
        return -0.18, 0.28
    elseif zfrac >= 0.50
        return -0.20, 0.36
    elseif zfrac >= 0.34
        return -0.16, 0.48
    elseif zfrac >= 0.20
        return -0.08, 0.64
    else
        return 0.02, 0.78
    end
end

function snap_to_mask_near(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                           spacing_cm::SVector{3, Float64}, desired::SVector{3, Float64},
                           k0::Int; z_window::Int=2)
    dims = size(mask)
    best = nothing
    best_d2 = Inf
    for k in max(1, k0 - z_window):min(dims[3], k0 + z_window)
        for j in 1:dims[2], i in 1:dims[1]
            mask[i, j, k] || continue
            p = voxel_center(origin_cm, spacing_cm, i, j, k)
            d = p - desired
            d2 = d[1]^2 + d[2]^2 + 0.25 * d[3]^2
            if d2 < best_d2
                best_d2 = d2
                best = p
            end
        end
    end
    best === nothing && error("Could not snap femoral waypoint near z-index $k0")
    return best
end

function point_at_mask_fraction(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                                spacing_cm::SVector{3, Float64}, mask_info,
                                zfrac::Float64, x_offset::Float64, y_fraction::Float64;
                                z_window::Int=3)
    lo_k = mask_info.lo_idx[3]
    hi_k = mask_info.hi_idx[3]
    k = clamp(round(Int, lo_k + zfrac * (hi_k - lo_k)), lo_k, hi_k)
    ss = slice_summary(mask, origin_cm, spacing_cm, k)
    while ss === nothing && k > lo_k
        k -= 1
        ss = slice_summary(mask, origin_cm, spacing_cm, k)
    end
    ss === nothing && error("No mask voxels found for z fraction $zfrac")

    width = ss.max_x - ss.min_x
    depth = ss.max_y - ss.min_y
    desired = SVector(
        ss.center_x + x_offset * width,
        ss.min_y + y_fraction * depth,
        ss.z,
    )
    return snap_to_mask_near(mask, origin_cm, spacing_cm, desired, k; z_window=z_window)
end

function approximate_femoral_trunk(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                                   spacing_cm::SVector{3, Float64}, mask_info)
    zfracs = collect(range(0.96, 0.08; length=17))
    points = SVector{3, Float64}[]

    for zfrac in zfracs
        x_offset, y_fraction = target_offsets_for_fraction(zfrac)
        push!(points, point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info,
            zfrac, x_offset, y_fraction; z_window=2))
    end

    return points
end

function nearest_vertex_for_zfrac(vertex_ids::Vector{Int}, vertices::Vector{SVector{3, Float64}},
                                  mask_info, zfrac::Float64)
    target_z = mask_info.lo_cm[3] + zfrac * (mask_info.hi_cm[3] - mask_info.lo_cm[3])
    best_vid = vertex_ids[1]
    best_d = Inf
    for vid in vertex_ids
        d = abs(vertices[vid][3] - target_z)
        if d < best_d
            best_d = d
            best_vid = vid
        end
    end
    return best_vid
end

function add_fixed_path!(vertices::Vector{SVector{3, Float64}},
                         parent_vertex::Vector{Int},
                         incoming_segment::Vector{Int},
                         children::Vector{Vector{Int}},
                         segment_start::Vector{Int},
                         segment_end::Vector{Int},
                         segment_diameter_cm::Vector{Float64},
                         segment_label::Vector{String},
                         is_xcat::Vector{Bool},
                         subtree_terminal_count::Vector{Int},
                         anchor_vertex::Int,
                         points::Vector{SVector{3, Float64}},
                         label::String)
    isempty(points) && return Int[]
    diameter_cm = fixed_diameter_cm(label)
    vertex_ids = Int[anchor_vertex]
    prev = anchor_vertex
    start_idx = norm(points[1] - vertices[anchor_vertex]) < 1e-8 ? 2 : 1
    for p in points[start_idx:end]
        if norm(p - vertices[prev]) < 1e-8
            push!(vertex_ids, prev)
            continue
        end
        push!(vertices, p)
        vid = length(vertices)
        push!(parent_vertex, prev)
        push!(incoming_segment, length(segment_start) + 1)
        push!(children, Int[])
        push!(subtree_terminal_count, 0)
        push!(segment_start, prev)
        push!(segment_end, vid)
        push!(segment_diameter_cm, diameter_cm)
        push!(segment_label, label)
        push!(is_xcat, true)
        push!(children[prev], vid)
        push!(vertex_ids, vid)
        prev = vid
    end
    return vertex_ids
end

function make_anatomic_seed_tree(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                                 spacing_cm::SVector{3, Float64}, mask_info,
                                 trunk_points::Vector{SVector{3, Float64}})
    length(trunk_points) >= 2 || error("Need at least two trunk points")

    vertices = SVector{3, Float64}[trunk_points[1]]
    parent_vertex = Int[0]
    incoming_segment = Int[0]
    children = [Int[]]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = Int[0]

    trunk_ids = add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
        subtree_terminal_count, 1, trunk_points, "femoral_trunk")

    profunda_anchor = nearest_vertex_for_zfrac(trunk_ids, vertices, mask_info, 0.84)
    profunda_points = SVector{3, Float64}[
        vertices[profunda_anchor],
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.76, 0.02, 0.48),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.64, 0.10, 0.60),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.52, 0.08, 0.69),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.38, 0.02, 0.76),
    ]
    profunda_ids = add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
        subtree_terminal_count, profunda_anchor, profunda_points, "profunda_femoris")

    lateral_anchor = nearest_vertex_for_zfrac(profunda_ids, vertices, mask_info, 0.76)
    lateral_points = SVector{3, Float64}[
        vertices[lateral_anchor],
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.74, 0.30, 0.30),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.62, 0.42, 0.33),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.48, 0.40, 0.36),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.34, 0.34, 0.42),
    ]
    add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
        subtree_terminal_count, lateral_anchor, lateral_points, "lateral_circumflex_descending")

    medial_anchor = nearest_vertex_for_zfrac(profunda_ids, vertices, mask_info, 0.80)
    medial_points = SVector{3, Float64}[
        vertices[medial_anchor],
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.80, -0.30, 0.54),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.72, -0.38, 0.68),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.66, -0.28, 0.78),
    ]
    add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
        subtree_terminal_count, medial_anchor, medial_points, "medial_circumflex")

    perforator_specs = [
        ("perforator_1", 0.64, 0.28, 0.86, 0.60),
        ("perforator_2", 0.52, 0.34, 0.89, 0.48),
        ("perforator_3", 0.40, 0.30, 0.91, 0.34),
    ]
    for (label, anchor_z, x_offset, y_fraction, tip_z) in perforator_specs
        anchor = nearest_vertex_for_zfrac(profunda_ids, vertices, mask_info, anchor_z)
        points = SVector{3, Float64}[
            vertices[anchor],
            point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, anchor_z - 0.02, x_offset * 0.55, 0.78),
            point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, tip_z, x_offset, y_fraction),
        ]
        add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
            segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
            subtree_terminal_count, anchor, points, label)
    end

    genicular_anchor = nearest_vertex_for_zfrac(trunk_ids, vertices, mask_info, 0.22)
    genicular_points = SVector{3, Float64}[
        vertices[genicular_anchor],
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.17, -0.22, 0.58),
        point_at_mask_fraction(mask, origin_cm, spacing_cm, mask_info, 0.10, -0.34, 0.50),
    ]
    add_fixed_path!(vertices, parent_vertex, incoming_segment, children,
        segment_start, segment_end, segment_diameter_cm, segment_label, is_xcat,
        subtree_terminal_count, genicular_anchor, genicular_points, "descending_genicular")

    tree = GrowthTree(
        "ThighFemoral",
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
        fixed_diameter_cm("femoral_trunk"),
    )
    return tree
end

function preserve_fixed_branch_diameters!(tree::GrowthTree)
    for s in eachindex(tree.segment_label)
        label = tree.segment_label[s]
        haskey(FIXED_BRANCH_DIAMETER_CM, label) || continue
        tree.segment_diameter_cm[s] = fixed_diameter_cm(label)
        tree.is_xcat[s] = true
    end
    return tree
end

function write_fixed_seed_csv(path::AbstractString, tree::GrowthTree)
    open(path, "w") do io
        println(io, "segment_id,label,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,diameter_cm")
        for s in eachindex(tree.segment_start)
            label = tree.segment_label[s]
            haskey(FIXED_BRANCH_DIAMETER_CM, label) || continue
            a = tree.vertices[tree.segment_start[s]]
            b = tree.vertices[tree.segment_end[s]]
            @printf(io, "%d,%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
                s, label, a[1], a[2], a[3], b[1], b[2], b[3], tree.segment_diameter_cm[s])
        end
    end
    return path
end

function write_trunk_csv(path::AbstractString, points::Vector{SVector{3, Float64}})
    open(path, "w") do io
        println(io, "point_id,x_cm,y_cm,z_cm,diameter_cm,label")
        for (idx, p) in enumerate(points)
            @printf(io, "%d,%.5f,%.5f,%.5f,%.5f,%s\n",
                idx, p[1], p[2], p[3], fixed_diameter_cm("femoral_trunk"), "femoral_trunk")
        end
    end
    return path
end

function js_escape(text::AbstractString)
    escaped = replace(String(text), "\\" => "\\\\")
    escaped = replace(escaped, "\"" => "\\\"")
    escaped = replace(escaped, "\n" => "\\n")
    return "\"" * escaped * "\""
end

function js_number_array(values; digits::Int=4)
    fmt = Printf.Format("%.$(digits)f")
    return "[" * join((Printf.format(fmt, v) for v in values), ",") * "]"
end

function js_int_array(values)
    return "[" * join(string.(values), ",") * "]"
end

function region_index_for_point(p::SVector{3, Float64}, center_cm::SVector{3, Float64})
    medial = p[1] < center_cm[1]
    anterior = p[2] < center_cm[2]
    if anterior && medial
        return 0
    elseif anterior && !medial
        return 1
    elseif !anterior && medial
        return 2
    else
        return 3
    end
end

function point_regions(points::Matrix{Float64}, center_cm::SVector{3, Float64})
    regions = Vector{Int}(undef, size(points, 1))
    for i in axes(points, 1)
        regions[i] = region_index_for_point(SVector(points[i, 1], points[i, 2], points[i, 3]), center_cm)
    end
    return regions
end

function mask_bitpack_base64(mask::BitArray{3})
    dims = size(mask)
    nbits = prod(dims)
    packed = zeros(UInt8, cld(nbits, 8))
    idx = 0
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        if mask[i, j, k]
            packed[div(idx, 8) + 1] |= UInt8(1) << (idx % 8)
        end
        idx += 1
    end
    return base64encode(packed)
end

function segments_js_array(tree::GrowthTree)
    io = IOBuffer()
    print(io, "[")
    for s in eachindex(tree.segment_start)
        s > 1 && print(io, ",")
        a = tree.vertices[tree.segment_start[s]]
        b = tree.vertices[tree.segment_end[s]]
        label = tree.segment_label[s]
        fixed = tree.is_xcat[s] || haskey(FIXED_BRANCH_DIAMETER_CM, label)
        color = fixed ? fixed_branch_color(label) : GROWN_BRANCH_COLOR
        print(io, "{")
        @printf(io, "\"id\":%d,", s)
        print(io, "\"label\":", js_escape(label), ",")
        print(io, "\"fixed\":", fixed ? "true" : "false", ",")
        @printf(io, "\"diameter\":%.5f,", tree.segment_diameter_cm[s])
        print(io, "\"color\":", js_escape(color), ",")
        @printf(io, "\"x1\":%.5f,\"y1\":%.5f,\"z1\":%.5f,", a[1], a[2], a[3])
        @printf(io, "\"x2\":%.5f,\"y2\":%.5f,\"z2\":%.5f", b[1], b[2], b[3])
        print(io, "}")
    end
    print(io, "]")
    return String(take!(io))
end

function nodes_js_array(tree::GrowthTree)
    fixed_vertex = falses(length(tree.vertices))
    degrees = zeros(Int, length(tree.vertices))
    for s in eachindex(tree.segment_start)
        a = tree.segment_start[s]
        b = tree.segment_end[s]
        degrees[a] += 1
        degrees[b] += 1
        if tree.is_xcat[s] || haskey(FIXED_BRANCH_DIAMETER_CM, tree.segment_label[s])
            fixed_vertex[a] = true
            fixed_vertex[b] = true
        end
    end

    io = IOBuffer()
    print(io, "[")
    for (idx, p) in enumerate(tree.vertices)
        idx > 1 && print(io, ",")
        print(io, "{")
        @printf(io, "\"id\":%d,\"x\":%.5f,\"y\":%.5f,\"z\":%.5f,", idx, p[1], p[2], p[3])
        @printf(io, "\"degree\":%d,", degrees[idx])
        print(io, "\"fixed\":", fixed_vertex[idx] ? "true" : "false")
        print(io, "}")
    end
    print(io, "]")
    return String(take!(io))
end

function fixed_branch_metadata_js(tree::GrowthTree)
    labels = String[]
    for label in FIXED_BRANCH_ORDER
        any(s -> tree.segment_label[s] == label && tree.is_xcat[s], eachindex(tree.segment_label)) || continue
        push!(labels, label)
    end
    for s in eachindex(tree.segment_label)
        tree.is_xcat[s] || continue
        label = tree.segment_label[s]
        label in labels && continue
        push!(labels, label)
    end

    entries = String[]
    for label in labels
        diameters = [tree.segment_diameter_cm[s] for s in eachindex(tree.segment_label)
            if tree.segment_label[s] == label && tree.is_xcat[s]]
        diameter = isempty(diameters) ? fixed_diameter_cm(label) : mean(diameters)
        push!(entries, "{" *
            "\"label\":" * js_escape(label) * "," *
            "\"diameter\":" * @sprintf("%.5f", diameter) * "," *
            "\"color\":" * js_escape(fixed_branch_color(label)) *
            "}")
    end
    return "[" * join(entries, ",") * "]"
end

function write_femoral_viewer(path::AbstractString, mask::BitArray{3}, origin_cm::SVector{3, Float64},
                              spacing_cm::SVector{3, Float64}, mask_info,
                              domain_points::Matrix{Float64}, tree::GrowthTree)
    display_points = VascularTreeSim._sample_points_for_viewer(domain_points; max_points=26_000)
    regions = point_regions(display_points, mask_info.center_cm)
    dx = js_number_array(display_points[:, 1])
    dy = js_number_array(display_points[:, 2])
    dz = js_number_array(display_points[:, 3])
    dr = js_int_array(regions)
    dims = size(mask)
    mask_b64 = mask_bitpack_base64(mask)
    segments_js = segments_js_array(tree)
    nodes_js = nodes_js_array(tree)
    fixed_js = fixed_branch_metadata_js(tree)
    origin_js = js_number_array(collect(origin_cm); digits=5)
    spacing_js = js_number_array(collect(spacing_cm); digits=5)
    center_js = js_number_array(collect(mask_info.center_cm); digits=5)

    html = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Raw NRRD Thigh Femoral Simulation</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      color-scheme: dark;
      --bg: #111318;
      --panel: #191d24;
      --panel-2: #202631;
      --line: #343b48;
      --text: #e7ebf2;
      --muted: #a9b1bf;
      --accent: #ffb703;
      --green: #2a9d8f;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; height: 100%; background: var(--bg); color: var(--text); font-family: Arial, sans-serif; }
    body { overflow: hidden; }
    #app { display: grid; grid-template-columns: 300px minmax(0, 1fr); height: 100vh; }
    #controls { background: var(--panel); border-right: 1px solid var(--line); padding: 14px; overflow: auto; }
    #stage { display: grid; grid-template-rows: minmax(360px, 58vh) minmax(280px, 42vh); min-width: 0; }
    #plot3d { min-width: 0; min-height: 0; }
    #slices { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); border-top: 1px solid var(--line); background: #0d0f14; min-height: 0; }
    .slice-panel { min-width: 0; min-height: 0; border-right: 1px solid var(--line); display: grid; grid-template-rows: auto minmax(0, 1fr); }
    .slice-panel:last-child { border-right: 0; }
    .slice-head { display: grid; grid-template-columns: 78px minmax(0, 1fr) 58px; gap: 8px; align-items: center; padding: 8px 10px; background: #151922; border-bottom: 1px solid var(--line); }
    .slice-title { color: var(--text); font-size: 12px; font-weight: 700; }
    .slice-value { color: var(--muted); font-size: 11px; text-align: right; font-variant-numeric: tabular-nums; }
    .canvas-wrap { display: grid; place-items: center; min-height: 0; padding: 8px; }
    canvas { width: 100%; height: 100%; max-width: 100%; max-height: 100%; object-fit: contain; image-rendering: pixelated; background: #05070a; border: 1px solid #29303b; }
    h1 { font-size: 16px; line-height: 1.2; margin: 0 0 12px; letter-spacing: 0; }
    .group { border-top: 1px solid var(--line); padding: 12px 0; }
    .group:first-of-type { border-top: 0; padding-top: 0; }
    .group-title { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0; margin-bottom: 8px; }
    label { display: flex; align-items: center; gap: 8px; min-height: 25px; font-size: 13px; color: var(--text); }
    input[type="checkbox"] { accent-color: var(--green); }
    input[type="range"] { width: 100%; accent-color: var(--accent); }
    select { width: 100%; background: var(--panel-2); color: var(--text); border: 1px solid var(--line); border-radius: 4px; padding: 6px 8px; }
    .row { display: grid; grid-template-columns: minmax(0, 1fr) 60px; align-items: center; gap: 8px; margin: 7px 0; }
    .value { color: var(--muted); font-size: 12px; text-align: right; font-variant-numeric: tabular-nums; }
    .swatch { width: 10px; height: 10px; border-radius: 50%; border: 1px solid rgba(255,255,255,.25); flex: 0 0 auto; }
    .note { color: var(--muted); font-size: 12px; line-height: 1.35; margin-top: 8px; }
    @media (max-width: 1100px) {
      #app { grid-template-columns: 250px minmax(0, 1fr); }
      #slices { grid-template-columns: 1fr; overflow: auto; }
      .slice-panel { min-height: 260px; border-right: 0; border-bottom: 1px solid var(--line); }
    }
  </style>
</head>
<body>
<div id="app">
  <aside id="controls">
    <h1>Thigh Vessel Viewer</h1>
    <div class="group">
      <div class="group-title">Layers</div>
      <label><input id="showMask" type="checkbox" checked> Mask sample</label>
      <label><input id="showPredicted" type="checkbox" checked> Predicted branches</label>
      <label><input id="showFixed" type="checkbox" checked> Fixed anatomy seeds</label>
      <label><input id="showNodes" type="checkbox"> Nodes</label>
    </div>
    <div class="group">
      <div class="group-title">Seed Branches</div>
      <div id="branchToggles"></div>
    </div>
    <div class="group">
      <div class="group-title">Mask Regions</div>
      <label><input class="regionToggle" type="checkbox" value="0" checked> Anteromedial</label>
      <label><input class="regionToggle" type="checkbox" value="1" checked> Anterolateral</label>
      <label><input class="regionToggle" type="checkbox" value="2" checked> Posteromedial</label>
      <label><input class="regionToggle" type="checkbox" value="3" checked> Posterolateral</label>
    </div>
    <div class="group">
      <div class="group-title">3D Slicer</div>
      <label><input id="clipEnabled" type="checkbox"> Slice 3D view</label>
      <div class="row">
        <select id="clipAxis">
          <option value="z">Axial Z</option>
          <option value="y">Coronal Y</option>
          <option value="x">Sagittal X</option>
        </select>
        <span id="clipAxisValue" class="value"></span>
      </div>
      <div class="row"><input id="clipCenter" type="range" min="0" max="389" value="195"><span id="clipCenterValue" class="value"></span></div>
      <div class="row"><input id="clipThickness" type="range" min="1" max="80" value="18"><span id="clipThicknessValue" class="value"></span></div>
    </div>
  </aside>
  <main id="stage">
    <div id="plot3d"></div>
    <section id="slices">
      <div class="slice-panel">
        <div class="slice-head"><div class="slice-title">Axial</div><input id="axialSlider" type="range"><div id="axialValue" class="slice-value"></div></div>
        <div class="canvas-wrap"><canvas id="axialCanvas"></canvas></div>
      </div>
      <div class="slice-panel">
        <div class="slice-head"><div class="slice-title">Coronal</div><input id="coronalSlider" type="range"><div id="coronalValue" class="slice-value"></div></div>
        <div class="canvas-wrap"><canvas id="coronalCanvas"></canvas></div>
      </div>
      <div class="slice-panel">
        <div class="slice-head"><div class="slice-title">Sagittal</div><input id="sagittalSlider" type="range"><div id="sagittalValue" class="slice-value"></div></div>
        <div class="canvas-wrap"><canvas id="sagittalCanvas"></canvas></div>
      </div>
    </section>
  </main>
</div>
<script>
const metadata = {
  dims: [$(dims[1]), $(dims[2]), $(dims[3])],
  origin: $origin_js,
  spacing: $spacing_js,
  center: $center_js
};
const maskPoints = { x: $dx, y: $dy, z: $dz, region: $dr };
const fixedBranches = $fixed_js;
const segments = $segments_js;
const nodes = $nodes_js;
const maskBitsB64 = "$(mask_b64)";
let plotInitialized = false;
let cameraState = null;

const axisIndex = { x: 0, y: 1, z: 2 };
const axisCoord = { x: "x", y: "y", z: "z" };
const regionNames = ["Anteromedial", "Anterolateral", "Posteromedial", "Posterolateral"];

function decodeMaskBits(b64) {
  const binary = atob(b64);
  const out = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) out[i] = binary.charCodeAt(i);
  return out;
}

const maskBits = decodeMaskBits(maskBitsB64);

function voxel(i, j, k) {
  const [nx, ny, nz] = metadata.dims;
  if (i < 0 || j < 0 || k < 0 || i >= nx || j >= ny || k >= nz) return false;
  const idx = i + nx * (j + ny * k);
  return ((maskBits[idx >> 3] >> (idx & 7)) & 1) === 1;
}

function cmToIndex(value, axis) {
  const ai = axisIndex[axis];
  return (value - metadata.origin[ai]) / metadata.spacing[ai] - 0.5;
}

function indexToCm(idx, axis) {
  const ai = axisIndex[axis];
  return metadata.origin[ai] + (idx + 0.5) * metadata.spacing[ai];
}

function regionForIndex(i, j) {
  const x = metadata.origin[0] + (i + 0.5) * metadata.spacing[0];
  const y = metadata.origin[1] + (j + 0.5) * metadata.spacing[1];
  const medial = x < metadata.center[0];
  const anterior = y < metadata.center[1];
  if (anterior && medial) return 0;
  if (anterior && !medial) return 1;
  if (!anterior && medial) return 2;
  return 3;
}

function activeRegions() {
  const selected = new Set();
  document.querySelectorAll(".regionToggle").forEach(cb => {
    if (cb.checked) selected.add(Number(cb.value));
  });
  return selected;
}

function activeBranchLabels() {
  const selected = new Set();
  document.querySelectorAll(".branchToggle").forEach(cb => {
    if (cb.checked) selected.add(cb.value);
  });
  return selected;
}

function branchColor(label) {
  const found = fixedBranches.find(b => b.label === label);
  return found ? found.color : "#ffb703";
}

function axisBounds() {
  const axis = document.getElementById("clipAxis").value;
  if (!document.getElementById("clipEnabled").checked) return null;
  const centerIdx = Number(document.getElementById("clipCenter").value);
  const thicknessIdx = Number(document.getElementById("clipThickness").value);
  const center = indexToCm(centerIdx, axis);
  const half = Math.max(thicknessIdx, 1) * metadata.spacing[axisIndex[axis]] * 0.5;
  return { axis, min: center - half, max: center + half, center };
}

function pointInClip(x, y, z, clip) {
  if (!clip) return true;
  const v = clip.axis === "x" ? x : clip.axis === "y" ? y : z;
  return v >= clip.min && v <= clip.max;
}

function segmentMidInClip(seg, clip) {
  if (!clip) return true;
  const x = 0.5 * (seg.x1 + seg.x2);
  const y = 0.5 * (seg.y1 + seg.y2);
  const z = 0.5 * (seg.z1 + seg.z2);
  return pointInClip(x, y, z, clip);
}

function lineArraysForSegments(list) {
  const x = [];
  const y = [];
  const z = [];
  for (const seg of list) {
    x.push(seg.x1, seg.x2, null);
    y.push(seg.y1, seg.y2, null);
    z.push(seg.z1, seg.z2, null);
  }
  return { x, y, z };
}

function diameterUm(diameterCm) {
  return diameterCm * 10000.0;
}

function diameterLineWidth(diameterCm) {
  const um = Math.max(diameterUm(diameterCm), 0.0);
  return Math.max(0.8, Math.min(8.5, 0.8 + 0.09 * Math.sqrt(um)));
}

const predictedDiameterBins = [
  { max: 10, label: "<=10 um", color: "#63d7cf" },
  { max: 25, label: "10-25 um", color: "#43c7b5" },
  { max: 50, label: "25-50 um", color: "#2a9d8f" },
  { max: 100, label: "50-100 um", color: "#268f83" },
  { max: 200, label: "100-200 um", color: "#4f9f62" },
  { max: 400, label: "200-400 um", color: "#8ab84f" },
  { max: 800, label: "400-800 um", color: "#d6a83f" },
  { max: Infinity, label: ">800 um", color: "#f77f00" }
];

function predictedDiameterBin(seg) {
  const um = diameterUm(seg.diameter);
  for (const bin of predictedDiameterBins) {
    if (um <= bin.max) return bin;
  }
  return predictedDiameterBins[predictedDiameterBins.length - 1];
}

function representativeBinDiameterCm(bin, observed) {
  if (observed.length > 0) {
    const total = observed.reduce((sum, seg) => sum + seg.diameter, 0.0);
    return total / observed.length;
  }
  return Math.min(bin.max, 1000.0) / 10000.0;
}

function make3dTraces() {
  const traces = [];
  const clip = axisBounds();
  const regions = activeRegions();
  const branchLabels = activeBranchLabels();
  if (document.getElementById("showMask").checked) {
    const x = [];
    const y = [];
    const z = [];
    for (let i = 0; i < maskPoints.x.length; i++) {
      if (!regions.has(maskPoints.region[i])) continue;
      if (!pointInClip(maskPoints.x[i], maskPoints.y[i], maskPoints.z[i], clip)) continue;
      x.push(maskPoints.x[i]);
      y.push(maskPoints.y[i]);
      z.push(maskPoints.z[i]);
    }
    traces.push({
      type: "scatter3d", mode: "markers", name: "Mask sample",
      x, y, z,
      marker: { size: 1.45, color: "#9aa4b2", opacity: 0.16 },
      hoverinfo: "skip"
    });
  }

  if (document.getElementById("showFixed").checked) {
    for (const branch of fixedBranches) {
      if (!branchLabels.has(branch.label)) continue;
      const list = segments.filter(s => s.fixed && s.label === branch.label && segmentMidInClip(s, clip));
      if (list.length === 0) continue;
      const arrays = lineArraysForSegments(list);
      traces.push({
        type: "scatter3d", mode: "lines", name: branch.label,
        x: arrays.x, y: arrays.y, z: arrays.z,
        line: { color: branch.color, width: diameterLineWidth(branch.diameter) },
        hoverinfo: "name"
      });
    }
  }

  if (document.getElementById("showPredicted").checked) {
    const visiblePredicted = segments.filter(s => !s.fixed && segmentMidInClip(s, clip));
    for (const bin of predictedDiameterBins) {
      const list = visiblePredicted.filter(s => predictedDiameterBin(s) === bin);
      if (list.length === 0) continue;
      const arrays = lineArraysForSegments(list);
      traces.push({
        type: "scatter3d", mode: "lines", name: `Predicted \${bin.label}`,
        x: arrays.x, y: arrays.y, z: arrays.z,
        line: { color: bin.color, width: diameterLineWidth(representativeBinDiameterCm(bin, list)) },
        hoverinfo: "name"
      });
    }
  }

  if (document.getElementById("showNodes").checked) {
    const x = [];
    const y = [];
    const z = [];
    const color = [];
    const text = [];
    for (const node of nodes) {
      if (!pointInClip(node.x, node.y, node.z, clip)) continue;
      x.push(node.x); y.push(node.y); z.push(node.z);
      color.push(node.fixed ? "#ffd166" : "#d9fff8");
      text.push(`node \${node.id}<br>degree \${node.degree}`);
    }
    traces.push({
      type: "scatter3d", mode: "markers", name: "Nodes",
      x, y, z, text,
      marker: { size: 2.8, color, opacity: 0.92 },
      hovertemplate: "%{text}<extra></extra>"
    });
  }
  return traces;
}

function render3d() {
  const traces = make3dTraces();
  const layout = {
    paper_bgcolor: "#111318",
    plot_bgcolor: "#111318",
    margin: { l: 0, r: 0, t: 0, b: 0 },
    scene: {
      bgcolor: "#111318",
      xaxis: { title: "x left (cm)", color: "#cbd2dd", gridcolor: "#303844", zerolinecolor: "#444b57" },
      yaxis: { title: "y posterior (cm)", color: "#cbd2dd", gridcolor: "#303844", zerolinecolor: "#444b57" },
      zaxis: { title: "z superior (cm)", color: "#cbd2dd", gridcolor: "#303844", zerolinecolor: "#444b57" },
      aspectmode: "data"
    },
    legend: { x: 0.99, xanchor: "right", y: 0.98, font: { color: "#e7ebf2" }, bgcolor: "rgba(16,18,24,.72)" }
  };
  if (cameraState) layout.scene.camera = cameraState;
  const config = { displaylogo: false, responsive: true };
  const plot = document.getElementById("plot3d");
  const promise = plotInitialized ? Plotly.react(plot, traces, layout, config) : Plotly.newPlot(plot, traces, layout, config);
  plotInitialized = true;
  promise.then(() => {
    plot.on("plotly_relayout", ev => {
      if (ev["scene.camera"]) cameraState = ev["scene.camera"];
    });
  });
}

function segmentPlaneOverlay(ctx, axis, sliceIdx, width, height, mode) {
  const plane = indexToCm(sliceIdx, axis);
  const axisKey = axisCoord[axis];
  const branchLabels = activeBranchLabels();
  const drawPredicted = document.getElementById("showPredicted").checked;
  const drawFixed = document.getElementById("showFixed").checked;
  const drawNodes = document.getElementById("showNodes").checked;
  const thickness = metadata.spacing[axisIndex[axis]] * 0.75;

  function toXY(x, y, z) {
    const ii = cmToIndex(x, "x") + 0.5;
    const jj = cmToIndex(y, "y") + 0.5;
    const kk = cmToIndex(z, "z") + 0.5;
    if (axis === "z") return [ii / metadata.dims[0] * width, height - jj / metadata.dims[1] * height];
    if (axis === "y") return [ii / metadata.dims[0] * width, height - kk / metadata.dims[2] * height];
    return [jj / metadata.dims[1] * width, height - kk / metadata.dims[2] * height];
  }

  for (const seg of segments) {
    if (seg.fixed && (!drawFixed || !branchLabels.has(seg.label))) continue;
    if (!seg.fixed && !drawPredicted) continue;
    const a = seg[axisKey + "1"];
    const b = seg[axisKey + "2"];
    const da = a - plane;
    const db = b - plane;
    const color = seg.fixed ? branchColor(seg.label) : "#2a9d8f";
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.lineWidth = diameterLineWidth(seg.diameter);

    if (Math.abs(da) <= thickness && Math.abs(db) <= thickness) {
      const p1 = toXY(seg.x1, seg.y1, seg.z1);
      const p2 = toXY(seg.x2, seg.y2, seg.z2);
      ctx.beginPath();
      ctx.moveTo(p1[0], p1[1]);
      ctx.lineTo(p2[0], p2[1]);
      ctx.stroke();
    } else if (da * db <= 0) {
      const t = Math.abs(b - a) < 1e-8 ? 0.0 : (plane - a) / (b - a);
      if (t >= -0.001 && t <= 1.001) {
        const x = seg.x1 + t * (seg.x2 - seg.x1);
        const y = seg.y1 + t * (seg.y2 - seg.y1);
        const z = seg.z1 + t * (seg.z2 - seg.z1);
        const p = toXY(x, y, z);
        const r = Math.max(1.5, diameterLineWidth(seg.diameter) * 0.85);
        ctx.beginPath();
        ctx.arc(p[0], p[1], r, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  if (drawNodes) {
    for (const node of nodes) {
      const v = axis === "x" ? node.x : axis === "y" ? node.y : node.z;
      if (Math.abs(v - plane) > thickness) continue;
      const p = toXY(node.x, node.y, node.z);
      ctx.fillStyle = node.fixed ? "#ffd166" : "#d9fff8";
      ctx.beginPath();
      ctx.arc(p[0], p[1], node.fixed ? 3.5 : 2.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function drawSlice(canvasId, axis, idx, valueId) {
  const [nx, ny, nz] = metadata.dims;
  const canvas = document.getElementById(canvasId);
  const regions = activeRegions();
  let w, h;
  if (axis === "z") { w = nx; h = ny; }
  else if (axis === "y") { w = nx; h = nz; }
  else { w = ny; h = nz; }
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(w, h);
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      let i, j, k;
      if (axis === "z") {
        i = px; j = ny - 1 - py; k = idx;
      } else if (axis === "y") {
        i = px; j = idx; k = nz - 1 - py;
      } else {
        i = idx; j = px; k = nz - 1 - py;
      }
      const visibleRegion = regions.has(regionForIndex(i, j));
      const on = voxel(i, j, k) && visibleRegion;
      const offset = 4 * (py * w + px);
      const v = on ? 118 : 11;
      image.data[offset] = v;
      image.data[offset + 1] = on ? 127 : 13;
      image.data[offset + 2] = on ? 138 : 18;
      image.data[offset + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  segmentPlaneOverlay(ctx, axis, idx, w, h);
  document.getElementById(valueId).textContent = `\${idx + 1} / \${metadata.dims[axisIndex[axis]].toString()}  \${indexToCm(idx, axis).toFixed(2)} cm`;
}

function renderSlices() {
  drawSlice("axialCanvas", "z", Number(document.getElementById("axialSlider").value), "axialValue");
  drawSlice("coronalCanvas", "y", Number(document.getElementById("coronalSlider").value), "coronalValue");
  drawSlice("sagittalCanvas", "x", Number(document.getElementById("sagittalSlider").value), "sagittalValue");
}

function updateClipControls() {
  const axis = document.getElementById("clipAxis").value;
  const ai = axisIndex[axis];
  const maxIdx = metadata.dims[ai] - 1;
  const center = document.getElementById("clipCenter");
  const current = Math.min(Number(center.value), maxIdx);
  center.max = String(maxIdx);
  center.value = String(current);
  document.getElementById("clipAxisValue").textContent = axis.toUpperCase();
  document.getElementById("clipCenterValue").textContent = indexToCm(current, axis).toFixed(2);
  const thickness = Number(document.getElementById("clipThickness").value);
  document.getElementById("clipThicknessValue").textContent = `\${(thickness * metadata.spacing[ai] * 10).toFixed(1)} mm`;
}

function renderAll() {
  updateClipControls();
  render3d();
  renderSlices();
}

function initControls() {
  const branchBox = document.getElementById("branchToggles");
  for (const branch of fixedBranches) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.className = "branchToggle";
    cb.value = branch.label;
    cb.checked = true;
    const swatch = document.createElement("span");
    swatch.className = "swatch";
    swatch.style.background = branch.color;
    const text = document.createElement("span");
    text.textContent = branch.label;
    label.append(cb, swatch, text);
    branchBox.appendChild(label);
  }

  const [nx, ny, nz] = metadata.dims;
  const axial = document.getElementById("axialSlider");
  axial.min = "0"; axial.max = String(nz - 1); axial.value = String(Math.round(cmToIndex(metadata.center[2], "z")));
  const coronal = document.getElementById("coronalSlider");
  coronal.min = "0"; coronal.max = String(ny - 1); coronal.value = String(Math.round(cmToIndex(metadata.center[1], "y")));
  const sagittal = document.getElementById("sagittalSlider");
  sagittal.min = "0"; sagittal.max = String(nx - 1); sagittal.value = String(Math.round(cmToIndex(metadata.center[0], "x")));

  document.querySelectorAll("input, select").forEach(el => {
    el.addEventListener("input", renderAll);
    el.addEventListener("change", renderAll);
  });
  updateClipControls();
}

initControls();
renderAll();
window.addEventListener("resize", () => Plotly.Plots.resize(document.getElementById("plot3d")));
</script>
</body>
</html>
"""
    open(path, "w") do io
        write(io, html)
    end
    return path
end

function write_femoral_summary(path::AbstractString, nhdr_path::AbstractString, raw_path::AbstractString,
                               fields, domain::VoxelShellDomain, mask_info, trunk_points,
                               coverage_points::Matrix{Float64}, graph_points::Matrix{Float64},
                               tree::GrowthTree, stats)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    trunk_len_cm = sum(norm(trunk_points[i] - trunk_points[i - 1]) for i in 2:length(trunk_points))
    fixed_segments = count(label -> haskey(FIXED_BRANCH_DIAMETER_CM, label), tree.segment_label)
    fixed_counts = Dict(label => count(==(label), tree.segment_label) for label in FIXED_BRANCH_ORDER)
    open(path, "w") do io
        println(io, "# Raw + NRRD Thigh Femoral Smoke Test")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- NHDR: $(nhdr_path)")
        println(io, "- RAW: $(raw_path)")
        println(io, "- Label used: $(LABEL_VALUE)")
        println(io, "- NRRD sizes: $(get(fields, "sizes", "unknown"))")
        println(io, "- Spacing: $(round.(Tuple(domain.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Foreground voxels: $(mask_info.foreground)")
        println(io, "- Bounding box extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        println(io, "- Approximate femoral trunk points: $(length(trunk_points))")
        println(io, "- Approximate femoral trunk length: $(round(trunk_len_cm * 10; digits=2)) mm")
        println(io, "- Approximate femoral trunk diameter: $(round(fixed_diameter_cm("femoral_trunk") * 10; digits=2)) mm")
        println(io, "- Coverage points: $(size(coverage_points, 1))")
        println(io, "- Graph points: $(size(graph_points, 1))")
        println(io)
        println(io, "## Anatomy Approximation")
        println(io)
        println(io, "- Femoral artery seed follows the anterior/medial proximal thigh, remains in the adductor-canal side of the mid thigh, and shifts posteriorly near the distal adductor hiatus.")
        println(io, "- Profunda femoris seed branches from the proximal femoral artery and tracks deeper/posterior in the thigh.")
        println(io, "- Lateral circumflex seed runs toward the anterior-lateral thigh; medial circumflex seed runs proximal/posteromedial.")
        println(io, "- Three perforator-style seeds leave profunda femoris toward the posterior compartment.")
        println(io, "- A distal descending-genicular-style seed leaves the femoral trunk near the adductor canal.")
        println(io, "- Every fixed seed point is snapped to the nearest foreground voxel in the binary thigh mask.")
        println(io)
        println(io, "## Anatomy Sources Used")
        println(io)
        println(io, "- https://en.wikipedia.org/wiki/Femoral_artery")
        println(io, "- https://en.wikipedia.org/wiki/Deep_femoral_artery")
        println(io, "- https://en.wikipedia.org/wiki/Lateral_circumflex_femoral_artery")
        println(io, "- https://en.wikipedia.org/wiki/Medial_circumflex_femoral_artery")
        println(io, "- https://en.wikipedia.org/wiki/Perforating_arteries")
        println(io, "- https://en.wikipedia.org/wiki/Adductor_canal")
        println(io)
        println(io, "## Fixed Seeds")
        println(io)
        for label in FIXED_BRANCH_ORDER
            println(io, "- $(label): $(fixed_counts[label]) segments, $(round(fixed_diameter_cm(label) * 10; digits=2)) mm")
        end
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- Fixed seed segments: $(fixed_segments)")
        println(io, "- Predicted/grown segments: $(length(tree.segment_start) - fixed_segments)")
        println(io, "- Total segments: $(length(tree.segment_start))")
        println(io, "- Terminal vertices: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum diameter: $(round(maximum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "The raw mask is binary, so the viewer's region toggles are coarse anteromedial/anterolateral/posteromedial/posterolateral partitions rather than true named-muscle labels.")
        println(io, "This is a coarse local smoke test, not a segmented-artery or capillary-resolution run.")
    end
    return path
end

function main_femoral()
    Random.seed!(42)
    nhdr_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NHDR
    max_branches = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 120
    mkpath(FEMORAL_OUTPUT_DIR)

    println("=" ^ 72)
    println("Raw + NRRD thigh femoral smoke test")
    println("NHDR: $(nhdr_path)")
    println("Output: $(FEMORAL_OUTPUT_DIR)")
    println("=" ^ 72)
    flush(stdout)

    mask, origin_cm, spacing_cm, raw_path, fields = load_binary_nrrd_mask(nhdr_path)
    domain, mask_info = build_domain_from_mask(mask, origin_cm, spacing_cm)
    trunk_points = approximate_femoral_trunk(mask, origin_cm, spacing_cm, mask_info)
    tree = make_anatomic_seed_tree(mask, origin_cm, spacing_cm, mask_info, trunk_points)
    trees = Dict("ThighFemoral" => tree)

    coverage_points = coverage_target_points_blockwise(domain; block_size=COVERAGE_BLOCK_SIZE)
    graph_points = copy(coverage_points)
    viewer_points = coverage_target_points_blockwise(domain; block_size=VIEWER_BLOCK_SIZE)

    println("[nrrd] dims=$(size(mask)) spacing_cm=$(spacing_cm) foreground=$(count(mask))")
    println("[seeds] fixed_segments=$(length(tree.segment_start)) trunk_points=$(length(trunk_points)) proximal=$(first(trunk_points)) distal=$(last(trunk_points))")
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) max_branches=$(max_branches)")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.45,
        capillary_diameter_cm=0.02,
        max_new_branches_per_tree=max_branches,
        graph_neighbors=12,
        min_frontier_separation_cm=0.75,
        max_path_nodes=18,
        frontier_batch=8,
        gamma=3.0,
        smooth_passes=8,
        spline_density=3,
        max_segment_length_cm=0.35,
        turn_penalty=0.35,
        graph_jitter_cm=0.04,
        use_gpu=false)

    preserve_fixed_branch_diameters!(tree)
    st = growth_stats["ThighFemoral"]
    csv_path = joinpath(FEMORAL_OUTPUT_DIR, "thigh_raw_nrrd_femoral_segments.csv")
    trunk_csv = joinpath(FEMORAL_OUTPUT_DIR, "approx_femoral_trunk_points.csv")
    seed_csv = joinpath(FEMORAL_OUTPUT_DIR, "approx_femoral_seed_segments.csv")
    domain_csv = joinpath(FEMORAL_OUTPUT_DIR, "thigh_raw_nrrd_femoral_domain_points.csv")
    viewer_path = joinpath(FEMORAL_OUTPUT_DIR, "thigh_raw_nrrd_femoral_viewer.html")
    summary_path = joinpath(FEMORAL_OUTPUT_DIR, "summary.md")

    write_growth_csv(csv_path, "ThighFemoral", tree)
    write_trunk_csv(trunk_csv, trunk_points)
    write_fixed_seed_csv(seed_csv, tree)
    write_points_csv(domain_csv, viewer_points)
    write_femoral_viewer(viewer_path, mask, origin_cm, spacing_cm, mask_info, viewer_points, tree)
    write_femoral_summary(summary_path, nhdr_path, raw_path, fields, domain, mask_info,
        trunk_points, coverage_points, graph_points, tree, st)

    fixed_segments = count(label -> haskey(FIXED_BRANCH_DIAMETER_CM, label), tree.segment_label)
    println()
    println("Result")
    println("  fixed seed segments  = $(fixed_segments)")
    println("  added branches       = $(st.added)")
    println("  total segments       = $(length(tree.segment_start))")
    println("  p95 coverage         = $(round(st.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(csv_path)")
    println("  $(trunk_csv)")
    println("  $(seed_csv)")
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_femoral()
end
