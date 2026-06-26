"""
    thigh_raw_nrrd_smoke.jl

Legacy smoke-test helper for a binary thigh mask stored as detached NRRD.
Direct execution is disabled: current right-leg XCAT workflows are
NRB-input-only. Helper functions remain available to the NRB runner.

Usage:
    julia --project=. examples/right_leg_xcat_50um_gpu.jl [terminal_um] [output_dir] [nrb_path] ...
"""

using VascularTreeSim
using StaticArrays
using Random
using Printf
using Dates
using LinearAlgebra
using Statistics

const DEFAULT_NHDR = joinpath(dirname(@__DIR__), "output", "thigh_raw_nrrd_smoke", "input", "thigh_muscle.nhdr")
const OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "thigh_raw_nrrd_smoke")
const LABEL_VALUE = UInt8(255)
const TERMINAL_DIAMETER_CM = 0.06
const COVERAGE_BLOCK_SIZE = 8
const VIEWER_BLOCK_SIZE = 5

function legacy_raw_entrypoint_error(script::AbstractString)
    error("$script is a legacy raw/NRRD entry point and is disabled. Use examples/right_leg_xcat_50um_gpu.jl with a `.nrb` input instead.")
end

function parse_nhdr(path::AbstractString)
    fields = Dict{String, String}()
    for raw_line in eachline(path)
        line = strip(raw_line)
        isempty(line) && continue
        startswith(line, "#") && continue
        startswith(line, "NRRD") && continue
        parts = split(line, ":"; limit=2)
        length(parts) == 2 || continue
        fields[strip(lowercase(parts[1]))] = strip(parts[2])
    end
    return fields
end

function parse_sizes(value::AbstractString)
    nums = parse.(Int, split(strip(value)))
    length(nums) == 3 || error("Only 3-D NRRD volumes are supported; got sizes=$value")
    return Tuple(nums)
end

function parse_vector_tuple(text::AbstractString)
    vals = parse.(Float64, split(replace(strip(text), "," => " ")))
    length(vals) == 3 || error("Expected 3-vector, got `$text`")
    return SVector(vals[1], vals[2], vals[3])
end

function parse_space_directions(value::AbstractString)
    matches = collect(eachmatch(r"\(([^)]*)\)", value))
    length(matches) == 3 || error("Expected three space direction vectors, got `$value`")
    return Tuple(parse_vector_tuple(m.captures[1]) for m in matches)
end

function parse_space_origin(value::AbstractString)
    m = match(r"\(([^)]*)\)", value)
    m === nothing && error("Expected space origin vector, got `$value`")
    return parse_vector_tuple(m.captures[1])
end

function nrrd_spacing_cm(directions)
    return SVector(norm(directions[1]), norm(directions[2]), norm(directions[3])) .* 0.1
end

function load_binary_nrrd_mask(nhdr_path::AbstractString; label_value::UInt8=LABEL_VALUE)
    fields = parse_nhdr(nhdr_path)
    get(fields, "type", "") in ("uchar", "uint8", "unsigned char") ||
        error("Only uchar NRRD volumes are supported in this smoke test; got type=$(get(fields, "type", "missing"))")
    get(fields, "encoding", "") == "raw" ||
        error("Only raw-encoded NRRD volumes are supported; got encoding=$(get(fields, "encoding", "missing"))")

    dims = parse_sizes(fields["sizes"])
    directions = parse_space_directions(fields["space directions"])
    origin_mm = parse_space_origin(get(fields, "space origin", "(0,0,0)"))
    spacing_cm = nrrd_spacing_cm(directions)
    origin_cm = origin_mm .* 0.1

    raw_path = joinpath(dirname(nhdr_path), fields["data file"])
    bytes = read(raw_path)
    expected = prod(dims)
    length(bytes) == expected ||
        error("Raw payload size mismatch: expected $expected bytes from sizes=$dims, got $(length(bytes))")

    mask = BitArray(reshape(bytes .== label_value, dims))
    return mask, origin_cm, spacing_cm, raw_path, fields
end

function voxel_center(origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64}, i::Int, j::Int, k::Int)
    return origin_cm + SVector(
        (i - 0.5) * spacing_cm[1],
        (j - 0.5) * spacing_cm[2],
        (k - 0.5) * spacing_cm[3],
    )
end

function mask_stats(mask::BitArray{3}, origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    dims = size(mask)
    count_fg = 0
    lo_idx = SVector(typemax(Int), typemax(Int), typemax(Int))
    hi_idx = SVector(typemin(Int), typemin(Int), typemin(Int))
    sum_p = SVector(0.0, 0.0, 0.0)

    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        mask[i, j, k] || continue
        count_fg += 1
        lo_idx = min.(lo_idx, SVector(i, j, k))
        hi_idx = max.(hi_idx, SVector(i, j, k))
        sum_p += voxel_center(origin_cm, spacing_cm, i, j, k)
    end

    count_fg > 0 || error("Mask has no foreground voxels")
    center = sum_p / count_fg
    return (
        foreground=count_fg,
        lo_idx=lo_idx,
        hi_idx=hi_idx,
        center_cm=center,
        lo_cm=voxel_center(origin_cm, spacing_cm, lo_idx...),
        hi_cm=voxel_center(origin_cm, spacing_cm, hi_idx...),
    )
end

function is_boundary(mask::BitArray{3}, i::Int, j::Int, k::Int)
    dims = size(mask)
    return i == 1 || i == dims[1] || j == 1 || j == dims[2] || k == 1 || k == dims[3] ||
        !mask[i - 1, j, k] || !mask[i + 1, j, k] ||
        !mask[i, j - 1, k] || !mask[i, j + 1, k] ||
        !mask[i, j, k - 1] || !mask[i, j, k + 1]
end

function boundary_sample_matrices(mask::BitArray{3}, origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64},
                                  center_cm::SVector{3, Float64}; sample_every::Int=8, max_points::Int=120_000)
    points = Float64[]
    normals = Float64[]
    dims = size(mask)
    seen = 0

    for k in 2:(dims[3] - 1), j in 2:(dims[2] - 1), i in 2:(dims[1] - 1)
        mask[i, j, k] || continue
        is_boundary(mask, i, j, k) || continue
        seen += 1
        seen % sample_every == 0 || continue
        p = voxel_center(origin_cm, spacing_cm, i, j, k)
        n = p - center_cm
        nrm = norm(n)
        n = nrm > 0 ? n / nrm : SVector(0.0, 0.0, 1.0)
        push!(points, p[1], p[2], p[3])
        push!(normals, n[1], n[2], n[3])
        div(length(points), 3) >= max_points && break
    end

    nrows = div(length(points), 3)
    nrows > 0 || error("No boundary points were sampled")
    return copy(reshape(points, 3, nrows)'), copy(reshape(normals, 3, nrows)')
end

function nearest_mask_seed(mask::BitArray{3}, origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64},
                           center_cm::SVector{3, Float64})
    dims = size(mask)
    best = center_cm
    best_d2 = Inf
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        mask[i, j, k] || continue
        p = voxel_center(origin_cm, spacing_cm, i, j, k)
        d2 = sum(abs2, p - center_cm)
        if d2 < best_d2
            best_d2 = d2
            best = p
        end
    end
    return best
end

function build_domain_from_mask(mask::BitArray{3}, origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    stats = mask_stats(mask, origin_cm, spacing_cm)
    surface_points, surface_normals = boundary_sample_matrices(mask, origin_cm, spacing_cm, stats.center_cm)
    lo = (minimum(surface_points[:, 1]) - 1e-6, minimum(surface_points[:, 2]) - 1e-6, minimum(surface_points[:, 3]) - 1e-6)
    hi = (maximum(surface_points[:, 1]) + 1e-6, maximum(surface_points[:, 2]) + 1e-6, maximum(surface_points[:, 3]) + 1e-6)
    surface_grid = VascularTreeSim._build_point_grid(surface_points, lo, hi)
    domain = VoxelShellDomain(mask, origin_cm, spacing_cm, stats.center_cm,
        surface_points, surface_normals, Matrix{Float64}[], Matrix{Float64}[],
        surface_grid, PointCloudGrid[])
    return domain, stats
end

function write_points_csv(path::AbstractString, points::Matrix{Float64})
    open(path, "w") do io
        println(io, "x_cm,y_cm,z_cm")
        for i in axes(points, 1)
            @printf(io, "%.5f,%.5f,%.5f\n", points[i, 1], points[i, 2], points[i, 3])
        end
    end
    return path
end

function write_summary(path::AbstractString, nhdr_path::AbstractString, raw_path::AbstractString, fields,
                       domain::VoxelShellDomain, mask_info, coverage_points::Matrix{Float64},
                       graph_points::Matrix{Float64}, tree::GrowthTree, stats)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    open(path, "w") do io
        println(io, "# Raw + NRRD Thigh Smoke Test")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- NHDR: $(nhdr_path)")
        println(io, "- RAW: $(raw_path)")
        println(io, "- Label used: $(LABEL_VALUE)")
        println(io, "- NRRD type: $(get(fields, "type", "unknown"))")
        println(io, "- NRRD sizes: $(get(fields, "sizes", "unknown"))")
        println(io, "- Spacing: $(round.(Tuple(domain.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Foreground voxels: $(mask_info.foreground)")
        println(io, "- Bounding box extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        println(io, "- Coverage points: $(size(coverage_points, 1))")
        println(io, "- Graph points: $(size(graph_points, 1))")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added branches: $(stats.added)")
        println(io, "- Total segments: $(length(tree.segment_start))")
        println(io, "- Terminal vertices: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum diameter: $(round(maximum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Coverage p50: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "This is a coarse local smoke test, not a production capillary-resolution run.")
    end
    return path
end

function main()
    Random.seed!(42)
    nhdr_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NHDR
    max_branches = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 120
    mkpath(OUTPUT_DIR)

    println("=" ^ 64)
    println("Raw + NRRD thigh vessel smoke test")
    println("NHDR: $(nhdr_path)")
    println("Output: $(OUTPUT_DIR)")
    println("=" ^ 64)
    flush(stdout)

    mask, origin_cm, spacing_cm, raw_path, fields = load_binary_nrrd_mask(nhdr_path)
    println("[nrrd] dims=$(size(mask)) spacing_cm=$(spacing_cm) foreground=$(count(mask))")
    flush(stdout)

    domain, mask_info = build_domain_from_mask(mask, origin_cm, spacing_cm)
    coverage_points = coverage_target_points_blockwise(domain; block_size=COVERAGE_BLOCK_SIZE)
    graph_points = copy(coverage_points)
    viewer_points = coverage_target_points_blockwise(domain; block_size=VIEWER_BLOCK_SIZE)
    seed = nearest_mask_seed(mask, origin_cm, spacing_cm, mask_info.center_cm)
    println("[domain] bbox_cm=$(mask_info.lo_cm) -> $(mask_info.hi_cm) center=$(mask_info.center_cm)")
    println("[growth] seed=$(seed) coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) max_branches=$(max_branches)")
    flush(stdout)

    tree = growth_tree_from_seed("ThighRaw", seed; terminal_diameter_cm=TERMINAL_DIAMETER_CM)
    trees = Dict("ThighRaw" => tree)
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

    st = growth_stats["ThighRaw"]
    csv_path = joinpath(OUTPUT_DIR, "thigh_raw_nrrd_segments.csv")
    domain_csv = joinpath(OUTPUT_DIR, "thigh_raw_nrrd_domain_points.csv")
    viewer_path = joinpath(OUTPUT_DIR, "thigh_raw_nrrd_viewer.html")
    summary_path = joinpath(OUTPUT_DIR, "summary.md")

    write_growth_csv(csv_path, "ThighRaw", tree)
    write_points_csv(domain_csv, viewer_points)
    growth_viewer_html(viewer_path, domain, trees, growth_stats, Dict("ThighRaw" => "#2a9d8f"); domain_stride=VIEWER_BLOCK_SIZE)
    write_summary(summary_path, nhdr_path, raw_path, fields, domain, mask_info, coverage_points, graph_points, tree, st)

    println()
    println("Result")
    println("  foreground voxels   = $(mask_info.foreground)")
    println("  added branches      = $(st.added)")
    println("  total segments      = $(length(tree.segment_start))")
    println("  p95 coverage        = $(round(st.p95 * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(csv_path)")
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    legacy_raw_entrypoint_error("examples/thigh_raw_nrrd_smoke.jl")
end
