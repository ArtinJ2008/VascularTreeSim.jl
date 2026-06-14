"""
    thigh_raw_nrrd_femoral_full.jl

Full thigh vessel-growth run for the detached raw+NRRD thigh mask. This reuses
the anatomy-informed femoral/profunda/circumflex/perforator seed geometry from
`thigh_raw_nrrd_femoral_smoke.jl`, but runs with denser sampling, many more
branches, CUDA acceleration, and a separate output folder.

Usage:
    julia --project=. examples/thigh_raw_nrrd_femoral_full.jl [nhdr_path] [max_branches] [coverage_block] [route_block] [terminal_um]

Defaults:
    max_branches=2500, coverage_block=4, route_block=4, terminal_um=400
"""

include(joinpath(@__DIR__, "thigh_raw_nrrd_femoral_smoke.jl"))

using CUDA

const FULL_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "thigh_raw_nrrd_femoral_full")
const FULL_VIEWER_BLOCK_SIZE = 4
const DEFAULT_FULL_TERMINAL_DIAMETER_UM = 400.0

terminal_diameter_cm(terminal_um::Real) = Float64(terminal_um) / 1e4

function terminal_label(terminal_um::Real)
    rounded = round(Float64(terminal_um); digits=3)
    if abs(rounded - round(Int, rounded)) < 1e-9
        return "$(round(Int, rounded))um"
    end
    return replace(string(rounded), "." => "p") * "um"
end

function full_output_dir(terminal_um::Real)
    if abs(Float64(terminal_um) - DEFAULT_FULL_TERMINAL_DIAMETER_UM) < 1e-9
        return FULL_OUTPUT_DIR
    end
    return joinpath(dirname(@__DIR__), "output", "thigh_raw_nrrd_femoral_full_$(terminal_label(terminal_um))")
end

function write_full_summary(path::AbstractString, nhdr_path::AbstractString, raw_path::AbstractString,
                            fields, domain::VoxelShellDomain, mask_info, trunk_points,
                            coverage_points::Matrix{Float64}, graph_points::Matrix{Float64},
                            tree::GrowthTree, stats; max_branches::Int,
                            coverage_block::Int, route_block::Int, used_gpu::Bool,
                            terminal_um::Float64)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    trunk_len_cm = sum(norm(trunk_points[i] - trunk_points[i - 1]) for i in 2:length(trunk_points))
    fixed_segments = count(label -> haskey(FIXED_BRANCH_DIAMETER_CM, label), tree.segment_label)
    fixed_counts = Dict(label => count(==(label), tree.segment_label) for label in FIXED_BRANCH_ORDER)

    open(path, "w") do io
        println(io, "# Raw + NRRD Thigh Full Femoral Simulation")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- NHDR: $(nhdr_path)")
        println(io, "- RAW: $(raw_path)")
        println(io, "- Label used: $(LABEL_VALUE)")
        println(io, "- NRRD sizes: $(get(fields, "sizes", "unknown"))")
        println(io, "- Spacing: $(round.(Tuple(domain.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Foreground voxels: $(mask_info.foreground)")
        println(io, "- Bounding box extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        println(io, "- CUDA requested: true")
        println(io, "- CUDA used: $(used_gpu)")
        println(io, "- CUDA device: $(used_gpu ? CUDA.name(CUDA.device()) : "none")")
        println(io, "- Max branches requested: $(max_branches)")
        println(io, "- Full-run terminal branch diameter: $(round(terminal_um; digits=3)) um")
        println(io, "- Coverage block size: $(coverage_block)")
        println(io, "- Route block size: $(route_block)")
        println(io, "- Approximate femoral trunk points: $(length(trunk_points))")
        println(io, "- Approximate femoral trunk length: $(round(trunk_len_cm * 10; digits=2)) mm")
        println(io, "- Approximate femoral trunk diameter: $(round(fixed_diameter_cm("femoral_trunk") * 10; digits=2)) mm")
        println(io, "- Coverage points: $(size(coverage_points, 1))")
        println(io, "- Graph points: $(size(graph_points, 1))")
        println(io)
        println(io, "## Fixed Anatomy Seeds")
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
        println(io, "The current input mask is binary, so region controls in the viewer are coarse anatomic quadrants rather than true named-muscle labels.")
        println(io, "This is the full local-resolution run for the current simulator settings; the model still uses its configured terminal branch diameter.")
    end
    return path
end

function main_full()
    Random.seed!(42)
    nhdr_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NHDR
    max_branches = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 2500
    coverage_block = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 4
    route_block = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : coverage_block
    terminal_um = length(ARGS) >= 5 ? parse(Float64, ARGS[5]) : DEFAULT_FULL_TERMINAL_DIAMETER_UM
    terminal_cm = terminal_diameter_cm(terminal_um)
    output_dir = full_output_dir(terminal_um)
    mkpath(output_dir)

    println("=" ^ 78)
    println("Raw + NRRD thigh FULL femoral simulation")
    println("NHDR: $(nhdr_path)")
    println("Output: $(output_dir)")
    println("Max branches: $(max_branches)")
    println("Terminal diameter: $(terminal_um) um")
    println("Coverage block: $(coverage_block), route block: $(route_block)")
    println("CUDA functional: $(CUDA.functional())")
    CUDA.functional() && println("CUDA device: $(CUDA.name(CUDA.device()))")
    println("=" ^ 78)
    flush(stdout)

    mask, origin_cm, spacing_cm, raw_path, fields = load_binary_nrrd_mask(nhdr_path)
    domain, mask_info = build_domain_from_mask(mask, origin_cm, spacing_cm)
    trunk_points = approximate_femoral_trunk(mask, origin_cm, spacing_cm, mask_info)
    tree = make_anatomic_seed_tree(mask, origin_cm, spacing_cm, mask_info, trunk_points)
    tree.terminal_diameter_cm = terminal_cm
    trees = Dict("ThighFemoral" => tree)

    coverage_points = coverage_target_points_blockwise(domain; block_size=coverage_block)
    graph_points = coverage_target_points_blockwise(domain; block_size=route_block)
    viewer_points = coverage_target_points_blockwise(domain; block_size=FULL_VIEWER_BLOCK_SIZE)

    println("[nrrd] dims=$(size(mask)) spacing_cm=$(spacing_cm) foreground=$(count(mask))")
    println("[seeds] fixed_segments=$(length(tree.segment_start)) trunk_points=$(length(trunk_points))")
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) max_branches=$(max_branches)")
    flush(stdout)

    used_gpu = CUDA.functional()
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
        use_gpu=used_gpu)

    preserve_fixed_branch_diameters!(tree)
    st = growth_stats["ThighFemoral"]
    csv_path = joinpath(output_dir, "thigh_raw_nrrd_femoral_full_segments.csv")
    trunk_csv = joinpath(output_dir, "approx_femoral_trunk_points.csv")
    seed_csv = joinpath(output_dir, "approx_femoral_seed_segments.csv")
    domain_csv = joinpath(output_dir, "thigh_raw_nrrd_femoral_full_domain_points.csv")
    viewer_path = joinpath(output_dir, "thigh_raw_nrrd_femoral_full_viewer.html")
    summary_path = joinpath(output_dir, "summary.md")

    write_growth_csv(csv_path, "ThighFemoral", tree)
    write_trunk_csv(trunk_csv, trunk_points)
    write_fixed_seed_csv(seed_csv, tree)
    write_points_csv(domain_csv, viewer_points)
    write_femoral_viewer(viewer_path, mask, origin_cm, spacing_cm, mask_info, viewer_points, tree)
    write_full_summary(summary_path, nhdr_path, raw_path, fields, domain, mask_info,
        trunk_points, coverage_points, graph_points, tree, st;
        max_branches=max_branches, coverage_block=coverage_block,
        route_block=route_block, used_gpu=used_gpu, terminal_um=terminal_um)

    fixed_segments = count(label -> haskey(FIXED_BRANCH_DIAMETER_CM, label), tree.segment_label)
    println()
    println("Full Simulation Result")
    println("  CUDA used            = $(used_gpu)")
    println("  fixed seed segments  = $(fixed_segments)")
    println("  added branches       = $(st.added)")
    println("  total segments       = $(length(tree.segment_start))")
    println("  p50 coverage         = $(round(st.p50 * 10; digits=2)) mm")
    println("  p95 coverage         = $(round(st.p95 * 10; digits=2)) mm")
    println("  max coverage         = $(round(st.max * 10; digits=2)) mm")
    println()
    println("Wrote")
    println("  $(csv_path)")
    println("  $(trunk_csv)")
    println("  $(seed_csv)")
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

main_full()
