@testset "Flow exports" begin
    tree = growth_tree_from_seed("FlowUnit", SVector(0.0, 0.0, 0.0); terminal_diameter_cm=8e-4)
    @test VascularTreeSim._add_branch_path!(tree, tree.root_vertex,
        [SVector(0.05, 0.0, 0.0), SVector(0.10, 0.0, 0.0)];
        max_segment_length_cm=0.05)
    @test !VascularTreeSim._add_branch_path!(tree, tree.root_vertex,
        [SVector(1.0, 0.0, 0.0)];
        max_branch_length_cm=0.25,
        max_segment_length_cm=0.05)

    outdir = mktempdir()
    hemo = joinpath(outdir, "hemodynamic.csv")
    orders = joinpath(outdir, "diameter_order.csv")
    topology = joinpath(outdir, "topology.csv")
    terminal_paths = joinpath(outdir, "terminal_paths.csv")

    write_hemodynamic_tree_csv(hemo, "FlowUnit", tree; min_explicit_diameter_um=4.0)
    write_diameter_order_audit_csv(orders, "FlowUnit", tree)
    write_flow_topology_audit_csv(topology, "FlowUnit", tree)
    write_terminal_path_audit_csv(terminal_paths, "FlowUnit", tree)

    @test isfile(hemo)
    @test isfile(orders)
    @test isfile(topology)
    @test isfile(terminal_paths)
    @test occursin("virtual_inlet", read(hemo, String))
    order_text = read(orders, String)
    @test occursin("strahler_order", order_text)
    @test occursin("median_diameter_um", order_text)
    @test occursin("max_branchpoint_generation", order_text)
    topology_text = read(topology, String)
    @test occursin("branchpoint_generation", topology_text)
    terminal_text = read(terminal_paths, String)
    @test occursin("path_segment_count", terminal_text)
    @test occursin("degree2_segment_count", terminal_text)
    terminal_rows = split(chomp(terminal_text), '\n')
    @test length(terminal_rows) == 2
    terminal_cells = split(terminal_rows[2], ',')
    @test parse(Int, terminal_cells[7]) == 2   # raw polyline generation
    @test parse(Int, terminal_cells[8]) == 1   # no new branchpoint generation
    @test parse(Int, terminal_cells[14]) == 2  # path_segment_count
end

@testset "Hydraulic growth guards" begin
    spacing = SVector(0.1, 0.1, 0.1)
    mask = trues(30, 3, 3)
    origin = SVector(0.0, 0.0, 0.0)
    dummy_points = reshape([0.0, 0.0, 0.0], 1, 3)
    dummy_normals = reshape([1.0, 0.0, 0.0], 1, 3)
    dummy_grid = VascularTreeSim._build_point_grid(dummy_points, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    domain = VoxelShellDomain(
        mask, origin, spacing, SVector(1.5, 0.15, 0.15),
        dummy_points, dummy_normals,
        Matrix{Float64}[], Matrix{Float64}[],
        dummy_grid, VascularTreeSim.PointCloudGrid[])

    graph_points = coverage_target_points(domain; stride=1)
    coverage_points = reshape([2.95, 0.15, 0.15], 1, 3)

    free_tree = growth_tree_from_seed("HydraulicFree", SVector(0.05, 0.15, 0.15);
        terminal_diameter_cm=8e-4)
    _, _, free_stats = grow_trees_mcp!(Dict("HydraulicFree" => free_tree), domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=1,
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=1,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false)
    @test free_stats["__global__"].added == 1
    @test free_stats["__global__"].coverage_points == 1
    @test free_stats["HydraulicFree"].initial_territory_points == 1
    @test free_stats["HydraulicFree"].territory_points == 1
    @test free_stats["HydraulicFree"].territory_fraction == 1.0
    @test free_stats["HydraulicFree"].branch_cap == 1
    @test free_stats["HydraulicFree"].branch_cap_fraction == 1.0
    @test free_stats["HydraulicFree"].territory_prior_fraction == 1.0

    guarded_tree = growth_tree_from_seed("HydraulicGuarded", SVector(0.05, 0.15, 0.15);
        terminal_diameter_cm=8e-4)
    outdir = mktempdir()
    progress_csv = joinpath(outdir, "progress.csv")
    _, _, guarded_stats = grow_trees_mcp!(Dict("HydraulicGuarded" => guarded_tree), domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=1,
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=1,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false,
        hydraulic_reference_length_cm=1.0,
        max_new_branch_resistance_rel=0.5,
        progress_csv_path=progress_csv)
    @test guarded_stats["__global__"].added == 0
    progress_text = read(progress_csv, String)
    @test occursin("hydraulic_rejected", progress_text)
    @test occursin(",1,", progress_text)
end

@testset "Hydraulic ranking changes branch choice" begin
    spacing = SVector(0.1, 0.1, 0.1)
    mask = trues(30, 15, 3)
    origin = SVector(0.0, 0.0, 0.0)
    dummy_points = reshape([0.0, 0.0, 0.0], 1, 3)
    dummy_normals = reshape([1.0, 0.0, 0.0], 1, 3)
    dummy_grid = VascularTreeSim._build_point_grid(dummy_points, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    domain = VoxelShellDomain(
        mask, origin, spacing, SVector(1.5, 0.75, 0.15),
        dummy_points, dummy_normals,
        Matrix{Float64}[], Matrix{Float64}[],
        dummy_grid, VascularTreeSim.PointCloudGrid[])
    graph_points = coverage_target_points(domain; stride=1)
    coverage_points = [
        0.05 1.05 0.15
        2.05 1.15 0.15
    ]

    function trunk_tree(name)
        tree = growth_tree_from_seed(name, SVector(0.05, 0.05, 0.15);
            terminal_diameter_cm=8e-4)
        @test VascularTreeSim._add_branch_path!(tree, tree.root_vertex,
            [SVector(2.05, 0.05, 0.15)];
            max_segment_length_cm=0.1)
        return tree
    end

    function side_leaf_x(tree)
        xs = [tree.vertices[v][1] for v in eachindex(tree.vertices)
            if isempty(tree.children[v]) && tree.vertices[v][2] > 0.5]
        @test length(xs) == 1
        return only(xs)
    end

    far_first = trunk_tree("FarFirst")
    grow_trees_mcp!(Dict("FarFirst" => far_first), domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=1,
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=1,
        frontier_candidate_factor=2,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false)

    flow_ranked = trunk_tree("FlowRanked")
    grow_trees_mcp!(Dict("FlowRanked" => flow_ranked), domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=1,
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=1,
        frontier_candidate_factor=2,
        hydraulic_cost_weight=1.0,
        hydraulic_rank_candidate_factor=2,
        hydraulic_existing_path_weight=10.0,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false)

    @test side_leaf_x(far_first) > 1.0
    @test side_leaf_x(flow_ranked) < 1.0
end
