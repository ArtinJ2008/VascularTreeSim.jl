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
    beds = joinpath(outdir, "terminal_beds.csv")
    orders = joinpath(outdir, "diameter_order.csv")
    topology = joinpath(outdir, "topology.csv")
    terminal_paths = joinpath(outdir, "terminal_paths.csv")

    write_hemodynamic_tree_csv(hemo, "FlowUnit", tree; min_explicit_diameter_um=4.0)
    write_terminal_bed_audit_csv(beds, "FlowUnit", tree;
        bed_terminal_diameter_um=8.0,
        min_explicit_diameter_um=4.0,
        terminal_bed_length_cm=0.05)
    write_diameter_order_audit_csv(orders, "FlowUnit", tree)
    write_flow_topology_audit_csv(topology, "FlowUnit", tree)
    write_terminal_path_audit_csv(terminal_paths, "FlowUnit", tree)

    @test isfile(hemo)
    @test isfile(beds)
    @test isfile(orders)
    @test isfile(topology)
    @test isfile(terminal_paths)
    hemo_text = read(hemo, String)
    @test occursin("virtual_inlet", hemo_text)
    @test occursin("gravity_head_mmHg", hemo_text)
    bed_text = read(beds, String)
    @test occursin("bed_terminal_diameter_um", bed_text)
    @test occursin("lumped_parallel_resistance_rel", bed_text)
    @test length(split(chomp(bed_text), '\n')) == 2
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

    sampled_paths = joinpath(outdir, "terminal_paths_sampled.csv")
    write_terminal_path_audit_csv(sampled_paths, "FlowUnit", tree;
        max_rows=1,
        include_path_segments=false)
    sampled_cells = split(split(chomp(read(sampled_paths, String)), '\n')[2], ',')
    @test sampled_cells[end] == ""

    header_only_paths = joinpath(outdir, "terminal_paths_header_only.csv")
    write_terminal_path_audit_csv(header_only_paths, "FlowUnit", tree; max_rows=0)
    @test length(split(chomp(read(header_only_paths, String)), '\n')) == 1

    filtered_topology = joinpath(outdir, "topology_filtered.csv")
    write_flow_topology_audit_csv(filtered_topology, "FlowUnit", tree;
        min_diameter_um=1.0e9)
    @test length(split(chomp(read(filtered_topology, String)), '\n')) == 1
end

@testset "Domain graph preserves elongated connected masks" begin
    mask = trues(3, 3, 900)
    origin = SVector(0.0, 0.0, 0.0)
    spacing = SVector(0.1, 0.1, 0.1)
    surface_points = reshape([0.0, 0.0, 0.0], 1, 3)
    surface_normals = reshape([1.0, 0.0, 0.0], 1, 3)
    surface_grid = VascularTreeSim._build_point_grid(surface_points,
        (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    domain = VoxelShellDomain(mask, origin, spacing, SVector(0.15, 0.15, 45.0),
        surface_points, surface_normals, Matrix{Float64}[], Matrix{Float64}[],
        surface_grid, PointCloudGrid[])

    graph_points = coverage_target_points_blockwise(domain; block_size=1)
    graph = build_domain_graph(graph_points, domain; k=8)
    _, components = VascularTreeSim.graph_connected_components(graph)
    @test components == 1
end

@testset "Piecewise Murray diameter law" begin
    terminal_cm = 0.02       # 200 um routed-growth terminal
    transition_cm = 0.02     # conduit law starts above routed terminal
    femoral_cm = 0.78        # 7.8 mm diameter-scale example

    @test murray_terminal_capacity(transition_cm, terminal_cm;
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm) ≈ 1.0
    @test murray_terminal_capacity(femoral_cm, terminal_cm;
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm) ≈ (femoral_cm / transition_cm)^2
    cap = murray_terminal_capacity(femoral_cm, terminal_cm;
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm)
    @test murray_diameter_from_terminals(cap, terminal_cm;
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm) ≈ femoral_cm

    tree = growth_tree_from_seed("Piecewise", SVector(0.0, 0.0, 0.0);
        terminal_diameter_cm=terminal_cm)
    @test VascularTreeSim._add_branch_path!(tree, tree.root_vertex,
        [SVector(0.10, 0.0, 0.0)];
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm,
        max_segment_length_cm=0.1)
    trunk_tip = tree.segment_end[1]
    for y in (0.10, -0.10, 0.20)
        @test VascularTreeSim._add_branch_path!(tree, trunk_tip,
            [SVector(0.10, y, 0.0)];
            gamma=3.0,
            proximal_gamma=2.0,
            transition_diameter_cm=transition_cm,
            max_segment_length_cm=0.1)
    end
    @test length(VascularTreeSim._branch_terminals(tree)) == 4
    @test tree.subtree_terminal_count[tree.segment_end[1]] == 4
    @test tree.segment_diameter_cm[1] ≈ terminal_cm * sqrt(4)

    tree.segment_diameter_cm[1] = 1.0
    VascularTreeSim._recompute_all_murray!(tree;
        target_diameter_cm=terminal_cm,
        gamma=3.0,
        proximal_gamma=2.0,
        transition_diameter_cm=transition_cm)
    @test tree.segment_diameter_cm[1] ≈ terminal_cm * sqrt(4)
end

@testset "Subdivision can skip fixed XCAT-only terminals" begin
    function one_segment_xcat_tree()
        tree = growth_tree_from_seed("XcatOnly", SVector(0.0, 0.0, 0.0);
            terminal_diameter_cm=0.02)
        @test VascularTreeSim._add_branch_path!(tree, tree.root_vertex,
            [SVector(0.1, 0.0, 0.0)];
            max_segment_length_cm=0.1)
        tree.is_xcat[1] = true
        tree.root_diameter_cm = tree.segment_diameter_cm[1]
        return tree
    end

    skipped = one_segment_xcat_tree()
    n_vertices = length(skipped.vertices)
    subdivide_terminals!(skipped;
        target_diameter_cm=0.01,
        skip_xcat=true)
    @test length(skipped.vertices) == n_vertices
    @test length(VascularTreeSim._branch_terminals(skipped)) == 1

    defaulted = one_segment_xcat_tree()
    subdivide_terminals!(defaulted; target_diameter_cm=0.01)
    @test length(defaulted.vertices) > n_vertices
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

    path_guarded_tree = growth_tree_from_seed("PathGuarded", SVector(0.05, 0.15, 0.15);
        terminal_diameter_cm=8e-4)
    _, _, path_guarded_stats = grow_trees_mcp!(Dict("PathGuarded" => path_guarded_tree), domain;
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
        max_terminal_path_length_cm=0.5)
    @test path_guarded_stats["__global__"].added == 0
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

@testset "Initial territory bootstrap prevents seed starvation" begin
    spacing = SVector(0.1, 0.1, 0.1)
    mask = trues(35, 3, 3)
    origin = SVector(0.0, 0.0, 0.0)
    dummy_points = reshape([0.0, 0.0, 0.0], 1, 3)
    dummy_normals = reshape([1.0, 0.0, 0.0], 1, 3)
    dummy_grid = VascularTreeSim._build_point_grid(dummy_points, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    domain = VoxelShellDomain(
        mask, origin, spacing, SVector(1.75, 0.15, 0.15),
        dummy_points, dummy_normals,
        Matrix{Float64}[], Matrix{Float64}[],
        dummy_grid, VascularTreeSim.PointCloudGrid[])
    graph_points = coverage_target_points(domain; stride=1)
    coverage_points = [
        2.85 0.15 0.15
        2.95 0.15 0.15
        3.05 0.15 0.15
        3.15 0.15 0.15
    ]

    function two_seed_trees()
        return Dict(
            "AStarved" => growth_tree_from_seed("AStarved", SVector(2.90, 0.15, 0.15);
                terminal_diameter_cm=8e-4),
            "ZDominant" => growth_tree_from_seed("ZDominant", SVector(0.05, 0.15, 0.15);
                terminal_diameter_cm=8e-4),
        )
    end

    base_trees = two_seed_trees()
    _, _, base_stats = grow_trees_mcp!(base_trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=Dict("AStarved" => 3, "ZDominant" => 1),
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=2,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false,
        territory_weights=Dict("AStarved" => 0.01, "ZDominant" => 1.0))
    @test base_stats["AStarved"].initial_territory_points == 0
    @test base_stats["AStarved"].added == 0

    boot_trees = two_seed_trees()
    _, _, boot_stats = grow_trees_mcp!(boot_trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=Dict("AStarved" => 3, "ZDominant" => 1),
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=2,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false,
        territory_weights=Dict("AStarved" => 0.01, "ZDominant" => 1.0),
        min_initial_territory_fraction=1.0,
        max_initial_territory_points_per_tree=4)
    @test boot_stats["AStarved"].initial_territory_points > 0
    @test boot_stats["AStarved"].territory_bootstrap_points > 0
    @test boot_stats["AStarved"].territory_points >= boot_stats["AStarved"].territory_floor_points
    @test boot_stats["AStarved"].territory_floor_maintenance_points > 0
    @test boot_stats["AStarved"].added > 0
end

@testset "Adaptive branch caps follow reachable territory" begin
    spacing = SVector(0.1, 0.1, 0.1)
    mask = trues(35, 3, 3)
    origin = SVector(0.0, 0.0, 0.0)
    dummy_points = reshape([0.0, 0.0, 0.0], 1, 3)
    dummy_normals = reshape([1.0, 0.0, 0.0], 1, 3)
    dummy_grid = VascularTreeSim._build_point_grid(dummy_points, (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0))
    domain = VoxelShellDomain(
        mask, origin, spacing, SVector(1.75, 0.15, 0.15),
        dummy_points, dummy_normals,
        Matrix{Float64}[], Matrix{Float64}[],
        dummy_grid, VascularTreeSim.PointCloudGrid[])
    graph_points = coverage_target_points(domain; stride=1)
    coverage_points = [
        2.75 0.15 0.15
        2.85 0.15 0.15
        2.95 0.15 0.15
        3.05 0.15 0.15
    ]

    trees = Dict(
        "Local" => growth_tree_from_seed("Local", SVector(2.80, 0.15, 0.15);
            terminal_diameter_cm=8e-4),
        "Remote" => growth_tree_from_seed("Remote", SVector(0.05, 0.15, 0.15);
            terminal_diameter_cm=8e-4),
    )
    _, _, stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=8e-4,
        max_new_branches_per_tree=Dict("Local" => 1, "Remote" => 3),
        adaptive_branch_caps=true,
        graph_neighbors=6,
        min_frontier_separation_cm=0.0,
        max_path_nodes=100,
        frontier_batch=2,
        smooth_passes=0,
        spline_density=1,
        max_segment_length_cm=0.1,
        graph_jitter_cm=0.0,
        use_gpu=false,
        component_reachability=false)

    @test stats["__global__"].adaptive_branch_caps
    @test stats["Local"].initial_branch_cap == 1
    @test stats["Local"].branch_cap > stats["Local"].initial_branch_cap
    @test stats["Local"].added > 1
    @test stats["Remote"].branch_cap == 0
end
