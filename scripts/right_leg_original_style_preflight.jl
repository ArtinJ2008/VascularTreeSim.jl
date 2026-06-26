include(joinpath(@__DIR__, "..", "examples", "right_leg_xcat_50um_gpu.jl"))

using Printf
using StaticArrays

function _env_float(name::AbstractString, default)
    return parse(Float64, get(ENV, name, string(default)))
end

function _env_int(name::AbstractString, default)
    return parse(Int, get(ENV, name, string(default)))
end

function _bytes_available(path::AbstractString)
    out = read(`df -B1 $path`, String)
    rows = split(strip(out), '\n')
    length(rows) >= 2 || return 0
    cols = split(rows[end])
    length(cols) >= 4 || return 0
    return parse(Int, cols[4])
end

function _human_bytes(bytes::Real)
    value = Float64(bytes)
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    idx = 1
    while value >= 1024.0 && idx < length(units)
        value /= 1024.0
        idx += 1
    end
    return @sprintf("%.2f %s", value, units[idx])
end

function main()
    work_root = get(ENV, "VTS_WORK_ROOT",
        "/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453")
    input_root = get(ENV, "VTS_INPUT_ROOT", joinpath(work_root, "input", "both_legs_xcat_input"))
    nrb_path = get(ENV, "VTS_XCAT_NRB_PATH", joinpath(input_root, "both_legs_1.nrb"))
    output_root = joinpath(work_root, "output")
    terminal_um = _env_float("VTS_TERMINAL_UM", 8.0)
    explicit_terminal_um = _env_float("VTS_EXPLICIT_TERMINAL_UM", 200.0)
    distal_gamma = _env_float("VTS_DISTAL_MURRAY_GAMMA", 3.0)
    proximal_gamma = _env_float("VTS_PROXIMAL_MURRAY_GAMMA", 3.0)
    transition_um = _env_float("VTS_MURRAY_TRANSITION_UM", explicit_terminal_um)
    target_arg = get(ENV, "VTS_TARGET_BRANCHES", "auto")
    target_mode = get(ENV, "VTS_TARGET_BRANCH_COUNT_MODE", "explicit")
    export_full_arterial_csv = parse_bool_arg(get(ENV, "VTS_EXPORT_FULL_ARTERIAL_CSV", "true"))
    export_full_geometry_csv = parse_bool_arg(get(ENV, "VTS_EXPORT_FULL_GEOMETRY_CSV", "false"))
    emit_hemodynamic_csv = parse_bool_arg(get(ENV, "VTS_WRITE_HEMODYNAMIC_CSV", "false"))
    emit_terminal_bed_csv = parse_bool_arg(get(ENV, "VTS_WRITE_TERMINAL_BED_CSV", "false"))
    emit_topology_audit_csv = parse_bool_arg(get(ENV, "VTS_WRITE_TOPOLOGY_AUDIT_CSV", "false"))
    emit_terminal_path_audit_csv = parse_bool_arg(get(ENV, "VTS_WRITE_TERMINAL_PATH_AUDIT_CSV", "false"))
    emit_root_territory_audit_csv = parse_bool_arg(get(ENV, "VTS_WRITE_ROOT_TERRITORY_AUDIT_CSV", "false"))
    emit_diameter_order_audit_csv = parse_bool_arg(get(ENV, "VTS_WRITE_DIAMETER_ORDER_AUDIT_CSV", "false"))
    nrb_voxel_spacing_cm = _env_float("VTS_NRB_DOMAIN_VOXEL_SPACING_CM", RIGHT_LEG_50UM_NRB_DOMAIN_VOXEL_SPACING_CM)
    nrb_domain_samples = parse_pair_arg(get(ENV, "VTS_NRB_DOMAIN_SAMPLES",
        "$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[1]),$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[2])"))
    route_dilation_voxels = _env_int("VTS_ROUTE_DILATION_VOXELS", RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS)
    growth_artery_seeds = _env_int("VTS_GROWTH_ARTERY_SEEDS", RIGHT_LEG_50UM_GROWTH_ARTERY_SEEDS)
    growth_artery_min_length_cm = _env_float("VTS_GROWTH_ARTERY_MIN_LENGTH_CM", 3.0)
    fixed_artery_min_length_cm = _env_float("VTS_FIXED_ARTERY_MIN_LENGTH_CM", RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM)

    final_terminal_cm = terminal_um / 1e4
    explicit_terminal_cm = explicit_terminal_um / 1e4
    growth_terminal_cm = final_terminal_cm < explicit_terminal_cm ? explicit_terminal_cm : final_terminal_cm
    transition_cm = transition_um / 1e4
    subdivision_factor = subdivision_leaf_factor(growth_terminal_cm, final_terminal_cm; gamma=distal_gamma)

    println("[preflight] parsing NRB: $nrb_path")
    surfaces = parse_xcat_grouped_nrb(nrb_path)
    soft_surfaces = right_leg_nrb_soft_surfaces(surfaces)
    route_domain = build_multipatch_voxel_domain_floodfill(soft_surfaces;
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE,
        voxel_spacing_cm=nrb_voxel_spacing_cm,
        outer_samples=nrb_domain_samples,
        dilation_radius=1)
    route_mask = copy(route_domain.mask)
    if route_dilation_voxels > 0
        route_mask = dilate_mask_box(route_mask, route_dilation_voxels)
    end

    artery_paths_all = nrb_group_seed_paths_from_surfaces(surfaces, "arteries_rleg";
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE)
    artery_paths_near = clip_paths_to_mask_proximity(artery_paths_all, route_mask,
        route_domain.origin_cm, route_domain.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    oriented_paths = [orient_right_leg_root_path(path) for path in artery_paths_near]
    fixed_artery_paths, _ = select_vessel_paths(oriented_paths;
        max_paths=0,
        min_length_cm=fixed_artery_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_artery")
    growth_paths, _ = select_vessel_paths(fixed_artery_paths;
        max_paths=growth_artery_seeds,
        min_length_cm=growth_artery_min_length_cm,
        score=:length_mean_diameter,
        role="growth_artery")
    isempty(growth_paths) && error("No growth artery seeds selected")

    root_diameters = [proximal_root_diameter_cm(path) for path in growth_paths]
    names = ["seed$(i)" for i in eachindex(root_diameters)]
    target_branches, requested_target_count = resolve_growth_target_branches(
        target_arg,
        target_mode,
        subdivision_factor,
        root_diameters,
        growth_terminal_cm,
        distal_gamma,
        proximal_gamma,
        transition_cm)
    capacity_caps = root_capacity_branch_caps(
        names,
        root_diameters,
        growth_terminal_cm,
        target_branches,
        distal_gamma,
        proximal_gamma,
        transition_cm)
    total_capacity = sum(values(capacity_caps))
    projected_final_leaves = target_branches * subdivision_factor

    # Conservative geometry/output estimates, calibrated from the previous
    # original-style run: 75.1M exported segments occupied about 66GB and ran
    # in about 1.17 hours. Full binary subdivision would be about
    # 2 * subdivision_factor segments per explicit CCO terminal.
    estimated_segments = target_branches * max(1, 2 * subdivision_factor - 1)
    bytes_per_segment =
        (export_full_arterial_csv ? 253.0 : 0.0) +
        (export_full_geometry_csv ? 253.0 : 0.0) +
        (emit_hemodynamic_csv ? 372.0 : 0.0) +
        (emit_terminal_bed_csv ? 136.0 : 0.0) +
        (emit_topology_audit_csv ? 215.0 : 0.0) +
        (emit_terminal_path_audit_csv ? 20.0 : 0.0) +
        (emit_root_territory_audit_csv ? 5.0 : 0.0) +
        (emit_diameter_order_audit_csv ? 5.0 : 0.0)
    bytes_per_segment <= 0.0 && (bytes_per_segment = 10.0)
    seconds_per_million_segments = 56.0
    estimated_bytes = estimated_segments * bytes_per_segment
    estimated_hours = estimated_segments / 1.0e6 * seconds_per_million_segments / 3600.0
    available_bytes = isdir(output_root) ? _bytes_available(output_root) : _bytes_available(work_root)

    println("work_root=$work_root")
    println("output_root=$output_root")
    println("available=$( _human_bytes(available_bytes) )")
    println("target_arg=$target_arg")
    println("target_mode=$target_mode")
    println("requested_target_count=$(requested_target_count === nothing ? "auto" : string(requested_target_count))")
    println("growth_seed_count=$(length(growth_paths))")
    println("growth_terminal_um=$(round(growth_terminal_cm * 1e4; digits=3))")
    println("final_terminal_um=$(round(final_terminal_cm * 1e4; digits=3))")
    println("distal_gamma=$distal_gamma")
    println("proximal_gamma=$proximal_gamma")
    println("transition_um=$transition_um")
    println("subdivision_factor=$subdivision_factor")
    println("target_growth_branches=$target_branches")
    println("total_root_capacity=$total_capacity")
    println("projected_final_leaves=$projected_final_leaves")
    println("estimated_explicit_segments=$estimated_segments")
    println("export_full_arterial_csv=$export_full_arterial_csv")
    println("export_full_geometry_csv=$export_full_geometry_csv")
    println("write_hemodynamic_csv=$emit_hemodynamic_csv")
    println("write_terminal_bed_csv=$emit_terminal_bed_csv")
    println("write_topology_audit_csv=$emit_topology_audit_csv")
    println("write_terminal_path_audit_csv=$emit_terminal_path_audit_csv")
    println("write_root_territory_audit_csv=$emit_root_territory_audit_csv")
    println("write_diameter_order_audit_csv=$emit_diameter_order_audit_csv")
    println("estimated_output_size=$( _human_bytes(estimated_bytes) )")
    println("estimated_runtime_hours=$(round(estimated_hours; digits=2))")
    println("largest_roots_mm=" * join(round.(root_diameters[1:min(end, 10)] .* 10; digits=3), ","))
    if estimated_bytes > 0.8 * available_bytes
        println("disk_status=NOT_ENOUGH_LOCAL_SPACE")
    else
        println("disk_status=space_ok")
    end
end

main()
