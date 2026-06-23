"""
    right_leg_xcat_50um_gpu.jl

Large right-leg GPU run using XCAT NRB vessel geometry for the fixed arteries,
active arterial seeds, and fixed vein exports. The XCAT raw label volume is
still used for the voxelized tissue/soft-target masks.

Usage:
    julia --project=. examples/right_leg_xcat_50um_gpu.jl [terminal_um] [output_dir] [raw_path] [organ_ids] [xcat_log] [target_branches_or_auto] [frontier_batch] [graph_block_size] [min_frontier_separation_cm] [max_segment_length_cm] [graph_neighbors] [max_path_nodes] [graph_jitter_cm] [snap_terminal_to_target] [max_terminal_snap_cm] [coverage_multiplier] [use_indexed_anchor] [use_astar_routing] [frontier_candidate_factor] [fixed_vein_exports] [max_anchor_gap_cm] [growth_artery_seeds] [growth_artery_min_length_cm] [fixed_artery_min_length_cm] [fixed_vein_min_length_cm]
"""

include(joinpath(@__DIR__, "right_leg_xcat_trial.jl"))

using CUDA
using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays

const RIGHT_LEG_50UM_DEFAULT_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_leg_corrected_50um_full")
const RIGHT_LEG_50UM_DEFAULT_NRB_PATH = joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "both_legs_1.nrb")
const RIGHT_LEG_50UM_TREE_NAME = "XCATRightLegCorrected50um"
const RIGHT_LEG_50UM_NRB_COORDINATE_SCALE = 0.1
const RIGHT_LEG_50UM_NRB_DOMAIN_VOXEL_SPACING_CM = 0.1
const RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES = (48, 48)
const RIGHT_LEG_50UM_EXPLICIT_TERMINAL_UM = 200.0
const RIGHT_LEG_50UM_SUBDIVISION_MAX_LD_RATIO = 25.0
const RIGHT_LEG_50UM_SUBDIVISION_CLIP_BELOW_UM = 0.0
const RIGHT_LEG_50UM_FRONTIER_BATCH = 4096
const RIGHT_LEG_50UM_GRAPH_BLOCK_SIZE = 1
const RIGHT_LEG_50UM_MIN_FRONTIER_SEPARATION_CM = 0.01
const RIGHT_LEG_50UM_MAX_SEGMENT_LENGTH_CM = 0.05
const RIGHT_LEG_50UM_GRAPH_NEIGHBORS = 24
const RIGHT_LEG_50UM_MAX_PATH_NODES = 48
const RIGHT_LEG_50UM_GRAPH_JITTER_CM = 0.005
const RIGHT_LEG_50UM_SNAP_TERMINAL_TO_TARGET = true
const RIGHT_LEG_50UM_MAX_TERMINAL_SNAP_CM = 0.15
const RIGHT_LEG_50UM_MAX_BRANCH_LENGTH_CM = Inf
const RIGHT_LEG_50UM_MAX_ROUTE_TORTUOSITY = 8.0
const RIGHT_LEG_50UM_MAX_ANCHOR_GAP_CM = MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM
const RIGHT_LEG_50UM_COVERAGE_MULTIPLIER = 1.10
const RIGHT_LEG_50UM_USE_INDEXED_ANCHOR = true
const RIGHT_LEG_50UM_USE_ASTAR_ROUTING = true
const RIGHT_LEG_50UM_FRONTIER_CANDIDATE_FACTOR = 32
const RIGHT_LEG_50UM_MAIN_VESSEL_OVERLAYS = 6
const RIGHT_LEG_50UM_GROWTH_ARTERY_SEEDS = 0
const RIGHT_LEG_50UM_GROWTH_ARTERY_MIN_LENGTH_CM = 8.0
const RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM = 3.0
const RIGHT_LEG_50UM_FIXED_VEIN_MIN_LENGTH_CM = 8.0
const RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS = 0
const RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS = 1
const RIGHT_LEG_50UM_MAX_DEFAULT_COVERAGE_POINTS = 50_000_000
const RIGHT_LEG_50UM_FLOW_EXPLICIT_MIN_DIAMETER_UM = 50.0
const RIGHT_LEG_50UM_TARGET_TISSUE_MODE = "soft"
const RIGHT_LEG_50UM_TARGET_DEMAND_MODE = "uniform"
const RIGHT_LEG_50UM_TARGET_DEMAND_WEIGHTS = "muscle=1.0,skin=0.2,fat=0.05,envelope=0.1,other_soft=0.1"
const RIGHT_LEG_50UM_HYDRAULIC_COST_WEIGHT = 0.25
const RIGHT_LEG_50UM_HYDRAULIC_RANK_CANDIDATE_FACTOR = 2
const RIGHT_LEG_50UM_HYDRAULIC_REFERENCE_LENGTH_CM = 1.0
const RIGHT_LEG_50UM_HYDRAULIC_EXISTING_PATH_WEIGHT = 0.5
const RIGHT_LEG_50UM_MIN_HYDRAULIC_SCORE_CM = 0.0
const RIGHT_LEG_50UM_MAX_NEW_BRANCH_RESISTANCE_REL = Inf
const RIGHT_LEG_50UM_MAX_TERMINAL_PATH_RESISTANCE_REL = Inf
const RIGHT_LEG_50UM_BLOOD_VISCOSITY_POISE = 0.035
const RIGHT_LEG_50UM_VESSEL_SOURCE = "nrb"

function parse_bool_arg(value::AbstractString)
    text = lowercase(strip(String(value)))
    text in ("true", "t", "yes", "y", "1") && return true
    text in ("false", "f", "no", "n", "0") && return false
    error("Expected a boolean value, got `$value`")
end

function parse_pair_arg(value::AbstractString)
    parts = split(strip(String(value)), ',')
    length(parts) == 2 || error("Expected pair `a,b`, got `$value`")
    return (parse(Int, strip(parts[1])), parse(Int, strip(parts[2])))
end

function enforce_graph_resolution_guard!(graph_block_size::Int, spacing_cm::SVector{3, Float64},
                                         terminal_um::Float64)
    max_quality_block = parse(Int, get(ENV, "VTS_MAX_GRAPH_BLOCK_SIZE", "1"))
    allow_coarse_graph = parse_bool_arg(get(ENV, "VTS_ALLOW_COARSE_GRAPH", "false"))
    route_sample_mm = graph_block_size * minimum(spacing_cm) * 10.0
    max_quality_mm = max_quality_block * minimum(spacing_cm) * 10.0
    if graph_block_size > max_quality_block && !allow_coarse_graph
        error("graph_block_size=$(graph_block_size) samples the route graph about every " *
              "$(round(route_sample_mm; digits=3)) mm, which exceeds the configured quality limit " *
              "VTS_MAX_GRAPH_BLOCK_SIZE=$(max_quality_block) (~$(round(max_quality_mm; digits=3)) mm). " *
              "For final low-diameter runs such as $(round(terminal_um; digits=3)) um, keep graph_block_size <= " *
              "$(max_quality_block). Set VTS_ALLOW_COARSE_GRAPH=true only for an explicitly labeled coarse smoke/preview run.")
    end
    if graph_block_size > 1
        @warn "Using coarse route graph" graph_block_size route_sample_mm terminal_um allow_coarse_graph
    end
    return nothing
end

function terminal_tag_um(terminal_um::Float64)
    if isapprox(terminal_um, round(terminal_um); atol=1e-6)
        return "$(round(Int, terminal_um))um"
    end
    return replace(@sprintf("%.3f", terminal_um), "." => "p") * "um"
end

function target_tissue_mode_from_env()
    mode = lowercase(strip(get(ENV, "VTS_TARGET_TISSUE_MODE", RIGHT_LEG_50UM_TARGET_TISSUE_MODE)))
    mode in ("soft", "muscle") || error("VTS_TARGET_TISSUE_MODE must be `soft` or `muscle`, got `$mode`")
    return mode
end

function target_demand_mode_from_env()
    mode = lowercase(strip(get(ENV, "VTS_TARGET_DEMAND_MODE", RIGHT_LEG_50UM_TARGET_DEMAND_MODE)))
    mode in ("uniform", "weighted") || error("VTS_TARGET_DEMAND_MODE must be `uniform` or `weighted`, got `$mode`")
    return mode
end

function parse_target_demand_weights(spec::AbstractString)
    weights = Dict{String, Float64}(
        "muscle" => 1.0,
        "skin" => 0.2,
        "fat" => 0.05,
        "envelope" => 0.1,
        "other_soft" => 0.1,
    )
    for item in split(spec, ',')
        text = strip(item)
        isempty(text) && continue
        parts = split(text, '=')
        length(parts) == 2 || error("Bad VTS_TARGET_DEMAND_WEIGHTS item `$text`; expected class=value")
        cls = lowercase(strip(parts[1]))
        haskey(weights, cls) || error("Unknown target demand class `$cls`")
        value = parse(Float64, strip(parts[2]))
        value >= 0.0 || error("Target demand weight for `$cls` must be nonnegative")
        weights[cls] = value
    end
    sum(values(weights)) > 0.0 || error("At least one target demand weight must be positive")
    return weights
end

function target_tissue_class(name::AbstractString)
    lname = lowercase(String(name))
    (startswith(lname, "musc") || lname == "rfoot_musc") && return "muscle"
    (occursin("skin", lname) || occursin("subcut", lname)) && return "skin"
    (occursin("fat", lname) || occursin("adip", lname)) && return "fat"
    (lname == "leg_right" || startswith(lname, "foot_right")) && return "envelope"
    return "other_soft"
end

function target_label_classes(name_to_labels)
    priority = Dict("other_soft" => 1, "envelope" => 2, "fat" => 3, "skin" => 4, "muscle" => 5)
    classes = Dict{UInt16, String}()
    for (name, labs) in name_to_labels
        cls = target_tissue_class(name)
        for lab in labs
            old = get(classes, lab, "")
            if isempty(old) || priority[cls] > priority[old]
                classes[lab] = cls
            end
        end
    end
    return classes
end

function branch_caps_from_weights(names::Vector{String}, weights::Vector{Float64}, total_cap::Int)
    caps = Dict(name => 0 for name in names)
    total_cap <= 0 && return caps
    total_weight = sum(weights)
    shares = total_weight > 0.0 ? weights ./ total_weight .* total_cap : fill(total_cap / length(names), length(names))
    floors = floor.(Int, shares)
    for (name, cap) in zip(names, floors)
        caps[name] = cap
    end
    remaining = total_cap - sum(values(caps))
    order = sortperm(collect(eachindex(names)); by=i -> shares[i] - floors[i], rev=true)
    for idx in Iterators.take(Iterators.cycle(order), remaining)
        caps[names[idx]] += 1
    end
    return caps
end

function make_right_leg_growth_trees(paths::Vector{XCATSeedPath}; terminal_diameter_cm::Float64)
    trees = Dict{String, GrowthTree}()
    names = String[]
    for (idx, path) in enumerate(paths)
        name = "$(RIGHT_LEG_50UM_TREE_NAME)_seed$(idx)"
        tree = make_single_raw_seed_tree(path;
            terminal_diameter_cm=terminal_diameter_cm,
            root_diameter_cm=proximal_root_diameter_cm(path))
        tree.name = name
        trees[name] = tree
        push!(names, name)
    end
    return trees, names
end

function random_points_in_mask(domain::VoxelShellDomain, n::Int; rng_seed::Int=42)
    mask = domain.mask
    dims = size(mask)
    foreground = Vector{UInt32}(undef, count(mask))
    idx = 0
    lin = UInt32(0)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        lin += UInt32(1)
        mask[i, j, k] || continue
        idx += 1
        foreground[idx] = lin
    end
    idx == length(foreground) || resize!(foreground, idx)

    rng = MersenneTwister(rng_seed)
    pts = Matrix{Float64}(undef, n, 3)
    nx, ny, _ = dims
    @inbounds for row in 1:n
        idx0 = Int(foreground[rand(rng, eachindex(foreground))]) - 1
        i = (idx0 % nx) + 1
        j = (div(idx0, nx) % ny) + 1
        k = div(idx0, nx * ny) + 1
        pts[row, 1] = domain.origin_cm[1] + (i - rand(rng)) * domain.spacing_cm[1]
        pts[row, 2] = domain.origin_cm[2] + (j - rand(rng)) * domain.spacing_cm[2]
        pts[row, 3] = domain.origin_cm[3] + (k - rand(rng)) * domain.spacing_cm[3]
    end
    return pts
end

function random_points_in_mask_by_demand(domain::VoxelShellDomain, raw_labels, full_dims,
                                         crop_lo::Tuple, label_classes::Dict{UInt16, String},
                                         n::Int; rng_seed::Int=42,
                                         demand_mode::AbstractString="uniform",
                                         class_weights::Dict{String, Float64}=parse_target_demand_weights(RIGHT_LEG_50UM_TARGET_DEMAND_WEIGHTS))
    mask = domain.mask
    dims = size(mask)
    nx, ny, _ = full_dims
    class_order = ["muscle", "skin", "fat", "envelope", "other_soft"]
    buckets = Dict(cls => UInt32[] for cls in class_order)

    lin = UInt32(0)
    for k in 1:dims[3], j in 1:dims[2], i in 1:dims[1]
        lin += UInt32(1)
        mask[i, j, k] || continue
        x = crop_lo[1] + i - 1
        y = crop_lo[2] + j - 1
        z = crop_lo[3] + k - 1
        raw_idx0 = (x - 1) + nx * ((y - 1) + ny * (z - 1))
        lab = raw_labels[raw_idx0 + 1]
        cls = get(label_classes, lab, "other_soft")
        haskey(buckets, cls) || (cls = "other_soft")
        push!(buckets[cls], lin)
    end

    class_counts = Dict(cls => length(buckets[cls]) for cls in class_order)
    weighted_mode = lowercase(String(demand_mode)) == "weighted"
    masses = Float64[]
    for cls in class_order
        weight = weighted_mode ? get(class_weights, cls, 0.0) : 1.0
        push!(masses, class_counts[cls] * weight)
    end
    total_mass = sum(masses)
    total_mass > 0.0 || error("No target voxels have positive sampling demand")
    cumulative = cumsum(masses)

    rng = MersenneTwister(rng_seed)
    pts = Matrix{Float64}(undef, n, 3)
    sampled_counts = Dict(cls => 0 for cls in class_order)
    @inbounds for row in 1:n
        r = rand(rng) * total_mass
        class_idx = searchsortedfirst(cumulative, r)
        class_idx = clamp(class_idx, 1, length(class_order))
        cls = class_order[class_idx]
        bucket = buckets[cls]
        while isempty(bucket)
            class_idx = rand(rng, eachindex(class_order))
            cls = class_order[class_idx]
            bucket = buckets[cls]
        end
        idx0 = Int(bucket[rand(rng, eachindex(bucket))]) - 1
        i = (idx0 % dims[1]) + 1
        j = (div(idx0, dims[1]) % dims[2]) + 1
        k = div(idx0, dims[1] * dims[2]) + 1
        pts[row, 1] = domain.origin_cm[1] + (i - rand(rng)) * domain.spacing_cm[1]
        pts[row, 2] = domain.origin_cm[2] + (j - rand(rng)) * domain.spacing_cm[2]
        pts[row, 3] = domain.origin_cm[3] + (k - rand(rng)) * domain.spacing_cm[3]
        sampled_counts[cls] += 1
    end

    audit = [
        (
            class=cls,
            voxel_count=class_counts[cls],
            demand_weight=weighted_mode ? get(class_weights, cls, 0.0) : 1.0,
            demand_mass=masses[i],
            expected_fraction=masses[i] / total_mass,
            sampled_count=sampled_counts[cls],
            sampled_fraction=n == 0 ? 0.0 : sampled_counts[cls] / n,
        )
        for (i, cls) in enumerate(class_order)
    ]
    return pts, audit
end

function write_target_demand_audit_csv(path::AbstractString, audit)
    open(path, "w") do io
        println(io, "class,voxel_count,demand_weight,demand_mass,expected_fraction,sampled_count,sampled_fraction")
        for row in audit
            println(io, join((
                row.class,
                row.voxel_count,
                row.demand_weight,
                row.demand_mass,
                row.expected_fraction,
                row.sampled_count,
                row.sampled_fraction,
            ), ","))
        end
    end
    return path
end

function nrb_target_audit(domain::VoxelShellDomain, sampled_count::Int; class_name::AbstractString="nrb_target")
    voxels = count(domain.mask)
    return [(
        class=String(class_name),
        voxel_count=voxels,
        demand_weight=1.0,
        demand_mass=Float64(voxels),
        expected_fraction=1.0,
        sampled_count=sampled_count,
        sampled_fraction=sampled_count == 0 ? 0.0 : 1.0,
    )]
end

_safe_ratio(numerator::Real, denominator::Real) =
    denominator == 0 || !isfinite(denominator) ? NaN : numerator / denominator

function _stat_value(stats, key::Symbol, default)
    return key in propertynames(stats) ? getproperty(stats, key) : default
end

function write_seed_territory_audit_csv(path::AbstractString,
                                        growth_tree_names::Vector{String},
                                        growth_artery_paths,
                                        growth_stats::Dict,
                                        branch_caps::Dict{String, Int},
                                        capacity_weights::Dict{String, Float64},
                                        territory_distance_weights::Dict{String, Float64})
    total_capacity = sum(get(capacity_weights, name, 0.0) for name in growth_tree_names)
    total_branch_cap = sum(get(branch_caps, name, 0) for name in growth_tree_names)
    total_territory_prior = sum(get(territory_distance_weights, name, 0.0)^3 for name in growth_tree_names)

    open(path, "w") do io
        println(io, "seed_name,surface,path_points,path_length_mm,proximal_diameter_mm,capacity_weight,capacity_fraction,territory_distance_weight_cm,territory_prior_fraction,branch_cap,branch_cap_fraction,initial_territory_points,initial_territory_fraction,final_territory_points,final_territory_fraction,final_to_capacity_fraction_ratio,final_to_territory_prior_ratio,added,terminals,p50_mm,p95_mm,max_mm,saturated,stall_rounds")
        for (name, artery_path) in zip(growth_tree_names, growth_artery_paths)
            stats = growth_stats[name]
            capacity_weight = get(capacity_weights, name, 0.0)
            capacity_fraction = _safe_ratio(capacity_weight, total_capacity)
            territory_weight = get(territory_distance_weights, name, 0.0)
            territory_prior_fraction = _safe_ratio(territory_weight^3, total_territory_prior)
            branch_cap = get(branch_caps, name, 0)
            branch_cap_fraction = _safe_ratio(branch_cap, total_branch_cap)
            final_fraction = _stat_value(stats, :territory_fraction, NaN)
            println(io, join((
                name,
                artery_path.surface,
                length(artery_path.points),
                10.0 * path_length_cm(artery_path),
                10.0 * proximal_root_diameter_cm(artery_path),
                capacity_weight,
                capacity_fraction,
                territory_weight,
                territory_prior_fraction,
                branch_cap,
                branch_cap_fraction,
                _stat_value(stats, :initial_territory_points, 0),
                _stat_value(stats, :initial_territory_fraction, NaN),
                _stat_value(stats, :territory_points, 0),
                final_fraction,
                _safe_ratio(final_fraction, capacity_fraction),
                _safe_ratio(final_fraction, territory_prior_fraction),
                _stat_value(stats, :added, 0),
                _stat_value(stats, :terminals, 0),
                10.0 * _stat_value(stats, :p50, NaN),
                10.0 * _stat_value(stats, :p95, NaN),
                10.0 * _stat_value(stats, :max, NaN),
                _stat_value(stats, :saturated, false),
                _stat_value(stats, :stall_rounds, 0),
            ), ","))
        end
    end
    return path
end

function write_right_leg_full_summary(path::AbstractString, raw_path, organ_ids_path, log_path,
                                      xcat_info, keep_desc, kept_counts, crop_lo, crop_hi,
                                      domain, mask_info, route_tissue_voxels::Int,
                                      route_dilation_voxels::Int,
                                      growth_artery_paths, fixed_artery_paths, vein_paths,
                                      artery_overlay_segments::Int, vein_overlay_segments::Int,
                                      tree::GrowthTree, stats;
                                      nrb_path::AbstractString,
                                      vessel_source::AbstractString,
                                      terminal_um::Float64,
                                      target_branches::Int,
                                      target_tissue_mode::AbstractString,
                                      target_demand_mode::AbstractString,
                                      target_demand_weights::AbstractString,
                                      seed_territory_audit_csv::AbstractString,
                                      flow_explicit_min_diameter_um::Float64,
                                      crop_pad_voxels::Int,
                                      muscle_tissue_voxels::Int,
                                      muscle_label_count::Int,
                                      target_label_count::Int,
                                      growth_artery_count::Int,
                                      growth_artery_min_length_cm::Float64,
                                      fixed_artery_min_length_cm::Float64,
                                      fixed_vein_min_length_cm::Float64,
                                      fixed_vessel_radius_voxels::Int,
                                      route_repair_stats,
                                      branch_caps::Dict{String, Int},
                                      coverage_count::Int,
                                      graph_count::Int,
                                      frontier_batch::Int,
                                      graph_block_size::Int,
                                      min_frontier_separation_cm::Float64,
                                      max_segment_length_cm::Float64,
                                      graph_neighbors::Int,
                                      max_path_nodes::Int,
                                      graph_jitter_cm::Float64,
                                      snap_terminal_to_target::Bool,
                                      max_terminal_snap_cm::Float64,
                                      max_branch_length_cm::Float64,
                                      max_route_tortuosity::Float64,
                                      hydraulic_cost_weight::Float64,
                                      hydraulic_rank_candidate_factor::Int,
                                      hydraulic_reference_length_cm::Float64,
                                      hydraulic_existing_path_weight::Float64,
                                      min_hydraulic_score_cm::Float64,
                                      max_new_branch_resistance_rel::Float64,
                                      max_terminal_path_resistance_rel::Float64,
                                      blood_viscosity_poise::Float64,
                                      max_anchor_gap_cm::Float64,
                                      coverage_multiplier::Float64,
                                      use_indexed_anchor::Bool,
                                      use_astar_routing::Bool,
                                      frontier_candidate_factor::Int,
                                      started_at,
                                      finished_at)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    xcat_segments = count(tree.is_xcat)
    grown_segments = length(tree.segment_start) - xcat_segments
    growth_surfaces = Set(path.surface for path in growth_artery_paths)
    fixed_non_growth_artery_paths = [path for path in fixed_artery_paths if !(path.surface in growth_surfaces)]
    route_repair_values = collect(values(route_repair_stats))
    route_repair_count = sum((stat.repaired for stat in route_repair_values); init=0)
    route_repair_skipped = sum((stat.skipped for stat in route_repair_values); init=0)
    route_repair_max_shift_cm = isempty(route_repair_values) ? 0.0 :
        maximum(stat.max_shift_cm for stat in route_repair_values)
    open(path, "w") do io
        println(io, "# XCAT Corrected Right-Leg GPU Run")
        println(io)
        println(io, "- Started: $(started_at)")
        println(io, "- Finished: $(finished_at)")
        println(io, "- XCAT NRB anatomy and vessel geometry: $(nrb_path)")
        println(io, "- XCAT raw tissue label volume: not used for geometry in this run")
        println(io, "- XCAT log: $(log_path) (metadata only)")
        println(io, "- Artery source: XCAT NRB grouped surface centerlines: arteries_rleg")
        println(io, "- Vein source: XCAT NRB grouped surface centerlines: veins_rleg")
        println(io, "- Domain source: XCAT NRB grouped surfaces")
        println(io, "- Target dims: $(size(domain.mask))")
        println(io, "- Foreground right-leg muscle/foot-muscle voxels: $(muscle_tissue_voxels)")
        println(io, "- Muscle surface count: $(muscle_label_count)")
        println(io, "- Growth target tissue mode: $(target_tissue_mode)")
        if target_tissue_mode == "muscle"
            println(io, "- Growth target mask: right-leg skeletal/foot muscle NRB surfaces only. The route mask can still include soft tissue corridors.")
        else
            println(io, "- Growth target mask: right-leg soft-tissue NRB envelope surfaces (`leg_right` and `foot_right*`).")
        end
        println(io, "- Growth target tissue voxels: $(mask_info.foreground)")
        println(io, "- Growth target surface count: $(target_label_count)")
        println(io, "- Growth target demand mode: uniform NRB-domain sampling")
        println(io, "- Seed territory audit CSV: $(seed_territory_audit_csv)")
        println(io, "- Routing mask: right-leg soft-tissue NRB envelope with fixed NRB vessels marked in")
        println(io, "- Routing tissue voxels: $(route_tissue_voxels)")
        println(io, "- Target extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        @printf(io, "- Fixed vessel proximity filter: kept path sections within %.1f mm of the soft target mask\n",
            MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM * 10.0)
        @printf(io, "- Active artery growth seed min length: %.2f mm\n", growth_artery_min_length_cm * 10.0)
        @printf(io, "- Fixed artery structural export min length: %.2f mm\n", fixed_artery_min_length_cm * 10.0)
        @printf(io, "- Fixed vein export min length: %.2f mm\n", fixed_vein_min_length_cm * 10.0)
        println(io, "- Fixed artery/vein route-mask radius: $(fixed_vessel_radius_voxels) voxel(s)")
        println(io, "- Final terminal branch diameter: $(round(terminal_um; digits=3)) um")
        println(io, "- Flow-ready hemodynamic export explicit minimum diameter: $(round(flow_explicit_min_diameter_um; digits=3)) um")
        println(io, "- Combined geometry CSV includes fixed XCAT veins for visualization/reference; arterial flow should use the arterial/hemodynamic CSV exports.")
        println(io, "- Target added routed-growth branches: $(target_branches)")
        println(io, "- Independent artery growth seeds: $(growth_artery_count)")
        println(io, "- Growth seed paths are oriented with the higher-z endpoint as the proximal/root side.")
        println(io, "- Seed capacity uses the maximum diameter in the first 8 proximal points, not the full-path maximum.")
        println(io, "- Growth capacity is distributed by proximal diameter cubed; territory distance weighting uses proximal diameter, so tissue-volume prior scales with diameter cubed rather than diameter to the ninth.")
        println(io, "- Component reachability: enabled; tissue targets can be claimed only by seeds connected through the legal route graph.")
        if target_tissue_mode == "muscle"
            println(io, "- Coverage targets are sampled uniformly from the NRB muscle target domain.")
        else
            println(io, "- Coverage targets are sampled uniformly from the NRB soft-tissue envelope target domain.")
        end
        println(io, "- Per-seed branch caps: " *
            join(["$(name)=$(branch_caps[name])" for name in sort(collect(keys(branch_caps)))], ", "))
        println(io, "- Coverage points: $(coverage_count)")
        println(io, "- Coverage multiplier: $(round(coverage_multiplier; digits=3))")
        println(io, "- Graph points: $(graph_count)")
        if :graph_components in propertynames(stats)
            println(io, "- Route graph components: $(stats.graph_components)")
            println(io, "- Route graph largest component: $(stats.graph_largest_component_nodes) nodes ($(round(100 * stats.graph_largest_component_fraction; digits=2))%)")
            println(io, "- Route graph isolated nodes: $(stats.graph_isolated_nodes)")
            println(io, "- Route graph edges: $(stats.graph_edges)")
        end
        if :unclaimed_targets in propertynames(stats)
            println(io, "- Unclaimed growth targets after reachability filtering: $(stats.unclaimed_targets)")
        end
        println(io, "- Graph block size: $(graph_block_size)")
        println(io, "- Frontier batch: $(frontier_batch)")
        println(io, "- Minimum frontier separation: $(round(min_frontier_separation_cm * 10; digits=3)) mm")
        println(io, "- Maximum segment length: $(round(max_segment_length_cm * 10; digits=3)) mm")
        if isfinite(max_branch_length_cm)
            println(io, "- Maximum added branch route length: $(round(max_branch_length_cm * 10; digits=3)) mm")
        else
            println(io, "- Maximum added branch route length: unbounded")
        end
        if isfinite(max_route_tortuosity)
            println(io, "- Maximum added branch route tortuosity: $(round(max_route_tortuosity; digits=3))")
        else
            println(io, "- Maximum added branch route tortuosity: unbounded")
        end
        println(io, "- Hydraulic candidate cost weight: $(hydraulic_cost_weight)")
        println(io, "- Hydraulic candidate ranking factor: $(hydraulic_rank_candidate_factor)")
        println(io, "- Hydraulic reference length: $(round(hydraulic_reference_length_cm * 10; digits=3)) mm")
        println(io, "- Hydraulic existing-path weight: $(hydraulic_existing_path_weight)")
        println(io, "- Minimum hydraulic score: $(round(min_hydraulic_score_cm * 10; digits=3)) mm")
        println(io, "- Maximum new-branch resistance relative to reference: " *
            (isfinite(max_new_branch_resistance_rel) ? string(round(max_new_branch_resistance_rel; digits=3)) : "unbounded"))
        println(io, "- Maximum terminal-path resistance relative to reference: " *
            (isfinite(max_terminal_path_resistance_rel) ? string(round(max_terminal_path_resistance_rel; digits=3)) : "unbounded"))
        println(io, "- Blood viscosity for hydraulic diagnostics: $(blood_viscosity_poise) P")
        println(io, "- Graph neighbors: $(graph_neighbors)")
        println(io, "- Maximum path nodes: $(max_path_nodes)")
        println(io, "- Graph jitter: $(round(graph_jitter_cm * 10; digits=3)) mm")
        println(io, "- Snap terminal to target: $(snap_terminal_to_target)")
        println(io, "- Maximum terminal snap: $(round(max_terminal_snap_cm * 10; digits=3)) mm")
        println(io, "- Maximum external anchor entry gap: $(round(max_anchor_gap_cm * 10; digits=3)) mm")
        println(io, "- Indexed anchor lookup: $(use_indexed_anchor)")
        println(io, "- A* routing: $(use_astar_routing)")
        println(io, "- Frontier candidate factor: $(frontier_candidate_factor)")
        @printf(io, "- Post-growth route repairs: %d repaired, %d skipped, max shift %.3f mm\n",
            route_repair_count, route_repair_skipped, route_repair_max_shift_cm * 10.0)
        println(io)
        println(io, "## XCAT Artery Growth Seeds")
        println(io)
        for artery_path in growth_artery_paths
            @printf(io, "- %s (growth seed): %d points, %.2f mm length, %.2f-%.2f mm diameter, %.2f mm proximal capacity diameter, root z %.2f cm\n",
                artery_path.surface, length(artery_path.points),
                path_length_cm(artery_path) * 10.0,
                minimum(artery_path.diameters_cm) * 10.0,
                maximum(artery_path.diameters_cm) * 10.0,
                proximal_root_diameter_cm(artery_path) * 10.0,
                first(artery_path.points)[3])
        end
        println(io, "- Fixed non-growth XCAT artery segments exported after growth: $(artery_overlay_segments)")
        println(io, "- All retained fixed artery paths are admitted to the route mask before growth. Non-growth arteries are fixed structural exports, not prediction seeds.")
        println(io, "- No forced connector segments were created between raw artery paths.")
        println(io)
        println(io, "## Fixed Non-growth XCAT Arteries")
        println(io)
        for artery_path in fixed_non_growth_artery_paths
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter, %.2f mm proximal capacity diameter\n",
                artery_path.surface, length(artery_path.points),
                path_length_cm(artery_path) * 10.0,
                minimum(artery_path.diameters_cm) * 10.0,
                maximum(artery_path.diameters_cm) * 10.0,
                proximal_root_diameter_cm(artery_path) * 10.0)
        end
        println(io)
        println(io, "## Fixed XCAT Veins")
        println(io)
        for vein_path in vein_paths
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter\n",
                vein_path.surface, length(vein_path.points),
                path_length_cm(vein_path) * 10.0,
                minimum(vein_path.diameters_cm) * 10.0,
                maximum(vein_path.diameters_cm) * 10.0)
        end
        println(io, "- Fixed vein segments exported after arterial growth: $(vein_overlay_segments)")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added predicted branches: $(stats.added)")
        println(io, "- XCAT structural artery seed + fixed vein segments exported: $(xcat_segments)")
        println(io, "- Predicted/grown segments exported: $(grown_segments)")
        println(io, "- Total segments exported: $(length(tree.segment_start))")
        println(io, "- Terminal vertices exported: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum displayed diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum displayed diameter: $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
        println(io, "- Coverage p50 before fixed-vein export: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95 before fixed-vein export: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max before fixed-vein export: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "This run uses corrected raw-label XCAT vessel extraction. Retained arteries are split into active growth seeds and fixed structural exports, and every retained fixed vessel is marked into the route mask before graph construction. Fixed XCAT veins are exported for downstream use, but they do not seed arterial growth because the arterial and venous trees are physiologically separate.")
    end
    return path
end

function main_right_leg_xcat_50um_gpu()
    Random.seed!(42)
    started_at = Dates.now()
    terminal_um = length(ARGS) >= 1 ? parse(Float64, ARGS[1]) : 50.0
    output_dir = length(ARGS) >= 2 ? ARGS[2] : RIGHT_LEG_50UM_DEFAULT_OUTPUT_DIR
    raw_path = length(ARGS) >= 3 ? ARGS[3] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "both_legs_act_1.raw")
    nrb_path = get(ENV, "VTS_XCAT_NRB_PATH", isfile(RIGHT_LEG_50UM_DEFAULT_NRB_PATH) ?
        RIGHT_LEG_50UM_DEFAULT_NRB_PATH : joinpath(dirname(raw_path), "both_legs_1.nrb"))
    organ_ids_path = length(ARGS) >= 4 ? ARGS[4] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "organ_ids.txt")
    log_path = length(ARGS) >= 5 ? ARGS[5] : joinpath(RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR, "both_legs_log")
    target_arg = length(ARGS) >= 6 ? ARGS[6] : "auto"
    frontier_batch = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : RIGHT_LEG_50UM_FRONTIER_BATCH
    graph_block_size = length(ARGS) >= 8 ? parse(Int, ARGS[8]) : RIGHT_LEG_50UM_GRAPH_BLOCK_SIZE
    min_frontier_separation_cm = length(ARGS) >= 9 ? parse(Float64, ARGS[9]) : RIGHT_LEG_50UM_MIN_FRONTIER_SEPARATION_CM
    max_segment_length_cm = length(ARGS) >= 10 ? parse(Float64, ARGS[10]) : RIGHT_LEG_50UM_MAX_SEGMENT_LENGTH_CM
    graph_neighbors = length(ARGS) >= 11 ? parse(Int, ARGS[11]) : RIGHT_LEG_50UM_GRAPH_NEIGHBORS
    max_path_nodes = length(ARGS) >= 12 ? parse(Int, ARGS[12]) : RIGHT_LEG_50UM_MAX_PATH_NODES
    graph_jitter_cm = length(ARGS) >= 13 ? parse(Float64, ARGS[13]) : RIGHT_LEG_50UM_GRAPH_JITTER_CM
    snap_terminal_to_target = length(ARGS) >= 14 ? parse_bool_arg(ARGS[14]) : RIGHT_LEG_50UM_SNAP_TERMINAL_TO_TARGET
    max_terminal_snap_cm = length(ARGS) >= 15 ? parse(Float64, ARGS[15]) : RIGHT_LEG_50UM_MAX_TERMINAL_SNAP_CM
    coverage_multiplier = length(ARGS) >= 16 ? parse(Float64, ARGS[16]) : RIGHT_LEG_50UM_COVERAGE_MULTIPLIER
    use_indexed_anchor = length(ARGS) >= 17 ? parse_bool_arg(ARGS[17]) : RIGHT_LEG_50UM_USE_INDEXED_ANCHOR
    use_astar_routing = length(ARGS) >= 18 ? parse_bool_arg(ARGS[18]) : RIGHT_LEG_50UM_USE_ASTAR_ROUTING
    frontier_candidate_factor = length(ARGS) >= 19 ? parse(Int, ARGS[19]) : RIGHT_LEG_50UM_FRONTIER_CANDIDATE_FACTOR
    fixed_vein_exports = length(ARGS) >= 20 ? parse(Int, ARGS[20]) : RIGHT_LEG_50UM_MAIN_VESSEL_OVERLAYS
    max_anchor_gap_cm = length(ARGS) >= 21 ? parse(Float64, ARGS[21]) : RIGHT_LEG_50UM_MAX_ANCHOR_GAP_CM
    requested_growth_artery_seeds = length(ARGS) >= 22 ? parse(Int, ARGS[22]) : RIGHT_LEG_50UM_GROWTH_ARTERY_SEEDS
    growth_artery_min_length_cm = length(ARGS) >= 23 ? parse(Float64, ARGS[23]) : RIGHT_LEG_50UM_GROWTH_ARTERY_MIN_LENGTH_CM
    fixed_artery_min_length_cm = length(ARGS) >= 24 ? parse(Float64, ARGS[24]) : RIGHT_LEG_50UM_FIXED_ARTERY_MIN_LENGTH_CM
    fixed_vein_min_length_cm = length(ARGS) >= 25 ? parse(Float64, ARGS[25]) : RIGHT_LEG_50UM_FIXED_VEIN_MIN_LENGTH_CM
    route_dilation_voxels = parse(Int, get(ENV, "VTS_ROUTE_DILATION_VOXELS", string(RIGHT_LEG_50UM_ROUTE_DILATION_VOXELS)))
    flow_explicit_min_diameter_um = parse(Float64, get(ENV, "VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM", string(RIGHT_LEG_50UM_FLOW_EXPLICIT_MIN_DIAMETER_UM)))
    max_branch_length_cm = parse(Float64, get(ENV, "VTS_MAX_BRANCH_LENGTH_CM", string(RIGHT_LEG_50UM_MAX_BRANCH_LENGTH_CM)))
    max_route_tortuosity = parse(Float64, get(ENV, "VTS_MAX_ROUTE_TORTUOSITY", string(RIGHT_LEG_50UM_MAX_ROUTE_TORTUOSITY)))
    hydraulic_cost_weight = parse(Float64, get(ENV, "VTS_HYDRAULIC_COST_WEIGHT", string(RIGHT_LEG_50UM_HYDRAULIC_COST_WEIGHT)))
    hydraulic_rank_candidate_factor = parse(Int, get(ENV, "VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR", string(RIGHT_LEG_50UM_HYDRAULIC_RANK_CANDIDATE_FACTOR)))
    hydraulic_reference_length_cm = parse(Float64, get(ENV, "VTS_HYDRAULIC_REFERENCE_LENGTH_CM", string(RIGHT_LEG_50UM_HYDRAULIC_REFERENCE_LENGTH_CM)))
    hydraulic_existing_path_weight = parse(Float64, get(ENV, "VTS_HYDRAULIC_EXISTING_PATH_WEIGHT", string(RIGHT_LEG_50UM_HYDRAULIC_EXISTING_PATH_WEIGHT)))
    min_hydraulic_score_cm = parse(Float64, get(ENV, "VTS_MIN_HYDRAULIC_SCORE_CM", string(RIGHT_LEG_50UM_MIN_HYDRAULIC_SCORE_CM)))
    max_new_branch_resistance_rel = parse(Float64, get(ENV, "VTS_MAX_NEW_BRANCH_RESISTANCE_REL", string(RIGHT_LEG_50UM_MAX_NEW_BRANCH_RESISTANCE_REL)))
    max_terminal_path_resistance_rel = parse(Float64, get(ENV, "VTS_MAX_TERMINAL_PATH_RESISTANCE_REL", string(RIGHT_LEG_50UM_MAX_TERMINAL_PATH_RESISTANCE_REL)))
    blood_viscosity_poise = parse(Float64, get(ENV, "VTS_BLOOD_VISCOSITY_POISE", string(RIGHT_LEG_50UM_BLOOD_VISCOSITY_POISE)))
    vessel_source = lowercase(strip(get(ENV, "VTS_XCAT_VESSEL_SOURCE", RIGHT_LEG_50UM_VESSEL_SOURCE)))
    vessel_source == "nrb" || error("Right-leg full runs now require VTS_XCAT_VESSEL_SOURCE=nrb; got `$vessel_source`")
    nrb_voxel_spacing_cm = parse(Float64, get(ENV, "VTS_NRB_DOMAIN_VOXEL_SPACING_CM", string(RIGHT_LEG_50UM_NRB_DOMAIN_VOXEL_SPACING_CM)))
    nrb_domain_samples = parse_pair_arg(get(ENV, "VTS_NRB_DOMAIN_SAMPLES", "$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[1]),$(RIGHT_LEG_50UM_NRB_DOMAIN_SAMPLES[2])"))
    explicit_terminal_um = parse(Float64, get(ENV, "VTS_EXPLICIT_TERMINAL_UM", string(RIGHT_LEG_50UM_EXPLICIT_TERMINAL_UM)))
    subdivision_max_ld_ratio = parse(Float64, get(ENV, "VTS_SUBDIVISION_MAX_LD_RATIO", string(RIGHT_LEG_50UM_SUBDIVISION_MAX_LD_RATIO)))
    subdivision_clip_below_um = parse(Float64, get(ENV, "VTS_SUBDIVISION_CLIP_BELOW_UM", string(RIGHT_LEG_50UM_SUBDIVISION_CLIP_BELOW_UM)))
    target_tissue_mode = target_tissue_mode_from_env()
    target_demand_mode = target_demand_mode_from_env()
    target_demand_weights_spec = get(ENV, "VTS_TARGET_DEMAND_WEIGHTS", RIGHT_LEG_50UM_TARGET_DEMAND_WEIGHTS)
    target_demand_weights = parse_target_demand_weights(target_demand_weights_spec)
    final_terminal_cm = terminal_um / 1e4
    explicit_terminal_cm = explicit_terminal_um / 1e4
    growth_terminal_cm = final_terminal_cm < explicit_terminal_cm ? explicit_terminal_cm : final_terminal_cm
    subdivision_clip_below_cm = subdivision_clip_below_um / 1e4

    mkpath(output_dir)
    xcat_info = parse_xcat_log(log_path)
    enforce_graph_resolution_guard!(graph_block_size, SVector(nrb_voxel_spacing_cm, nrb_voxel_spacing_cm, nrb_voxel_spacing_cm), terminal_um)
    crop_pad_voxels = 0
    keep_desc = "nrb-right-leg-surfaces"
    lower_count = 0
    upper_count = 0
    crop_lo = (1, 1, 1)
    crop_hi = (0, 0, 0)

    surfaces = parse_xcat_grouped_nrb(nrb_path)
    soft_surfaces = right_leg_nrb_soft_surfaces(surfaces)
    muscle_surfaces = right_leg_nrb_muscle_surfaces(surfaces, soft_surfaces)
    target_surfaces = target_tissue_mode == "muscle" ? muscle_surfaces : soft_surfaces
    target_surface_count = length(target_surfaces)
    muscle_surface_count = length(muscle_surfaces)
    route_surface_count = length(soft_surfaces)

    domain = build_multipatch_voxel_domain_floodfill(target_surfaces;
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE,
        voxel_spacing_cm=nrb_voxel_spacing_cm,
        outer_samples=nrb_domain_samples,
        dilation_radius=1)
    mask_info = mask_stats(domain.mask, domain.origin_cm, domain.spacing_cm)
    muscle_domain = target_tissue_mode == "muscle" ? domain : build_multipatch_voxel_domain_floodfill(muscle_surfaces;
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE,
        voxel_spacing_cm=nrb_voxel_spacing_cm,
        outer_samples=nrb_domain_samples,
        dilation_radius=1)
    route_domain = target_tissue_mode == "soft" ? domain : build_multipatch_voxel_domain_floodfill(soft_surfaces;
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE,
        voxel_spacing_cm=nrb_voxel_spacing_cm,
        outer_samples=nrb_domain_samples,
        dilation_radius=1)
    route_mask_info = mask_stats(route_domain.mask, route_domain.origin_cm, route_domain.spacing_cm)

    target_mask = domain.mask
    mask = muscle_domain.mask
    route_mask = copy(route_domain.mask)
    origin_cm = domain.origin_cm
    spacing_cm = domain.spacing_cm
    route_origin_cm = route_domain.origin_cm
    route_spacing_cm = route_domain.spacing_cm
    crop_hi = size(target_mask)

    nhdr_path, mask_raw_path = write_right_leg_mask_artifacts(output_dir, mask, muscle_domain.origin_cm, muscle_domain.spacing_cm)
    target_nhdr_path, target_mask_raw_path = write_right_leg_target_mask_artifacts(output_dir, target_mask, origin_cm, spacing_cm)

    artery_paths_all = nrb_group_seed_paths_from_surfaces(surfaces, "arteries_rleg";
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE)
    artery_paths_near = clip_paths_to_mask_proximity(artery_paths_all, route_mask, route_origin_cm, route_spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    oriented_artery_paths_near = [orient_right_leg_root_path(path) for path in artery_paths_near]
    fixed_artery_paths, fixed_artery_audit = select_vessel_paths(oriented_artery_paths_near;
        max_paths=0,
        min_length_cm=fixed_artery_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_artery")
    artery_paths, growth_artery_audit = select_vessel_paths(fixed_artery_paths;
        max_paths=requested_growth_artery_seeds,
        min_length_cm=growth_artery_min_length_cm,
        score=:length_mean_diameter,
        role="growth_artery")
    isempty(artery_paths) && error("No usable NRB artery seed paths were selected")
    growth_artery_count = length(artery_paths)
    growth_artery_paths = artery_paths[1:growth_artery_count]
    growth_surfaces = Set(path.surface for path in growth_artery_paths)
    artery_overlay_paths = [path for path in fixed_artery_paths if !(path.surface in growth_surfaces)]

    vein_paths_all = nrb_group_seed_paths_from_surfaces(surfaces, "veins_rleg";
        coordinate_scale=RIGHT_LEG_50UM_NRB_COORDINATE_SCALE)
    vein_paths_near = clip_paths_to_mask_proximity(vein_paths_all, route_mask, route_origin_cm, route_spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    vein_paths, vein_audit = select_vessel_paths(vein_paths_near;
        max_paths=fixed_vein_exports,
        min_length_cm=fixed_vein_min_length_cm,
        score=:length_mean_diameter,
        role="fixed_vein")

    selected_fixed_paths = vcat(fixed_artery_paths, vein_paths)
    mark_paths_in_mask!(route_mask, selected_fixed_paths, route_origin_cm, route_spacing_cm;
        radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)
    route_nhdr_path, route_mask_raw_path = write_right_leg_route_mask_artifacts(output_dir, route_mask, route_origin_cm, route_spacing_cm)
    route_domain = VoxelShellDomain(route_mask, route_domain.origin_cm, route_domain.spacing_cm, route_domain.center_cm,
        route_domain.outer_surface_points, route_domain.outer_surface_normals,
        route_domain.cavity_surface_points, route_domain.cavity_surface_normals,
        route_domain.outer_query_grid, route_domain.cavity_query_grids)
    route_mask_info = mask_stats(route_mask, route_origin_cm, route_spacing_cm)

    trees, growth_tree_names = make_right_leg_growth_trees(growth_artery_paths; terminal_diameter_cm=growth_terminal_cm)
    root_diameters = [trees[name].root_diameter_cm for name in growth_tree_names]
    capacity_weights = Dict(name => artery_weight(path) for (name, path) in zip(growth_tree_names, growth_artery_paths))
    territory_distance_weights = Dict(name => proximal_root_diameter_cm(path) for (name, path) in zip(growth_tree_names, growth_artery_paths))
    # Subdivision is a symmetric binary bifurcation (gamma=3): each growth terminal
    # yields 2^(ceil(3*log2(growth/final)) - 1) leaves, NOT the continuous (growth/final)^3.
    # Use the actual binary leaf count so target_branches back-computes to the requested total.
    subdivision_factor = final_terminal_cm < growth_terminal_cm ? 2 ^ max(0, ceil(Int, 3 * log2(growth_terminal_cm / final_terminal_cm)) - 1) : 1
    target_branches = lowercase(target_arg) == "auto" ?
        ceil(Int, (maximum(root_diameters) / growth_terminal_cm)^3) :
        max(1, ceil(Int, parse(Int, target_arg) / subdivision_factor))
    branch_caps = branch_caps_from_weights(growth_tree_names,
        [capacity_weights[name] for name in growth_tree_names], target_branches)
    coverage_count = max(target_branches, ceil(Int, target_branches * coverage_multiplier))
    max_default_coverage_points = parse(Int, get(ENV, "VTS_MAX_COVERAGE_POINTS", string(RIGHT_LEG_50UM_MAX_DEFAULT_COVERAGE_POINTS)))
    allow_huge_coverage = parse_bool_arg(get(ENV, "VTS_ALLOW_HUGE_COVERAGE", "false"))
    if coverage_count > max_default_coverage_points && !allow_huge_coverage
        error("Requested coverage_count=$(coverage_count), which exceeds the safety limit $(max_default_coverage_points). " *
              "For very large runs, pass an explicit target branch count or set VTS_MAX_COVERAGE_POINTS / VTS_ALLOW_HUGE_COVERAGE after confirming memory and disk budget. " *
              "This guard prevents accidental 6um auto runs from allocating billions of target points.")
    end
    progress_csv_path = joinpath(output_dir, "growth_progress.csv")
    target_demand_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_target_demand_audit.csv")
    seed_territory_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_seed_territory_audit.csv")
    checkpoint_dir_arg = strip(get(ENV, "VTS_CHECKPOINT_DIR", ""))
    checkpoint_dir = isempty(checkpoint_dir_arg) ? nothing : checkpoint_dir_arg
    checkpoint_interval_seconds = parse(Float64, get(ENV, "VTS_CHECKPOINT_INTERVAL_SECONDS", "Inf"))
    checkpoint_keep = parse(Int, get(ENV, "VTS_CHECKPOINT_KEEP", "2"))

    println("=" ^ 78)
    println("XCAT corrected right-leg GPU run")
    println("Started: $(started_at)")
    println("Raw tissue labels: $(raw_path)")
    println("NRB vessel geometry: $(nrb_path)")
    println("Anatomy/domain source: XCAT NRB")
    println("Vessel source: $(vessel_source)")
    println("Output: $(output_dir)")
    println("Final terminal diameter: $(terminal_um) um")
    println("Explicit routed growth terminal diameter: $(round(growth_terminal_cm * 1e4; digits=3)) um")
    final_terminal_cm < growth_terminal_cm &&
        println("Post-growth subdivision: $(round(growth_terminal_cm * 1e4; digits=3)) um -> $(round(final_terminal_cm * 1e4; digits=3)) um")
    requested_growth_artery_seeds <= 0 &&
        println("Growth seed count set to all eligible XCAT artery paths.")
    println("Growth seeds: " * join([path.surface for path in growth_artery_paths], ", "))
    println("Seed proximal capacity diameters: " * join(["$(round(trees[name].root_diameter_cm * 10; digits=3)) mm" for name in growth_tree_names], ", "))
    println("Target added routed-growth branches: $(target_branches)")
    println("Per-seed branch caps: " * join(["$(name)=$(branch_caps[name])" for name in growth_tree_names], ", "))
    println("Coverage points: $(coverage_count)")
    println("Target demand mode: $(target_demand_mode)")
    println("Target demand weights: $(target_demand_weights_spec)")
    println("Graph block size: $(graph_block_size)")
    println("Frontier batch: $(frontier_batch)")
    println("Minimum frontier separation: $(round(min_frontier_separation_cm * 10; digits=3)) mm")
    println("Maximum segment length: $(round(max_segment_length_cm * 10; digits=3)) mm")
    println("Maximum branch route length: " * (isfinite(max_branch_length_cm) ? "$(round(max_branch_length_cm * 10; digits=3)) mm" : "unbounded"))
    println("Maximum route tortuosity: " * (isfinite(max_route_tortuosity) ? string(round(max_route_tortuosity; digits=3)) : "unbounded"))
    println("Hydraulic cost weight: $(hydraulic_cost_weight)")
    println("Hydraulic rank candidate factor: $(hydraulic_rank_candidate_factor)")
    println("Hydraulic reference length: $(round(hydraulic_reference_length_cm * 10; digits=3)) mm")
    println("Hydraulic existing-path weight: $(hydraulic_existing_path_weight)")
    println("Minimum hydraulic score: $(round(min_hydraulic_score_cm * 10; digits=3)) mm")
    println("Maximum new-branch resistance rel: " * (isfinite(max_new_branch_resistance_rel) ? string(round(max_new_branch_resistance_rel; digits=3)) : "unbounded"))
    println("Maximum terminal-path resistance rel: " * (isfinite(max_terminal_path_resistance_rel) ? string(round(max_terminal_path_resistance_rel; digits=3)) : "unbounded"))
    println("Blood viscosity: $(blood_viscosity_poise) P")
    println("Snap terminal to target: $(snap_terminal_to_target)")
    println("Maximum terminal snap: $(round(max_terminal_snap_cm * 10; digits=3)) mm")
    println("Maximum external anchor entry gap: $(round(max_anchor_gap_cm * 10; digits=3)) mm")
    println("Indexed anchor lookup: $(use_indexed_anchor)")
    println("A* routing: $(use_astar_routing)")
    println("Frontier candidate factor: $(frontier_candidate_factor)")
    if checkpoint_dir !== nothing && isfinite(checkpoint_interval_seconds) && checkpoint_interval_seconds > 0
        println("Checkpoint directory: $(checkpoint_dir)")
        println("Checkpoint interval: $(round(checkpoint_interval_seconds / 3600; digits=2)) hr")
        println("Checkpoints kept: $(checkpoint_keep)")
    else
        println("Checkpointing: disabled")
    end
    println("Growth artery seed min length: $(round(growth_artery_min_length_cm * 10; digits=3)) mm")
    println("Fixed artery export min length: $(round(fixed_artery_min_length_cm * 10; digits=3)) mm")
    println("Fixed vein export min length: $(round(fixed_vein_min_length_cm * 10; digits=3)) mm")
    println("NRB domain voxel spacing: $(round(nrb_voxel_spacing_cm * 10; digits=3)) mm")
    println("NRB domain samples per surface: $(nrb_domain_samples)")
    println("NRB right-leg muscle surfaces: $(muscle_surface_count)")
    println("NRB right-leg soft/route surfaces: $(route_surface_count)")
    println("Target dims: $(size(target_mask)), target_foreground=$(count(target_mask)), target_surfaces=$(target_surface_count)")
    println("Muscle mask foreground=$(count(mask)), muscle_surfaces=$(muscle_surface_count)")
    println("Route tissue mask foreground=$(count(route_mask)) route_surfaces=$(route_surface_count) dilation_voxels=$(route_dilation_voxels) fixed_vessel_radius_voxels=$(RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS)")
    println("[artery paths] growth=$(length(growth_artery_paths)) fixed_non_growth=$(length(artery_overlay_paths)) fixed_total=$(length(fixed_artery_paths)) from $(length(artery_paths_all)) $(vessel_source) artery paths ($(length(artery_paths_near)) near-soft-target clipped paths)")
    println("[fixed veins] selected=$(length(vein_paths)) from $(length(vein_paths_all)) $(vessel_source) vein paths ($(length(vein_paths_near)) near-soft-target clipped paths)")
    println("[coverage] sampling $(coverage_count) random target points inside the target mask")
    flush(stdout)

    coverage_points = random_points_in_mask(domain, coverage_count; rng_seed=42)
    target_demand_audit = nrb_target_audit(domain, coverage_count; class_name=target_tissue_mode == "muscle" ? "nrb_muscle_target" : "nrb_soft_target")
    write_target_demand_audit_csv(target_demand_audit_csv, target_demand_audit)
    graph_points = coverage_target_points_blockwise(route_domain; block_size=graph_block_size)
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1)) frontier_batch=$(frontier_batch)")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, route_domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.0,
        capillary_diameter_cm=growth_terminal_cm,
        max_new_branches_per_tree=branch_caps,
        graph_neighbors=graph_neighbors,
        min_frontier_separation_cm=min_frontier_separation_cm,
        max_path_nodes=max_path_nodes,
        target_p95_distance_cm=Inf,
        target_max_distance_cm=Inf,
        frontier_batch=frontier_batch,
        gamma=3.0,
        smooth_passes=8,
        spline_density=3,
        max_segment_length_cm=max_segment_length_cm,
        max_branch_length_cm=max_branch_length_cm,
        max_route_tortuosity=max_route_tortuosity,
        hydraulic_cost_weight=hydraulic_cost_weight,
        hydraulic_rank_candidate_factor=hydraulic_rank_candidate_factor,
        hydraulic_reference_length_cm=hydraulic_reference_length_cm,
        hydraulic_existing_path_weight=hydraulic_existing_path_weight,
        min_hydraulic_score_cm=min_hydraulic_score_cm,
        max_new_branch_resistance_rel=max_new_branch_resistance_rel,
        max_terminal_path_resistance_rel=max_terminal_path_resistance_rel,
        blood_viscosity_poise=blood_viscosity_poise,
        turn_penalty=0.35,
        graph_jitter_cm=graph_jitter_cm,
        snap_terminal_to_target=snap_terminal_to_target,
        max_terminal_snap_cm=max_terminal_snap_cm,
        max_anchor_gap_cm=max_anchor_gap_cm,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        component_reachability=true,
        use_gpu=true,
        tree_weights=capacity_weights,
        territory_weights=territory_distance_weights,
        precompute_target_graph_indices=true,
        progress_csv_path=progress_csv_path,
        checkpoint_dir=checkpoint_dir,
        checkpoint_interval_seconds=checkpoint_interval_seconds,
        checkpoint_keep=checkpoint_keep)

    write_seed_territory_audit_csv(seed_territory_audit_csv,
        growth_tree_names, growth_artery_paths, growth_stats,
        branch_caps, capacity_weights, territory_distance_weights)
    stats = growth_stats["__global__"]
    route_repair_stats = Dict{String, NamedTuple}()
    for name in growth_tree_names
        if final_terminal_cm < growth_terminal_cm
            subdivide_terminals!(trees[name];
                target_diameter_cm=final_terminal_cm,
                gamma=3.0,
                max_ld_ratio=subdivision_max_ld_ratio,
                clip_below_diameter_cm=subdivision_clip_below_cm,
                skip_xcat=true,
                domain=route_domain)
        end
        route_repair_stats[name] = repair_grown_segments_to_mask!(
            trees[name], route_mask, route_origin_cm, route_spacing_cm;
            max_radius_voxels=2,
            max_passes=3)
    end
    route_repair_total = sum((stat.repaired for stat in values(route_repair_stats)); init=0)
    route_repair_skipped = sum((stat.skipped for stat in values(route_repair_stats)); init=0)
    route_repair_max_shift_cm = isempty(route_repair_stats) ? 0.0 :
        maximum(stat.max_shift_cm for stat in values(route_repair_stats))
    println("[route repair] repaired=$(route_repair_total) skipped=$(route_repair_skipped) max_shift=$(round(route_repair_max_shift_cm * 10; digits=3)) mm")
    for (path, name) in zip(growth_artery_paths, growth_tree_names)
        restore_xcat_seed_diameters!(trees[name], [path])
    end
    tree = merge_growth_trees(RIGHT_LEG_50UM_TREE_NAME,
        [trees[name] for name in growth_tree_names]; terminal_diameter_cm=final_terminal_cm)
    artery_overlay_segments = append_fixed_overlays!(tree, artery_overlay_paths)
    vein_overlay_segments = append_vein_overlays!(tree, vein_paths)

    terminal_tag = terminal_tag_um(terminal_um)
    csv_path = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_segments.csv")
    arterial_csv_path = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_arterial_segments.csv")
    hemodynamic_csv_path = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_hemodynamic_segments.csv")
    topology_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_topology_audit.csv")
    terminal_path_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_terminal_path_audit.csv")
    root_territory_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_root_territory_audit.csv")
    diameter_order_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(terminal_tag)_diameter_order_audit.csv")
    xcat_fixed_csv = joinpath(output_dir, "xcat_right_leg_corrected_fixed_vessels.csv")
    source_tag = replace(vessel_source, "-" => "_")
    raw_artery_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(source_tag)_artery_centerlines.csv")
    raw_vein_csv = joinpath(output_dir, "xcat_right_leg_corrected_$(source_tag)_vein_centerlines.csv")
    growth_artery_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_growth_artery_path_audit.csv")
    fixed_artery_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_fixed_artery_path_audit.csv")
    vein_audit_csv = joinpath(output_dir, "xcat_right_leg_corrected_vein_path_audit.csv")
    summary_path = joinpath(output_dir, "summary.md")

    write_growth_csv(csv_path, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_arterial_growth_csv(arterial_csv_path, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_hemodynamic_tree_csv(hemodynamic_csv_path, RIGHT_LEG_50UM_TREE_NAME, tree;
        min_explicit_diameter_um=flow_explicit_min_diameter_um)
    write_flow_topology_audit_csv(topology_audit_csv, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_terminal_path_audit_csv(terminal_path_audit_csv, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_root_territory_audit_csv(root_territory_audit_csv, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_diameter_order_audit_csv(diameter_order_audit_csv, RIGHT_LEG_50UM_TREE_NAME, tree)
    write_xcat_seed_csv(xcat_fixed_csv, tree)
    write_xcat_paths_csv(raw_artery_csv, fixed_artery_paths)
    write_xcat_paths_csv(raw_vein_csv, vein_paths)
    write_vessel_path_audit_csv(growth_artery_audit_csv, growth_artery_audit;
        mask=route_mask, origin_cm=route_origin_cm, spacing_cm=route_spacing_cm,
        paths=fixed_artery_paths)
    write_vessel_path_audit_csv(fixed_artery_audit_csv, fixed_artery_audit;
        mask=route_mask, origin_cm=route_origin_cm, spacing_cm=route_spacing_cm,
        paths=oriented_artery_paths_near)
    write_vessel_path_audit_csv(vein_audit_csv, vein_audit;
        mask=route_mask, origin_cm=route_origin_cm, spacing_cm=route_spacing_cm,
        paths=vein_paths_near)
    write_right_leg_full_summary(summary_path, raw_path, organ_ids_path, log_path,
        xcat_info, keep_desc, (lower_count, upper_count), crop_lo, crop_hi,
        domain, mask_info, count(route_mask), route_dilation_voxels, growth_artery_paths, fixed_artery_paths, vein_paths, artery_overlay_segments,
        vein_overlay_segments, tree, stats;
        nrb_path=nrb_path,
        vessel_source=vessel_source,
        terminal_um=terminal_um,
        target_branches=target_branches,
        target_tissue_mode=target_tissue_mode,
        target_demand_mode=target_demand_mode,
        target_demand_weights=target_demand_weights_spec,
        seed_territory_audit_csv=seed_territory_audit_csv,
        flow_explicit_min_diameter_um=flow_explicit_min_diameter_um,
        crop_pad_voxels=crop_pad_voxels,
        muscle_tissue_voxels=count(mask),
        muscle_label_count=muscle_surface_count,
        target_label_count=target_surface_count,
        growth_artery_count=growth_artery_count,
        growth_artery_min_length_cm=growth_artery_min_length_cm,
        fixed_artery_min_length_cm=fixed_artery_min_length_cm,
        fixed_vein_min_length_cm=fixed_vein_min_length_cm,
        fixed_vessel_radius_voxels=RIGHT_LEG_50UM_ROUTE_VESSEL_RADIUS_VOXELS,
        route_repair_stats=route_repair_stats,
        branch_caps=branch_caps,
        coverage_count=size(coverage_points, 1),
        graph_count=size(graph_points, 1),
        frontier_batch=frontier_batch,
        graph_block_size=graph_block_size,
        min_frontier_separation_cm=min_frontier_separation_cm,
        max_segment_length_cm=max_segment_length_cm,
        graph_neighbors=graph_neighbors,
        max_path_nodes=max_path_nodes,
        graph_jitter_cm=graph_jitter_cm,
        snap_terminal_to_target=snap_terminal_to_target,
        max_terminal_snap_cm=max_terminal_snap_cm,
        max_branch_length_cm=max_branch_length_cm,
        max_route_tortuosity=max_route_tortuosity,
        hydraulic_cost_weight=hydraulic_cost_weight,
        hydraulic_rank_candidate_factor=hydraulic_rank_candidate_factor,
        hydraulic_reference_length_cm=hydraulic_reference_length_cm,
        hydraulic_existing_path_weight=hydraulic_existing_path_weight,
        min_hydraulic_score_cm=min_hydraulic_score_cm,
        max_new_branch_resistance_rel=max_new_branch_resistance_rel,
        max_terminal_path_resistance_rel=max_terminal_path_resistance_rel,
        blood_viscosity_poise=blood_viscosity_poise,
        max_anchor_gap_cm=max_anchor_gap_cm,
        coverage_multiplier=coverage_multiplier,
        use_indexed_anchor=use_indexed_anchor,
        use_astar_routing=use_astar_routing,
        frontier_candidate_factor=frontier_candidate_factor,
        started_at=started_at,
        finished_at=Dates.now())

    println()
    println("XCAT Corrected Right-Leg $(terminal_tag) Result")
    println("  added predicted branches = $(stats.added)")
    println("  xcat structural/fixed segments exported = $(count(tree.is_xcat))")
    println("  fixed non-growth artery segments = $(artery_overlay_segments)")
    println("  fixed vein segments       = $(vein_overlay_segments)")
    println("  total segments exported   = $(length(tree.segment_start))")
    println("  flow explicit min diameter = $(round(flow_explicit_min_diameter_um; digits=3)) um")
    println("  min diameter              = $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
    println("  max diameter              = $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
    println("  p95 coverage              = $(round(stats.p95 * 10; digits=2)) mm")
    :unclaimed_targets in propertynames(stats) &&
        println("  unclaimed targets         = $(stats.unclaimed_targets)")
    println()
    println("Wrote")
    println("  $(nhdr_path)")
    println("  $(mask_raw_path)")
    println("  $(target_nhdr_path)")
    println("  $(target_mask_raw_path)")
    println("  $(target_demand_audit_csv)")
    println("  $(seed_territory_audit_csv)")
    println("  $(csv_path)")
    println("  $(arterial_csv_path)")
    println("  $(hemodynamic_csv_path)")
    println("  $(topology_audit_csv)")
    println("  $(terminal_path_audit_csv)")
    println("  $(root_territory_audit_csv)")
    println("  $(diameter_order_audit_csv)")
    println("  $(xcat_fixed_csv)")
    println("  $(raw_artery_csv)")
    println("  $(raw_vein_csv)")
    println("  $(growth_artery_audit_csv)")
    println("  $(fixed_artery_audit_csv)")
    println("  $(vein_audit_csv)")
    println("  $(progress_csv_path)")
    checkpoint_dir !== nothing && println("  $(checkpoint_dir)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_right_leg_xcat_50um_gpu()
end
