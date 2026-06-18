"""
    right_leg_xcat_trial.jl

Build a right-leg muscle mask from a both-legs XCAT color-code RAW volume,
seed growth from XCAT right-leg artery centerlines, overlay the main XCAT
right-leg veins after growth, and write the rich femoral viewer.

Usage:
    julia --project=. examples/right_leg_xcat_trial.jl [max_branches] [terminal_um] [output_dir] [input_dir] [geometry_dir] [block_size] [main_veins]
"""

include(joinpath(@__DIR__, "thigh_xcat_femoral_100um.jl"))

using Dates
using LinearAlgebra
using Printf
using Random
using StaticArrays
using Statistics

const RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR = raw"D:\UCI\Vessel Prediction\New Outputs\both_legs_xcat_input"
const RIGHT_LEG_TRIAL_DEFAULT_GEOMETRY_DIR = raw"D:\UCI\Vessel Prediction\New Outputs\both_legs_xcat_geometry"
const RIGHT_LEG_TRIAL_DEFAULT_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_right_leg_trial_100um")
const RIGHT_LEG_TRIAL_TREE_NAME = "XCATRightLegTrial"
const MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM = 1.25

function right_leg_muscle_labels(name_to_labels)
    labels = UInt16[]
    for (name, labs) in name_to_labels
        if startswith(name, "musc") || name == "rfoot_musc"
            append!(labels, labs)
        end
    end
    unique!(labels)
    !isempty(labels) || error("No muscle labels found in organ ID file")
    return labels
end

function right_leg_soft_target_labels(name_to_labels)
    labels = UInt16[]
    for (name, labs) in name_to_labels
        lname = lowercase(String(name))
        is_right_leg_envelope = lname == "leg_right" || startswith(lname, "foot_right")
        is_muscle = startswith(lname, "musc") || lname == "rfoot_musc"
        is_fat_or_skin = occursin("fat", lname) || occursin("adip", lname) ||
            occursin("skin", lname) || occursin("subcut", lname)
        if is_right_leg_envelope || is_muscle || is_fat_or_skin
            append!(labels, labs)
        end
    end
    unique!(labels)
    !isempty(labels) || error("No right-leg soft-tissue target labels found in organ ID file")
    return labels
end

function right_leg_route_tissue_labels(name_to_labels)
    labels = UInt16[]
    for (name, labs) in name_to_labels
        lname = lowercase(String(name))
        is_right_leg_envelope = lname == "leg_right" || startswith(lname, "foot_right")
        is_muscle = startswith(lname, "musc") || lname == "rfoot_musc"
        is_xcat_vessel = lname == "arteries_rleg" || lname == "veins_rleg"
        is_fat_or_skin = occursin("fat", lname) || occursin("adip", lname) ||
            occursin("skin", lname) || occursin("subcut", lname)
        if is_right_leg_envelope || is_muscle || is_xcat_vessel || is_fat_or_skin
            append!(labels, labs)
        end
    end
    unique!(labels)
    !isempty(labels) || error("No right-leg soft-tissue route labels found in organ ID file")
    return labels
end

function vessel_aware_crop_pad_voxels(spacing_cm;
                                      max_vessel_distance_cm::Float64=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM,
                                      vessel_radius_voxels::Int=1,
                                      margin_voxels::Int=2)
    max_vessel_distance_cm < 0 && error("max_vessel_distance_cm must be >= 0")
    vessel_radius_voxels < 0 && error("vessel_radius_voxels must be >= 0")
    margin_voxels < 0 && error("margin_voxels must be >= 0")
    spacing_min = minimum(spacing_cm)
    spacing_min > 0 || error("spacing must be positive")
    return max(2, ceil(Int, max_vessel_distance_cm / spacing_min) + vessel_radius_voxels + margin_voxels)
end

function build_right_leg_label_mask(raw_labels, dims, target_labels::Vector{UInt16}, keep_x;
                                    description::AbstractString="Right-leg label")
    target_set = Set(target_labels)
    keep = Set(keep_x)
    nx, ny, nz = dims
    mask = falses(nx, ny, nz)
    kept_counts = Dict{UInt16, Int}()

    for idx0 in 0:(length(raw_labels) - 1)
        lab = raw_labels[idx0 + 1]
        lab in target_set || continue
        x = (idx0 % nx) + 1
        x in keep || continue
        y = (div(idx0, nx) % ny) + 1
        z = div(idx0, nx * ny) + 1
        mask[x, y, z] = true
        kept_counts[lab] = get(kept_counts, lab, 0) + 1
    end

    count(mask) > 0 || error("$(description) mask is empty")
    return mask, kept_counts
end

function build_right_leg_muscle_mask(raw_labels, dims, target_labels::Vector{UInt16}, keep_x)
    return build_right_leg_label_mask(raw_labels, dims, target_labels, keep_x;
        description="Right-leg muscle")
end

function crop_mask_to_bounds(full_mask::BitArray{3}, crop_lo::Tuple, crop_hi::Tuple)
    return full_mask[crop_lo[1]:crop_hi[1], crop_lo[2]:crop_hi[2], crop_lo[3]:crop_hi[3]]
end

function write_right_leg_binary_mask_artifacts(output_dir::AbstractString, mask::BitArray{3},
                                               origin_cm::SVector{3, Float64},
                                               spacing_cm::SVector{3, Float64};
                                               basename::AbstractString)
    raw_filename = "$(basename).raw"
    nhdr_filename = "$(basename).nhdr"
    raw_path = joinpath(output_dir, raw_filename)
    nhdr_path = joinpath(output_dir, nhdr_filename)
    open(raw_path, "w") do io
        for k in axes(mask, 3), j in axes(mask, 2), i in axes(mask, 1)
            write(io, UInt8(mask[i, j, k] ? 1 : 0))
        end
    end
    open(nhdr_path, "w") do io
        println(io, "NRRD0005")
        println(io, "type: uchar")
        println(io, "dimension: 3")
        println(io, "space: left-posterior-superior")
        println(io, "sizes: $(size(mask, 1)) $(size(mask, 2)) $(size(mask, 3))")
        println(io, "space directions: ($(spacing_cm[1] * 10),0,0) (0,$(spacing_cm[2] * 10),0) (0,0,$(spacing_cm[3] * 10))")
        println(io, "space origin: ($(origin_cm[1] * 10),$(origin_cm[2] * 10),$(origin_cm[3] * 10))")
        println(io, "encoding: raw")
        println(io, "endian: little")
        println(io, "data file: $(raw_filename)")
    end
    return nhdr_path, raw_path
end

function write_right_leg_mask_artifacts(output_dir::AbstractString, mask::BitArray{3},
                                        origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    return write_right_leg_binary_mask_artifacts(output_dir, mask, origin_cm, spacing_cm;
        basename="right_leg_muscle_mask")
end

function write_right_leg_target_mask_artifacts(output_dir::AbstractString, mask::BitArray{3},
                                               origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    return write_right_leg_binary_mask_artifacts(output_dir, mask, origin_cm, spacing_cm;
        basename="right_leg_target_tissue_mask")
end

function build_right_leg_route_tissue_mask(raw_labels, dims, route_labels::Vector{UInt16},
                                           keep_x, crop_lo::Tuple, crop_hi::Tuple)
    route_set = Set(route_labels)
    keep = Set(keep_x)
    nx, ny, _ = dims
    sx = crop_hi[1] - crop_lo[1] + 1
    sy = crop_hi[2] - crop_lo[2] + 1
    sz = crop_hi[3] - crop_lo[3] + 1
    mask = falses(sx, sy, sz)
    for z in crop_lo[3]:crop_hi[3], y in crop_lo[2]:crop_hi[2], x in crop_lo[1]:crop_hi[1]
        x in keep || continue
        idx0 = (x - 1) + nx * ((y - 1) + ny * (z - 1))
        raw_labels[idx0 + 1] in route_set || continue
        mask[x - crop_lo[1] + 1, y - crop_lo[2] + 1, z - crop_lo[3] + 1] = true
    end
    count(mask) > 0 || error("Right-leg route tissue mask is empty")
    return mask
end

function _dilate_axis(mask::BitArray{3}, radius::Int, axis::Int)
    radius <= 0 && return copy(mask)
    nx, ny, nz = size(mask)
    out = falses(nx, ny, nz)
    if axis == 1
        for k in 1:nz, j in 1:ny
            count = 0
            hi0 = min(nx, radius + 1)
            for i in 1:hi0
                count += mask[i, j, k] ? 1 : 0
            end
            for i in 1:nx
                out[i, j, k] = count > 0
                old_i = i - radius
                new_i = i + radius + 1
                old_i >= 1 && (count -= mask[old_i, j, k] ? 1 : 0)
                new_i <= nx && (count += mask[new_i, j, k] ? 1 : 0)
            end
        end
    elseif axis == 2
        for k in 1:nz, i in 1:nx
            count = 0
            hi0 = min(ny, radius + 1)
            for j in 1:hi0
                count += mask[i, j, k] ? 1 : 0
            end
            for j in 1:ny
                out[i, j, k] = count > 0
                old_j = j - radius
                new_j = j + radius + 1
                old_j >= 1 && (count -= mask[i, old_j, k] ? 1 : 0)
                new_j <= ny && (count += mask[i, new_j, k] ? 1 : 0)
            end
        end
    elseif axis == 3
        for j in 1:ny, i in 1:nx
            count = 0
            hi0 = min(nz, radius + 1)
            for k in 1:hi0
                count += mask[i, j, k] ? 1 : 0
            end
            for k in 1:nz
                out[i, j, k] = count > 0
                old_k = k - radius
                new_k = k + radius + 1
                old_k >= 1 && (count -= mask[i, j, old_k] ? 1 : 0)
                new_k <= nz && (count += mask[i, j, new_k] ? 1 : 0)
            end
        end
    else
        error("axis must be 1, 2, or 3")
    end
    return out
end

function dilate_mask_box(mask::BitArray{3}, radius::Int)
    radius <= 0 && return copy(mask)
    return _dilate_axis(_dilate_axis(_dilate_axis(mask, radius, 1), radius, 2), radius, 3)
end

function fill_axial_slice_spans(mask::BitArray{3})
    nx, ny, nz = size(mask)
    out = copy(mask)
    for k in 1:nz
        for j in 1:ny
            lo = 0
            hi = 0
            for i in 1:nx
                if mask[i, j, k]
                    lo == 0 && (lo = i)
                    hi = i
                end
            end
            if lo > 0
                for i in lo:hi
                    out[i, j, k] = true
                end
            end
        end
        for i in 1:nx
            lo = 0
            hi = 0
            for j in 1:ny
                if mask[i, j, k]
                    lo == 0 && (lo = j)
                    hi = j
                end
            end
            if lo > 0
                for j in lo:hi
                    out[i, j, k] = true
                end
            end
        end
    end
    return out
end

function write_right_leg_route_mask_artifacts(output_dir::AbstractString, mask::BitArray{3},
                                             origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    return write_right_leg_binary_mask_artifacts(output_dir, mask, origin_cm, spacing_cm;
        basename="right_leg_route_tissue_mask")
end

function select_existing_path(paths::Vararg{AbstractString})
    for path in paths
        isfile(path) && return path
    end
    error("None of these paths exist: $(join(paths, ", "))")
end

function _csv_column_map(header_line::AbstractString)
    header = split(header_line, ',')
    return Dict(_csv_unquote(name) => idx for (idx, name) in enumerate(header))
end

function load_xcat_paths_from_raw_segments(path::AbstractString; label_prefix::String="")
    isfile(path) || error("Missing XCAT raw-space segment CSV: $path")
    lines = readlines(path)
    length(lines) >= 2 || return XCATSeedPath[]
    col = _csv_column_map(lines[1])
    required = [
        "surface", "segment_index",
        "raw_x1_cm", "raw_y1_cm", "raw_z1_cm",
        "raw_x2_cm", "raw_y2_cm", "raw_z2_cm",
        "mean_diameter_cm",
    ]
    for name in required
        haskey(col, name) || error("Raw-space segment CSV missing `$name`: $path")
    end

    rows = Dict{String, Vector{NamedTuple}}()
    order = String[]
    for line in lines[2:end]
        isempty(strip(line)) && continue
        parts = split(line, ',')
        surface = _csv_unquote(parts[col["surface"]])
        if !haskey(rows, surface)
            rows[surface] = NamedTuple[]
            push!(order, surface)
        end
        push!(rows[surface], (
            segment_index=parse(Int, parts[col["segment_index"]]),
            p1=SVector(
                parse(Float64, parts[col["raw_x1_cm"]]),
                parse(Float64, parts[col["raw_y1_cm"]]),
                parse(Float64, parts[col["raw_z1_cm"]]),
            ),
            p2=SVector(
                parse(Float64, parts[col["raw_x2_cm"]]),
                parse(Float64, parts[col["raw_y2_cm"]]),
                parse(Float64, parts[col["raw_z2_cm"]]),
            ),
            diameter_cm=parse(Float64, parts[col["mean_diameter_cm"]]),
        ))
    end

    paths = XCATSeedPath[]
    for surface in order
        sorted_rows = sort(rows[surface], by=r -> r.segment_index)
        isempty(sorted_rows) && continue
        points = SVector{3, Float64}[first(sorted_rows).p1]
        diameters = Float64[first(sorted_rows).diameter_cm]
        indices = Int[1]
        for row in sorted_rows
            if norm(row.p2 - last(points)) >= 1e-4
                push!(points, row.p2)
                push!(diameters, row.diameter_cm)
                push!(indices, row.segment_index + 1)
            else
                diameters[end] = max(diameters[end], row.diameter_cm)
            end
        end
        length(points) >= 2 || continue
        label = isempty(label_prefix) ? surface : string(label_prefix, surface)
        push!(paths, XCATSeedPath(label, indices, points, diameters))
    end
    return paths
end

function path_length_cm(path::XCATSeedPath)
    return sum(norm(path.points[i] - path.points[i - 1]) for i in 2:length(path.points))
end

function orient_right_leg_root_path(path::XCATSeedPath)
    length(path.points) >= 2 || return path
    first_z = first(path.points)[3]
    last_z = last(path.points)[3]
    if last_z > first_z
        return XCATSeedPath(path.surface, reverse(path.point_indices), reverse(path.points), reverse(path.diameters_cm))
    end
    return path
end

function proximal_root_diameter_cm(path::XCATSeedPath; window_points::Int=8)
    n = min(max(window_points, 1), length(path.diameters_cm))
    return maximum(path.diameters_cm[1:n])
end

function select_main_paths(paths::Vector{XCATSeedPath}; max_paths::Int=6, min_length_cm::Float64=5.0)
    ranked = sort(
        [p for p in paths if path_length_cm(p) >= min_length_cm],
        by=p -> -(path_length_cm(p) * mean(p.diameters_cm)),
    )
    max_paths <= 0 && return ranked
    return ranked[1:min(max_paths, length(ranked))]
end

function select_longest_paths(paths::Vector{XCATSeedPath}; max_paths::Int=6, min_length_cm::Float64=5.0)
    ranked = sort(
        [p for p in paths if path_length_cm(p) >= min_length_cm],
        by=p -> -path_length_cm(p),
    )
    return ranked[1:min(max_paths, length(ranked))]
end

function vessel_path_score(path::XCATSeedPath, score::Symbol)
    if score == :length
        return path_length_cm(path)
    elseif score == :length_mean_diameter
        return path_length_cm(path) * mean(path.diameters_cm)
    elseif score == :proximal_diameter
        return proximal_root_diameter_cm(path)
    elseif score == :capacity
        return proximal_root_diameter_cm(path)^3
    else
        error("Unknown vessel path score `$score`")
    end
end

function select_vessel_paths(paths::Vector{XCATSeedPath};
                             max_paths::Int=0,
                             min_length_cm::Float64=0.0,
                             min_proximal_diameter_cm::Float64=0.0,
                             score::Symbol=:length_mean_diameter,
                             role::AbstractString="vessel")
    rows = NamedTuple[]
    eligible = XCATSeedPath[]
    eligible_score = Float64[]
    for path in paths
        len_cm = path_length_cm(path)
        prox_d = proximal_root_diameter_cm(path)
        mean_d = mean(path.diameters_cm)
        max_d = maximum(path.diameters_cm)
        reason = ""
        if len_cm < min_length_cm
            reason = "below_min_length"
        elseif prox_d < min_proximal_diameter_cm
            reason = "below_min_proximal_diameter"
        end
        path_score = vessel_path_score(path, score)
        if isempty(reason)
            push!(eligible, path)
            push!(eligible_score, path_score)
        end
        push!(rows, (
            role=String(role),
            surface=path.surface,
            points=length(path.points),
            length_cm=len_cm,
            min_diameter_cm=minimum(path.diameters_cm),
            mean_diameter_cm=mean_d,
            max_diameter_cm=max_d,
            proximal_diameter_cm=prox_d,
            score=path_score,
            selected=false,
            rank=0,
            reason=isempty(reason) ? "eligible" : reason,
        ))
    end

    order = sort(collect(eachindex(eligible)); by=i -> (-eligible_score[i], eligible[i].surface))
    selected_count = max_paths <= 0 ? length(order) : min(max_paths, length(order))
    selected_surfaces = Set{String}()
    selected_rank = Dict{String, Int}()
    selected = XCATSeedPath[]
    for (rank, idx) in enumerate(order[1:selected_count])
        path = eligible[idx]
        push!(selected, path)
        push!(selected_surfaces, path.surface)
        selected_rank[path.surface] = rank
    end

    audited = NamedTuple[]
    for row in rows
        if row.surface in selected_surfaces
            push!(audited, merge(row, (selected=true, rank=selected_rank[row.surface], reason="selected")))
        elseif row.reason == "eligible"
            push!(audited, merge(row, (reason="eligible_not_selected",)))
        else
            push!(audited, row)
        end
    end
    audited = sort(audited; by=r -> (r.selected ? 0 : 1, r.rank == 0 ? typemax(Int) : r.rank, -r.score, r.surface))
    return selected, audited
end

function raw_label_centerline_paths(raw_labels, dims, labels::Vector{UInt16}, keep_x,
                                    spacing_cm::SVector{3, Float64};
                                    label_prefix::String="raw_label_",
                                    min_slices::Int=4,
                                    min_voxels_per_slice::Int=1,
                                    max_slice_gap::Int=2,
                                    max_step_cm::Float64=1.25)
    label_to_row = Dict(label => idx for (idx, label) in enumerate(labels))
    nx, ny, nz = dims
    nlabels = length(labels)
    counts = zeros(Int, nlabels, nz)
    sum_x = zeros(Float64, nlabels, nz)
    sum_y = zeros(Float64, nlabels, nz)
    sum_z = zeros(Float64, nlabels, nz)
    keep = Set(keep_x)

    for idx0 in 0:(length(raw_labels) - 1)
        lab = raw_labels[idx0 + 1]
        row = get(label_to_row, lab, 0)
        row == 0 && continue
        x_i = (idx0 % nx) + 1
        x_i in keep || continue
        y_i = (div(idx0, nx) % ny) + 1
        z_i = div(idx0, nx * ny) + 1
        x = (x_i - 0.5) * spacing_cm[1]
        y = (y_i - 0.5) * spacing_cm[2]
        z = (z_i - 0.5) * spacing_cm[3]
        counts[row, z_i] += 1
        sum_x[row, z_i] += x
        sum_y[row, z_i] += y
        sum_z[row, z_i] += z
    end

    paths = XCATSeedPath[]
    for (row, label) in enumerate(labels)
        all_points = SVector{3, Float64}[]
        all_diameters = Float64[]
        all_point_indices = Int[]
        for z_i in 1:nz
            c = counts[row, z_i]
            c >= min_voxels_per_slice || continue
            push!(all_points, SVector(sum_x[row, z_i] / c, sum_y[row, z_i] / c, sum_z[row, z_i] / c))
            # Equivalent circular diameter from this slice's voxelized cross-section.
            area_cm2 = c * spacing_cm[1] * spacing_cm[2]
            push!(all_diameters, 2.0 * sqrt(area_cm2 / pi))
            push!(all_point_indices, z_i)
        end
        length(all_points) >= min_slices || continue

        run_start = 1
        part = 1
        for idx in 2:(length(all_points) + 1)
            split_here = idx > length(all_points) ||
                all_point_indices[idx] - all_point_indices[idx - 1] > max_slice_gap ||
                norm(all_points[idx] - all_points[idx - 1]) > max_step_cm
            split_here || continue

            if idx - run_start >= min_slices
                label_text = string(label_prefix, label)
                if part > 1 || idx <= length(all_points)
                    label_text *= "_part$(part)"
                end
                push!(paths, XCATSeedPath(
                    label_text,
                    all_point_indices[run_start:(idx - 1)],
                    all_points[run_start:(idx - 1)],
                    all_diameters[run_start:(idx - 1)],
                ))
                part += 1
            end
            run_start = idx
        end
    end
    return paths
end

function mask_index_from_point(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                               spacing_cm::SVector{3, Float64}, point::SVector{3, Float64})
    return (
        floor(Int, (point[1] - origin_cm[1]) / spacing_cm[1]) + 1,
        floor(Int, (point[2] - origin_cm[2]) / spacing_cm[2]) + 1,
        floor(Int, (point[3] - origin_cm[3]) / spacing_cm[3]) + 1,
    )
end

function mask_value(mask::BitArray{3}, i::Int, j::Int, k::Int)
    dims = size(mask)
    (1 <= i <= dims[1] && 1 <= j <= dims[2] && 1 <= k <= dims[3]) || return false
    return mask[i, j, k]
end

function mask_value_at_point(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                             spacing_cm::SVector{3, Float64}, point::SVector{3, Float64})
    return mask_value(mask, mask_index_from_point(mask, origin_cm, spacing_cm, point)...)
end

function vessel_path_mask_counts(path::XCATSeedPath, mask::BitArray{3},
                                 origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64})
    outside = count(p -> !mask_value_at_point(mask, origin_cm, spacing_cm, p), path.points)
    return (outside_points=outside, inside_points=length(path.points) - outside)
end

function mark_paths_in_mask!(mask::BitArray{3}, paths::Vector{XCATSeedPath},
                             origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64};
                             radius_voxels::Int=1)
    radius_voxels < 0 && error("radius_voxels must be >= 0")
    dims = size(mask)
    step_cm = 0.5 * minimum(spacing_cm)
    mark_point!(point) = begin
        i0, j0, k0 = mask_index_from_point(mask, origin_cm, spacing_cm, point)
        for k in max(1, k0 - radius_voxels):min(dims[3], k0 + radius_voxels),
            j in max(1, j0 - radius_voxels):min(dims[2], j0 + radius_voxels),
            i in max(1, i0 - radius_voxels):min(dims[1], i0 + radius_voxels)
            mask[i, j, k] = true
        end
    end
    for path in paths
        isempty(path.points) && continue
        mark_point!(first(path.points))
        for idx in 2:length(path.points)
            a = path.points[idx - 1]
            b = path.points[idx]
            segment_length = norm(b - a)
            steps = max(1, ceil(Int, segment_length / step_cm))
            for step in 1:steps
                mark_point!(a + (step / steps) * (b - a))
            end
        end
    end
    return mask
end

function nearest_mask_point_cm(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                               spacing_cm::SVector{3, Float64}, point::SVector{3, Float64};
                               max_radius_voxels::Int=2)
    max_radius_voxels < 0 && error("max_radius_voxels must be >= 0")
    i0, j0, k0 = mask_index_from_point(mask, origin_cm, spacing_cm, point)
    dims = size(mask)
    best_point = point
    best_d2 = Inf
    for radius in 0:max_radius_voxels
        for k in max(1, k0 - radius):min(dims[3], k0 + radius),
            j in max(1, j0 - radius):min(dims[2], j0 + radius),
            i in max(1, i0 - radius):min(dims[1], i0 + radius)
            mask[i, j, k] || continue
            candidate = SVector(
                origin_cm[1] + (i - 0.5) * spacing_cm[1],
                origin_cm[2] + (j - 0.5) * spacing_cm[2],
                origin_cm[3] + (k - 0.5) * spacing_cm[3])
            d2 = sum(abs2, candidate - point)
            if d2 < best_d2
                best_d2 = d2
                best_point = candidate
            end
        end
        isfinite(best_d2) && return (found=true, point=best_point, distance_cm=sqrt(best_d2))
    end
    return (found=false, point=point, distance_cm=Inf)
end

function vertex_touches_xcat_segment(tree::GrowthTree, vertex::Int)
    incoming = tree.incoming_segment[vertex]
    incoming != 0 && tree.is_xcat[incoming] && return true
    for child in tree.children[vertex]
        seg = tree.incoming_segment[child]
        seg != 0 && tree.is_xcat[seg] && return true
    end
    return false
end

function vertex_touches_grown_segment(tree::GrowthTree, vertex::Int)
    incoming = tree.incoming_segment[vertex]
    incoming != 0 && !tree.is_xcat[incoming] && return true
    for child in tree.children[vertex]
        seg = tree.incoming_segment[child]
        seg != 0 && !tree.is_xcat[seg] && return true
    end
    return false
end

function repair_grown_segments_to_mask!(tree::GrowthTree, mask::BitArray{3},
                                        origin_cm::SVector{3, Float64},
                                        spacing_cm::SVector{3, Float64};
                                        max_radius_voxels::Int=2,
                                        max_passes::Int=3)
    max_passes < 1 && error("max_passes must be >= 1")
    repaired = 0
    skipped = 0
    max_shift_cm = 0.0
    completed_passes = 0
    for pass in 1:max_passes
        completed_passes = pass
        repaired_this_pass = 0
        for vertex in eachindex(tree.vertices)
            vertex_touches_grown_segment(tree, vertex) || continue
            vertex_touches_xcat_segment(tree, vertex) && continue
            point = tree.vertices[vertex]
            mask_value_at_point(mask, origin_cm, spacing_cm, point) && continue
            nearest = nearest_mask_point_cm(mask, origin_cm, spacing_cm, point;
                max_radius_voxels=max_radius_voxels)
            if nearest.found
                tree.vertices[vertex] = nearest.point
                repaired += 1
                repaired_this_pass += 1
                max_shift_cm = max(max_shift_cm, nearest.distance_cm)
            else
                skipped += 1
            end
        end
        n_before = length(tree.segment_start)
        for seg_id in 1:n_before
            tree.is_xcat[seg_id] && continue
            a = tree.vertices[tree.segment_start[seg_id]]
            b = tree.vertices[tree.segment_end[seg_id]]
            midpoint = 0.5 * (a + b)
            mask_value_at_point(mask, origin_cm, spacing_cm, midpoint) && continue
            nearest = nearest_mask_point_cm(mask, origin_cm, spacing_cm, midpoint;
                max_radius_voxels=max_radius_voxels)
            if nearest.found
                VascularTreeSim._split_segment!(tree, seg_id, nearest.point)
                repaired += 1
                repaired_this_pass += 1
                max_shift_cm = max(max_shift_cm, nearest.distance_cm)
            else
                skipped += 1
            end
        end
        repaired_this_pass == 0 && break
    end
    return (repaired=repaired, skipped=skipped, max_shift_cm=max_shift_cm, passes=completed_passes)
end

repair_grown_segment_midpoints_to_mask!(args...; kwargs...) = repair_grown_segments_to_mask!(args...; kwargs...)

function write_vessel_path_audit_csv(path::AbstractString, rows;
                                     mask::Union{Nothing, BitArray{3}}=nothing,
                                     origin_cm::Union{Nothing, SVector{3, Float64}}=nothing,
                                     spacing_cm::Union{Nothing, SVector{3, Float64}}=nothing,
                                     paths::Vector{XCATSeedPath}=XCATSeedPath[])
    by_surface = Dict(p.surface => p for p in paths)
    open(path, "w") do io
        println(io, "role,surface,selected,rank,reason,points,length_mm,min_diameter_mm,mean_diameter_mm,max_diameter_mm,proximal_diameter_mm,score,outside_mask_points")
        for row in rows
            outside = ""
            if mask !== nothing && origin_cm !== nothing && spacing_cm !== nothing && haskey(by_surface, row.surface)
                outside = string(vessel_path_mask_counts(by_surface[row.surface], mask, origin_cm, spacing_cm).outside_points)
            end
            @printf(io, "%s,%s,%s,%d,%s,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%.8f,%s\n",
                row.role, row.surface, string(row.selected), row.rank, row.reason, row.points,
                10.0 * row.length_cm,
                10.0 * row.min_diameter_cm,
                10.0 * row.mean_diameter_cm,
                10.0 * row.max_diameter_cm,
                10.0 * row.proximal_diameter_cm,
                row.score,
                outside)
        end
    end
    return path
end

function distance_to_mask_cm(mask::BitArray{3}, origin_cm::SVector{3, Float64},
                             spacing_cm::SVector{3, Float64}, point::SVector{3, Float64};
                             max_distance_cm::Float64)
    i0, j0, k0 = mask_index_from_point(mask, origin_cm, spacing_cm, point)
    mask_value(mask, i0, j0, k0) && return 0.0
    dims = size(mask)
    max_radius = ceil(Int, max_distance_cm / minimum(spacing_cm))
    best2 = Inf
    for r in 1:max_radius
        for k in max(1, k0 - r):min(dims[3], k0 + r),
            j in max(1, j0 - r):min(dims[2], j0 + r),
            i in max(1, i0 - r):min(dims[1], i0 + r)
            (abs(i - i0) == r || abs(j - j0) == r || abs(k - k0) == r) || continue
            mask[i, j, k] || continue
            delta = SVector(
                (i - i0) * spacing_cm[1],
                (j - j0) * spacing_cm[2],
                (k - k0) * spacing_cm[3],
            )
            best2 = min(best2, sum(abs2, delta))
        end
        if isfinite(best2)
            return sqrt(best2)
        end
    end
    return Inf
end

function clip_paths_to_mask_proximity(paths::Vector{XCATSeedPath}, mask::BitArray{3},
                                      origin_cm::SVector{3, Float64}, spacing_cm::SVector{3, Float64};
                                      max_distance_cm::Float64,
                                      min_points::Int=4,
                                      max_step_cm::Float64=1.25)
    clipped = XCATSeedPath[]
    for path in paths
        keep = Bool[
            distance_to_mask_cm(mask, origin_cm, spacing_cm, p; max_distance_cm=max_distance_cm) <= max_distance_cm
            for p in path.points
        ]
        run_start = nothing
        part = 1
        for idx in 1:(length(path.points) + 1)
            split_here = idx > length(path.points) ||
                !keep[idx] ||
                (idx > 1 && run_start !== nothing && norm(path.points[idx] - path.points[idx - 1]) > max_step_cm)

            if split_here
                if run_start !== nothing && idx - run_start >= min_points
                    suffix = part == 1 ? "_near" : "_near$(part)"
                    push!(clipped, XCATSeedPath(
                        path.surface * suffix,
                        path.point_indices[run_start:(idx - 1)],
                        path.points[run_start:(idx - 1)],
                        path.diameters_cm[run_start:(idx - 1)],
                    ))
                    part += 1
                end
                run_start = nothing
            elseif run_start === nothing
                run_start = idx
            end
        end
    end
    return clipped
end

function write_xcat_paths_csv(path::AbstractString, paths::Vector{XCATSeedPath})
    open(path, "w") do io
        println(io, "surface,point_index,raw_x_cm,raw_y_cm,raw_z_cm,diameter_cm,diameter_mm")
        for p in paths
            for idx in eachindex(p.points)
                point = p.points[idx]
                @printf(io, "%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    p.surface, p.point_indices[idx], point[1], point[2], point[3],
                    p.diameters_cm[idx], 10.0 * p.diameters_cm[idx])
            end
        end
    end
    return path
end

function make_single_raw_seed_tree(path::XCATSeedPath; terminal_diameter_cm::Float64, root_diameter_cm::Union{Nothing, Float64}=nothing)
    length(path.points) >= 2 || error("Cannot build seed tree from fewer than two points: $(path.surface)")
    vertices = copy(path.points)
    parent_vertex = [idx == 1 ? 0 : idx - 1 for idx in eachindex(vertices)]
    incoming_segment = [idx == 1 ? 0 : idx - 1 for idx in eachindex(vertices)]
    children = [Int[] for _ in eachindex(vertices)]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = zeros(Int, length(vertices))

    for idx in 2:length(vertices)
        push!(children[idx - 1], idx)
        push!(segment_start, idx - 1)
        push!(segment_end, idx)
        push!(segment_diameter_cm, 0.5 * (path.diameters_cm[idx - 1] + path.diameters_cm[idx]))
        push!(segment_label, path.surface)
        push!(is_xcat, true)
    end

    return GrowthTree(
        path.surface,
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
        terminal_diameter_cm,
        1,
        root_diameter_cm === nothing ? maximum(segment_diameter_cm) : root_diameter_cm,
    )
end

function merge_growth_trees(name::String, trees::Vector{GrowthTree}; terminal_diameter_cm::Float64)
    vertices = SVector{3, Float64}[]
    parent_vertex = Int[]
    incoming_segment = Int[]
    children = Vector{Int}[]
    segment_start = Int[]
    segment_end = Int[]
    segment_diameter_cm = Float64[]
    segment_label = String[]
    is_xcat = Bool[]
    subtree_terminal_count = Int[]

    root_vertex = 1
    root_diameter_cm = 0.0
    seg_offset = 0
    for tree in trees
        vertex_offset = length(vertices)
        isempty(vertices) && (root_vertex = tree.root_vertex)
        root_diameter_cm = max(root_diameter_cm, tree.root_diameter_cm)
        append!(vertices, tree.vertices)
        append!(parent_vertex, [p == 0 ? 0 : p + vertex_offset for p in tree.parent_vertex])
        append!(incoming_segment, [s == 0 ? 0 : s + seg_offset for s in tree.incoming_segment])
        append!(children, [[child + vertex_offset for child in kids] for kids in tree.children])
        append!(subtree_terminal_count, tree.subtree_terminal_count)
        append!(segment_start, [s + vertex_offset for s in tree.segment_start])
        append!(segment_end, [s + vertex_offset for s in tree.segment_end])
        append!(segment_diameter_cm, tree.segment_diameter_cm)
        append!(segment_label, tree.segment_label)
        append!(is_xcat, tree.is_xcat)
        seg_offset = length(segment_start)
    end

    return GrowthTree(
        name,
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
        terminal_diameter_cm,
        root_vertex,
        root_diameter_cm,
    )
end

function coverage_stats_for_tree(tree::GrowthTree, points::Matrix{Float64})
    dists = Float64[]
    size(points, 1) == 0 && return (added=0, p50=NaN, p95=NaN, max=NaN)
    for i in axes(points, 1)
        p = SVector(points[i, 1], points[i, 2], points[i, 3])
        push!(dists, VascularTreeSim._tree_segment_distance_cm(tree, p))
    end
    return (
        added=0,
        p50=quantile(dists, 0.50),
        p95=quantile(dists, 0.95),
        max=maximum(dists),
    )
end

function artery_weight(path::XCATSeedPath)
    return max(proximal_root_diameter_cm(path), 1e-6)^3
end

function append_disconnected_overlay_path!(tree::GrowthTree, path::XCATSeedPath)
    length(path.points) >= 2 || return 0

    push!(tree.vertices, first(path.points))
    root = length(tree.vertices)
    push!(tree.parent_vertex, 0)
    push!(tree.incoming_segment, 0)
    push!(tree.children, Int[])
    push!(tree.subtree_terminal_count, 0)

    prev = root
    added = 0
    for idx in 2:length(path.points)
        point = path.points[idx]
        norm(point - tree.vertices[prev]) < 1e-8 && continue
        push!(tree.vertices, point)
        vid = length(tree.vertices)
        push!(tree.parent_vertex, prev)
        push!(tree.incoming_segment, length(tree.segment_start) + 1)
        push!(tree.children, Int[])
        push!(tree.subtree_terminal_count, 0)
        push!(tree.children[prev], vid)

        push!(tree.segment_start, prev)
        push!(tree.segment_end, vid)
        push!(tree.segment_diameter_cm, 0.5 * (path.diameters_cm[idx - 1] + path.diameters_cm[idx]))
        push!(tree.segment_label, path.surface)
        push!(tree.is_xcat, true)

        prev = vid
        added += 1
    end
    return added
end

function append_vein_overlays!(tree::GrowthTree, vein_paths::Vector{XCATSeedPath})
    added = 0
    for path in vein_paths
        added += append_disconnected_overlay_path!(tree, path)
    end
    return added
end

append_fixed_overlays!(tree::GrowthTree, paths::Vector{XCATSeedPath}) = append_vein_overlays!(tree, paths)

function write_right_leg_trial_summary(path::AbstractString, raw_path, organ_ids_path, log_path,
                                       artery_source, vein_source, xcat_info, keep_desc,
                                       kept_counts, crop_lo, crop_hi, domain, mask_info,
                                       artery_paths, connector_gaps, vein_paths, overlay_segments,
                                       tree, stats; max_branches::Int, block_size::Int,
                                       terminal_um::Float64, artery_tree_count::Int=1,
                                       per_tree_branch_cap::Int=max_branches,
                                       artery_overlay_segments::Int=0)
    ext_cm = (mask_info.hi_cm .- mask_info.lo_cm) .+ domain.spacing_cm
    xcat_segments = count(tree.is_xcat)
    grown_segments = length(tree.segment_start) - xcat_segments
    open(path, "w") do io
        println(io, "# XCAT Right-Leg Trial")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- XCAT raw: $(raw_path)")
        println(io, "- Organ IDs: $(organ_ids_path)")
        println(io, "- XCAT log: $(log_path)")
        println(io, "- Artery source: $(artery_source)")
        println(io, "- Vein source: $(vein_source)")
        println(io, "- Source dims: $(xcat_info.dims)")
        println(io, "- Source slice range: $(xcat_info.start_slice)-$(xcat_info.end_slice)")
        println(io, "- Source spacing: $(round.(Tuple(xcat_info.spacing_cm .* 10); digits=3)) mm")
        println(io, "- Kept right-leg half: $(keep_desc)")
        println(io, "- leg_right half counts: lower-x=$(kept_counts[1]), upper-x=$(kept_counts[2])")
        println(io, "- Crop lo index: $(crop_lo)")
        println(io, "- Crop hi index: $(crop_hi)")
        println(io, "- Cropped dims: $(size(domain.mask))")
        println(io, "- Foreground right-leg muscle/foot-muscle voxels: $(mask_info.foreground)")
        println(io, "- Cropped extent: $(round(ext_cm[1]; digits=2)) x $(round(ext_cm[2]; digits=2)) x $(round(ext_cm[3]; digits=2)) cm")
        println(io, "- Terminal branch diameter: $(round(terminal_um; digits=3)) um")
        println(io, "- Total branch budget requested: $(max_branches)")
        println(io, "- Independent artery seed trees: $(artery_tree_count)")
        println(io, "- Per-artery-tree branch cap: $(per_tree_branch_cap)")
        println(io, "- Coverage/route block size: $(block_size)")
        @printf(io, "- Fixed vessel proximity filter: kept path sections within %.1f mm of the muscle mask\n",
            MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM * 10.0)
        println(io)
        println(io, "## XCAT Artery Seeds")
        println(io)
        for (idx, path) in enumerate(artery_paths)
            role = idx == 1 ? "growth seed" : "display overlay"
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter\n",
                "$(path.surface) ($(role))", length(path.points), path_length_cm(path) * 10.0,
                minimum(path.diameters_cm) * 10.0, maximum(path.diameters_cm) * 10.0)
        end
        println(io, "- Artery overlay segments added after growth: $(artery_overlay_segments)")
        if isempty(connector_gaps)
            println(io, "- No forced connector segments were created between raw artery seed paths.")
        else
            for (surface, gap) in sort(collect(connector_gaps); by=first)
                @printf(io, "- %s attachment gap: %.2f mm\n", surface, gap * 10.0)
            end
        end
        println(io)
        println(io, "## XCAT Vein Overlays")
        println(io)
        for path in vein_paths
            @printf(io, "- %s: %d points, %.2f mm length, %.2f-%.2f mm diameter\n",
                path.surface, length(path.points), path_length_cm(path) * 10.0,
                minimum(path.diameters_cm) * 10.0, maximum(path.diameters_cm) * 10.0)
        end
        println(io, "- Vein overlay segments added after growth: $(overlay_segments)")
        println(io)
        println(io, "## Growth Result")
        println(io)
        println(io, "- Added predicted branches: $(stats.added)")
        println(io, "- XCAT fixed artery + vein segments shown: $(xcat_segments)")
        println(io, "- Predicted/grown segments shown: $(grown_segments)")
        println(io, "- Total segments shown: $(length(tree.segment_start))")
        println(io, "- Terminal vertices shown: $(count(v -> isempty(v), tree.children))")
        println(io, "- Minimum displayed diameter: $(round(minimum(tree.segment_diameter_cm) * 1e4; digits=1)) um")
        println(io, "- Maximum displayed diameter: $(round(maximum(tree.segment_diameter_cm) * 10; digits=2)) mm")
        println(io, "- Coverage p50 before vein overlay: $(round(stats.p50 * 10; digits=2)) mm")
        println(io, "- Coverage p95 before vein overlay: $(round(stats.p95 * 10; digits=2)) mm")
        println(io, "- Coverage max before vein overlay: $(round(stats.max * 10; digits=2)) mm")
        println(io)
        println(io, "Raw-label vessel paths are clipped to sections near the muscle-domain mask before use/display. The longest remaining raw-label `arteries_rleg` path is used as the fixed arterial growth seed. Other main raw-label arteries and veins are added only after growth as disconnected fixed overlays, so they are visible in the viewer but do not attract or seed predicted branches.")
    end
    return path
end

function main_right_leg_xcat_trial()
    Random.seed!(42)
    max_branches = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 1000
    terminal_um = length(ARGS) >= 2 ? parse(Float64, ARGS[2]) : 100.0
    output_dir = length(ARGS) >= 3 ? ARGS[3] : RIGHT_LEG_TRIAL_DEFAULT_OUTPUT_DIR
    input_dir = length(ARGS) >= 4 ? ARGS[4] : RIGHT_LEG_TRIAL_DEFAULT_INPUT_DIR
    geometry_dir = length(ARGS) >= 5 ? ARGS[5] : RIGHT_LEG_TRIAL_DEFAULT_GEOMETRY_DIR
    block_size = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 10
    max_main_veins = length(ARGS) >= 7 ? parse(Int, ARGS[7]) : 6
    terminal_cm = terminal_um / 1e4

    raw_path = joinpath(input_dir, "both_legs_act_1.raw")
    organ_ids_path = joinpath(input_dir, "organ_ids.txt")
    log_path = joinpath(input_dir, "both_legs_log")
    artery_source = "voxelized raw label volume: arteries_rleg"
    vein_source = "voxelized raw label volume: veins_rleg"

    mkpath(output_dir)
    xcat_info = parse_xcat_log(log_path)
    name_to_labels, _ = parse_organ_ids(organ_ids_path)
    raw_labels = load_xcat_uint16_raw(raw_path, xcat_info.dims)

    right_leg_labels = labels_for(name_to_labels, ["leg_right"])
    keep_x, lower_count, upper_count, keep_desc = choose_right_half(raw_labels, xcat_info.dims, right_leg_labels)
    target_labels = right_leg_muscle_labels(name_to_labels)
    full_mask, muscle_counts = build_right_leg_muscle_mask(raw_labels, xcat_info.dims, target_labels, keep_x)
    mask, origin_cm, crop_lo, crop_hi = crop_mask(full_mask, xcat_info.spacing_cm; pad=2)
    nhdr_path, mask_raw_path = write_right_leg_mask_artifacts(output_dir, mask, origin_cm, xcat_info.spacing_cm)

    domain, mask_info = build_domain_from_mask(mask, origin_cm, xcat_info.spacing_cm)
    artery_labels = labels_for(name_to_labels, ["arteries_rleg"])
    artery_paths_all = raw_label_centerline_paths(raw_labels, xcat_info.dims, artery_labels, keep_x,
        xcat_info.spacing_cm; label_prefix="arteries_rleg_raw_")
    artery_paths_near = clip_paths_to_mask_proximity(artery_paths_all, mask, origin_cm, xcat_info.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    artery_paths = select_longest_paths(artery_paths_near; max_paths=6, min_length_cm=8.0)
    isempty(artery_paths) && error("No usable raw-label artery seed paths were selected")
    ordered_artery_paths = artery_paths
    connector_gaps = Dict{String, Float64}()
    growth_artery_path = first(ordered_artery_paths)
    artery_overlay_paths = ordered_artery_paths[2:end]
    tree = make_single_raw_seed_tree(growth_artery_path; terminal_diameter_cm=terminal_cm)
    tree.name = RIGHT_LEG_TRIAL_TREE_NAME
    trees = Dict(RIGHT_LEG_TRIAL_TREE_NAME => tree)
    per_tree_branch_cap = max_branches

    coverage_points = coverage_target_points_blockwise(domain; block_size=block_size)
    graph_points = copy(coverage_points)
    viewer_points = coverage_target_points_blockwise(domain; block_size=max(block_size, 6))
    vein_labels = labels_for(name_to_labels, ["veins_rleg"])
    vein_paths_all = raw_label_centerline_paths(raw_labels, xcat_info.dims, vein_labels, keep_x,
        xcat_info.spacing_cm; label_prefix="vein_raw_")
    vein_paths_near = clip_paths_to_mask_proximity(vein_paths_all, mask, origin_cm, xcat_info.spacing_cm;
        max_distance_cm=MAX_FIXED_VESSEL_DISTANCE_TO_MUSCLE_CM)
    main_vein_paths = select_main_paths(vein_paths_near; max_paths=max_main_veins, min_length_cm=8.0)

    println("=" ^ 78)
    println("XCAT right-leg trial")
    println("Input: $(input_dir)")
    println("Geometry: $(geometry_dir)")
    println("Output: $(output_dir)")
    println("Terminal diameter: $(terminal_um) um")
    println("Max predicted branches: $(max_branches)")
    println("Block size: $(block_size)")
    println("Kept half: $(keep_desc) lower=$(lower_count) upper=$(upper_count)")
    println("Cropped dims: $(size(mask)), foreground=$(count(mask)), muscle_labels=$(length(muscle_counts))")
    fixed_artery_segments = length(tree.segment_start)
    println("[artery seeds] selected=$(length(ordered_artery_paths)) from $(length(artery_paths_all)) raw-space artery paths ($(length(artery_paths_near)) near-muscle clipped paths) fixed_segments=$(fixed_artery_segments)")
    println("[artery seeds] growth_seed=$(growth_artery_path.surface) overlay_paths=$(length(artery_overlay_paths)) no_forced_connectors=true")
    println("[vein overlays] selected=$(length(main_vein_paths)) from $(length(vein_paths_all)) raw-space vein paths ($(length(vein_paths_near)) near-muscle clipped paths)")
    println("[growth] coverage=$(size(coverage_points, 1)) graph=$(size(graph_points, 1))")
    flush(stdout)

    _, _, growth_stats = grow_trees_mcp!(trees, domain;
        coverage_points_cm=coverage_points,
        graph_points_cm=graph_points,
        effective_supply_radius_cm=0.55,
        capillary_diameter_cm=terminal_cm,
        max_new_branches_per_tree=max_branches,
        graph_neighbors=16,
        min_frontier_separation_cm=0.75,
        max_path_nodes=24,
        target_p95_distance_cm=0.80,
        frontier_batch=12,
        gamma=3.0,
        smooth_passes=8,
        spline_density=3,
        max_segment_length_cm=0.35,
        turn_penalty=0.35,
        graph_jitter_cm=0.04,
        use_gpu=false)

    restore_xcat_seed_diameters!(tree, [growth_artery_path])
    total_added = growth_stats[RIGHT_LEG_TRIAL_TREE_NAME].added
    coverage_stats = coverage_stats_for_tree(tree, coverage_points)
    stats = (added=total_added, p50=coverage_stats.p50, p95=coverage_stats.p95, max=coverage_stats.max)
    artery_overlay_segments = append_fixed_overlays!(tree, artery_overlay_paths)
    overlay_segments = append_vein_overlays!(tree, main_vein_paths)

    csv_path = joinpath(output_dir, "xcat_right_leg_trial_segments.csv")
    xcat_fixed_csv = joinpath(output_dir, "xcat_right_leg_fixed_vessels.csv")
    raw_artery_csv = joinpath(output_dir, "xcat_right_leg_raw_artery_centerlines.csv")
    raw_vein_csv = joinpath(output_dir, "xcat_right_leg_raw_vein_centerlines.csv")
    domain_csv = joinpath(output_dir, "xcat_right_leg_trial_domain_points.csv")
    viewer_path = joinpath(output_dir, "xcat_right_leg_trial_viewer.html")
    summary_path = joinpath(output_dir, "summary.md")

    write_growth_csv(csv_path, RIGHT_LEG_TRIAL_TREE_NAME, tree)
    write_xcat_seed_csv(xcat_fixed_csv, tree)
    write_xcat_paths_csv(raw_artery_csv, ordered_artery_paths)
    write_xcat_paths_csv(raw_vein_csv, main_vein_paths)
    write_points_csv(domain_csv, viewer_points)
    write_femoral_viewer(viewer_path, mask, origin_cm, xcat_info.spacing_cm, mask_info, viewer_points, tree)
    write_right_leg_trial_summary(summary_path, raw_path, organ_ids_path, log_path,
        artery_source, vein_source, xcat_info, keep_desc, (lower_count, upper_count),
        crop_lo, crop_hi, domain, mask_info, ordered_artery_paths, connector_gaps,
        main_vein_paths, overlay_segments, tree, stats;
        max_branches=max_branches, block_size=block_size, terminal_um=terminal_um,
        artery_tree_count=1, per_tree_branch_cap=per_tree_branch_cap,
        artery_overlay_segments=artery_overlay_segments)

    println()
    println("XCAT Right-Leg Trial Result")
    println("  added predicted branches = $(stats.added)")
    println("  xcat fixed segments shown = $(count(tree.is_xcat))")
    println("  vein overlay segments     = $(overlay_segments)")
    println("  total segments shown      = $(length(tree.segment_start))")
    println("  p95 coverage              = $(round(stats.p95 * 10; digits=2)) mm")
    println("  viewer                    = $(viewer_path)")
    println()
    println("Wrote")
    println("  $(nhdr_path)")
    println("  $(mask_raw_path)")
    println("  $(csv_path)")
    println("  $(xcat_fixed_csv)")
    println("  $(raw_artery_csv)")
    println("  $(raw_vein_csv)")
    println("  $(domain_csv)")
    println("  $(viewer_path)")
    println("  $(summary_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main_right_leg_xcat_trial()
end
