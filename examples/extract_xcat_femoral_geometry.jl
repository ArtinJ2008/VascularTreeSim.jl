"""
    extract_xcat_femoral_geometry.jl

Extract right-leg vessel centerlines from an XCAT grouped NRB file and write
inspectable CSV files. XCAT repeats many vessel objects under names like
`arteries_rleg`; this script keeps every patch and ranks artery patches that
look like the femoral trunk by axial span, length, and diameter.

Usage:
    julia --project=. examples/extract_xcat_femoral_geometry.jl [nrb_path] [output_dir]
"""

using VascularTreeSim
using LinearAlgebra
using Printf
using Statistics
using Dates
using StaticArrays

const DEFAULT_NRB = raw"C:\Users\msdjv\Downloads\both_thigh_corrected_1.nrb"
const DEFAULT_LOG = raw"C:\Users\msdjv\Downloads\both_thigh_corrected_log"
const DEFAULT_OUTPUT_DIR = joinpath(dirname(@__DIR__), "output", "xcat_femoral_geometry")
const RIGHT_ARTERY_GROUP = "arteries_rleg"
const RIGHT_VEIN_GROUP = "veins_rleg"
const XCAT_Z_REFERENCE_CM = 100.0

struct CenterlineSummary
    group::String
    name::String
    npoints::Int
    length_mm::Float64
    mean_radius_mm::Float64
    max_radius_mm::Float64
    min_diameter_mm::Float64
    mean_diameter_mm::Float64
    max_diameter_mm::Float64
    lo::SVector{3, Float64}
    hi::SVector{3, Float64}
    start::SVector{3, Float64}
    stop::SVector{3, Float64}
    axial_span_mm::Float64
    femoral_score::Float64
end

function csv_text(text::AbstractString)
    escaped = replace(String(text), "\"" => "\"\"")
    return "\"" * escaped * "\""
end

function parse_xcat_log(path::AbstractString)
    text = read(path, String)
    parse_int(pattern) = parse(Int, match(pattern, text).captures[1])
    parse_float(pattern) = parse(Float64, match(pattern, text).captures[1])
    nx = parse_int(r"x_array_size\s*=\s*(\d+)")
    ny = parse_int(r"y_array_size\s*=\s*(\d+)")
    start_slice = parse_int(r"starting slice number\s*=\s*(\d+)")
    end_slice = parse_int(r"ending slice number\s*=\s*(\d+)")
    pixel_width_cm = parse_float(r"pixel width\s*=\s*([0-9.]+)")
    slice_width_cm = parse_float(r"slice width\s*=\s*([0-9.]+)")
    nz = end_slice - start_slice + 1
    return (
        dims=(nx, ny, nz),
        spacing_cm=SVector(pixel_width_cm, pixel_width_cm, slice_width_cm),
        start_slice=start_slice,
        end_slice=end_slice,
    )
end

function xcat_mm_to_raw_cm(p::SVector{3, Float64}, log_info)
    x_cm = 0.1 * p[1] + 0.5 * log_info.dims[1] * log_info.spacing_cm[1]
    y_cm = 0.1 * p[2] + 0.5 * log_info.dims[2] * log_info.spacing_cm[2]
    z_cm = (XCAT_Z_REFERENCE_CM - 0.1 * p[3]) - log_info.start_slice * log_info.spacing_cm[3]
    return SVector(x_cm, y_cm, z_cm)
end

function in_raw_volume(p::SVector{3, Float64}, log_info)
    hi = SVector(
        log_info.dims[1] * log_info.spacing_cm[1],
        log_info.dims[2] * log_info.spacing_cm[2],
        log_info.dims[3] * log_info.spacing_cm[3],
    )
    return all(p .>= SVector(0.0, 0.0, 0.0)) && all(p .<= hi)
end

function centerline_bounds(cl::XCATCenterline)
    lo = SVector(Inf, Inf, Inf)
    hi = SVector(-Inf, -Inf, -Inf)
    for p in cl.centers
        lo = min.(lo, p)
        hi = max.(hi, p)
    end
    return lo, hi
end

function global_bounds(surfaces)
    lo = SVector(Inf, Inf, Inf)
    hi = SVector(-Inf, -Inf, -Inf)
    for surface in surfaces
        slo, shi = xcat_bounds(surface)
        lo = min.(lo, slo)
        hi = max.(hi, shi)
    end
    return lo, hi
end

function choose_long_axis(surfaces)
    leg = [s for s in surfaces if xcat_group_name(s) == "leg_right"]
    source = isempty(leg) ? surfaces : leg
    lo, hi = global_bounds(source)
    spans = hi .- lo
    axis = argmax(Tuple(spans))
    return axis, lo, hi
end

function summarize_centerline(group::String, cl::XCATCenterline, long_axis::Int)
    lo, hi = centerline_bounds(cl)
    length_mm = VascularTreeSim.xcat_centerline_length_mm(cl)
    diameters = 2.0 .* cl.radii
    axial_span = hi[long_axis] - lo[long_axis]
    # Long, thick, long-axis-aligned patches are the best femoral trunk
    # candidates. This is a ranking signal only; every patch is still exported.
    femoral_score = axial_span * mean(diameters) + 0.25 * length_mm + 0.5 * maximum(diameters)
    return CenterlineSummary(
        group,
        cl.name,
        length(cl.centers),
        length_mm,
        mean(cl.radii),
        maximum(cl.radii),
        minimum(diameters),
        mean(diameters),
        maximum(diameters),
        lo,
        hi,
        first(cl.centers),
        last(cl.centers),
        axial_span,
        femoral_score,
    )
end

function write_group_counts(path::AbstractString, surfaces)
    counts = Dict{String, Int}()
    for surface in surfaces
        group = xcat_group_name(surface)
        counts[group] = get(counts, group, 0) + 1
    end
    open(path, "w") do io
        println(io, "group,count")
        for group in sort(collect(keys(counts)))
            println(io, "$(csv_text(group)),$(counts[group])")
        end
    end
    return path
end

function write_centerline_points(path::AbstractString, centerlines::AbstractVector{XCATCenterline})
    open(path, "w") do io
        println(io, "group,surface,point_index,x_mm,y_mm,z_mm,x_cm,y_cm,z_cm,radius_mm,diameter_mm,radius_cm,diameter_cm")
        for cl in centerlines
            group = xcat_group_name(cl.name)
            for (idx, p) in enumerate(cl.centers)
                radius = cl.radii[idx]
                @printf(io, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(group), csv_text(cl.name), idx,
                    p[1], p[2], p[3], 0.1 * p[1], 0.1 * p[2], 0.1 * p[3],
                    radius, 2.0 * radius, 0.1 * radius, 0.2 * radius)
            end
        end
    end
    return path
end

function write_segment_points(path::AbstractString, centerlines::AbstractVector{XCATCenterline})
    open(path, "w") do io
        println(io, "group,surface,segment_index,x1_mm,y1_mm,z1_mm,x2_mm,y2_mm,z2_mm,diameter1_mm,diameter2_mm,mean_diameter_mm,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,mean_diameter_cm")
        for cl in centerlines
            group = xcat_group_name(cl.name)
            for idx in 1:(length(cl.centers) - 1)
                a = cl.centers[idx]
                b = cl.centers[idx + 1]
                d1 = 2.0 * cl.radii[idx]
                d2 = 2.0 * cl.radii[idx + 1]
                md = 0.5 * (d1 + d2)
                @printf(io, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(group), csv_text(cl.name), idx,
                    a[1], a[2], a[3], b[1], b[2], b[3], d1, d2, md,
                    0.1 * a[1], 0.1 * a[2], 0.1 * a[3],
                    0.1 * b[1], 0.1 * b[2], 0.1 * b[3], 0.1 * md)
            end
        end
    end
    return path
end

function write_summaries(path::AbstractString, summaries::AbstractVector{CenterlineSummary})
    open(path, "w") do io
        println(io, "rank,group,surface,npoints,length_mm,mean_radius_mm,max_radius_mm,min_diameter_mm,mean_diameter_mm,max_diameter_mm,lo_x_mm,lo_y_mm,lo_z_mm,hi_x_mm,hi_y_mm,hi_z_mm,start_x_mm,start_y_mm,start_z_mm,end_x_mm,end_y_mm,end_z_mm,axial_span_mm,femoral_score")
        for (rank, s) in enumerate(summaries)
            @printf(io, "%d,%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                rank, csv_text(s.group), csv_text(s.name), s.npoints,
                s.length_mm, s.mean_radius_mm, s.max_radius_mm,
                s.min_diameter_mm, s.mean_diameter_mm, s.max_diameter_mm,
                s.lo[1], s.lo[2], s.lo[3], s.hi[1], s.hi[2], s.hi[3],
                s.start[1], s.start[2], s.start[3], s.stop[1], s.stop[2], s.stop[3],
                s.axial_span_mm, s.femoral_score)
        end
    end
    return path
end

function write_selected_femoral(path::AbstractString, selected::AbstractVector{XCATCenterline})
    open(path, "w") do io
        println(io, "surface,point_index,x_mm,y_mm,z_mm,x_cm,y_cm,z_cm,radius_mm,diameter_mm,radius_cm,diameter_cm")
        for cl in selected
            for (idx, p) in enumerate(cl.centers)
                radius = cl.radii[idx]
                @printf(io, "%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(cl.name), idx,
                    p[1], p[2], p[3], 0.1 * p[1], 0.1 * p[2], 0.1 * p[3],
                    radius, 2.0 * radius, 0.1 * radius, 0.2 * radius)
            end
        end
    end
    return path
end

function write_raw_space_centerlines(path::AbstractString, centerlines::AbstractVector{XCATCenterline}, log_info)
    open(path, "w") do io
        println(io, "group,surface,point_index,raw_x_cm,raw_y_cm,raw_z_cm,xcat_x_mm,xcat_y_mm,xcat_z_mm,radius_mm,diameter_mm,radius_cm,diameter_cm")
        for cl in centerlines
            group = xcat_group_name(cl.name)
            for (idx, p) in enumerate(cl.centers)
                raw_p = xcat_mm_to_raw_cm(p, log_info)
                in_raw_volume(raw_p, log_info) || continue
                radius = cl.radii[idx]
                @printf(io, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(group), csv_text(cl.name), idx,
                    raw_p[1], raw_p[2], raw_p[3],
                    p[1], p[2], p[3],
                    radius, 2.0 * radius, 0.1 * radius, 0.2 * radius)
            end
        end
    end
    return path
end

function raw_space_path_stats(cl::XCATCenterline, log_info)
    raw_points = SVector{3, Float64}[]
    raw_diameters_cm = Float64[]
    point_indices = Int[]
    for (idx, p) in enumerate(cl.centers)
        raw_p = xcat_mm_to_raw_cm(p, log_info)
        in_raw_volume(raw_p, log_info) || continue
        push!(raw_points, raw_p)
        push!(raw_diameters_cm, 0.2 * cl.radii[idx])
        push!(point_indices, idx)
    end
    if length(raw_points) < 2
        return nothing
    end
    length_cm = sum(norm(raw_points[i] - raw_points[i - 1]) for i in 2:length(raw_points))
    mean_diameter_cm = mean(raw_diameters_cm)
    max_diameter_cm = maximum(raw_diameters_cm)
    score = length_cm * mean_diameter_cm
    return (
        centerline=cl,
        point_indices=point_indices,
        raw_points=raw_points,
        raw_diameters_cm=raw_diameters_cm,
        length_cm=length_cm,
        mean_diameter_cm=mean_diameter_cm,
        max_diameter_cm=max_diameter_cm,
        score=score,
    )
end

function write_raw_space_path_stats(path::AbstractString, stats)
    open(path, "w") do io
        println(io, "rank,group,surface,points,segments,length_mm,mean_diameter_mm,max_diameter_mm,score")
        for (rank, st) in enumerate(stats)
            @printf(io, "%d,%s,%s,%d,%d,%.8f,%.8f,%.8f,%.8f\n",
                rank, csv_text(xcat_group_name(st.centerline.name)), csv_text(st.centerline.name),
                length(st.raw_points), length(st.raw_points) - 1,
                st.length_cm * 10.0,
                st.mean_diameter_cm * 10.0,
                st.max_diameter_cm * 10.0,
                st.score)
        end
    end
    return path
end

function write_selected_raw_space_centerlines(path::AbstractString, stats)
    open(path, "w") do io
        println(io, "group,surface,point_index,raw_x_cm,raw_y_cm,raw_z_cm,radius_mm,diameter_mm,radius_cm,diameter_cm")
        for st in stats
            cl = st.centerline
            group = xcat_group_name(cl.name)
            for local_idx in eachindex(st.raw_points)
                p = st.raw_points[local_idx]
                diameter_cm = st.raw_diameters_cm[local_idx]
                radius_cm = 0.5 * diameter_cm
                @printf(io, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(group), csv_text(cl.name), st.point_indices[local_idx],
                    p[1], p[2], p[3],
                    radius_cm * 10.0, diameter_cm * 10.0,
                    radius_cm, diameter_cm)
            end
        end
    end
    return path
end

function write_raw_space_segments(path::AbstractString, centerlines::AbstractVector{XCATCenterline}, log_info)
    open(path, "w") do io
        println(io, "group,surface,segment_index,raw_x1_cm,raw_y1_cm,raw_z1_cm,raw_x2_cm,raw_y2_cm,raw_z2_cm,xcat_x1_mm,xcat_y1_mm,xcat_z1_mm,xcat_x2_mm,xcat_y2_mm,xcat_z2_mm,mean_diameter_mm,mean_diameter_cm")
        for cl in centerlines
            group = xcat_group_name(cl.name)
            for idx in 1:(length(cl.centers) - 1)
                a = cl.centers[idx]
                b = cl.centers[idx + 1]
                raw_a = xcat_mm_to_raw_cm(a, log_info)
                raw_b = xcat_mm_to_raw_cm(b, log_info)
                (in_raw_volume(raw_a, log_info) && in_raw_volume(raw_b, log_info)) || continue
                md = cl.radii[idx] + cl.radii[idx + 1]
                @printf(io, "%s,%s,%d,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n",
                    csv_text(group), csv_text(cl.name), idx,
                    raw_a[1], raw_a[2], raw_a[3],
                    raw_b[1], raw_b[2], raw_b[3],
                    a[1], a[2], a[3],
                    b[1], b[2], b[3],
                    md, 0.1 * md)
            end
        end
    end
    return path
end

function print_top(io, title::AbstractString, summaries; n::Int=8)
    println(io, title)
    for s in Iterators.take(summaries, min(n, length(summaries)))
        @printf(io, "  %-28s length=%8.3f mm meanD=%7.3f mm maxD=%7.3f mm axial=%8.3f mm score=%9.3f\n",
            s.name, s.length_mm, s.mean_diameter_mm, s.max_diameter_mm,
            s.axial_span_mm, s.femoral_score)
    end
end

function main()
    nrb_path = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_NRB
    output_dir = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_OUTPUT_DIR
    log_path = length(ARGS) >= 3 ? ARGS[3] : DEFAULT_LOG
    mkpath(output_dir)

    println("=" ^ 78)
    println("XCAT femoral geometry extraction")
    println("NRB: $nrb_path")
    println("Output: $output_dir")
    println("Log: $log_path")
    println("Started: $(Dates.now())")
    flush(stdout)

    surfaces = parse_xcat_grouped_nrb(nrb_path)
    group_counts_path = write_group_counts(joinpath(output_dir, "xcat_group_counts.csv"), surfaces)
    long_axis, leg_lo, leg_hi = choose_long_axis(surfaces)
    axis_name = ("x", "y", "z")[long_axis]
    println("Parsed $(length(surfaces)) surfaces. Long leg axis: $axis_name")

    artery_surfaces = [s for s in surfaces if xcat_group_name(s) == RIGHT_ARTERY_GROUP]
    vein_surfaces = [s for s in surfaces if xcat_group_name(s) == RIGHT_VEIN_GROUP]
    println("Right-leg artery patches: $(length(artery_surfaces))")
    println("Right-leg vein patches: $(length(vein_surfaces))")

    artery_cls = XCATCenterline[]
    vein_cls = XCATCenterline[]
    for surface in artery_surfaces
        push!(artery_cls, xcat_centerline_from_surface(surface; circumferential_samples=64))
    end
    for surface in vein_surfaces
        push!(vein_cls, xcat_centerline_from_surface(surface; circumferential_samples=64))
    end

    artery_summaries = sort(
        [summarize_centerline(RIGHT_ARTERY_GROUP, cl, long_axis) for cl in artery_cls],
        by=s -> -s.femoral_score,
    )
    vein_summaries = sort(
        [summarize_centerline(RIGHT_VEIN_GROUP, cl, long_axis) for cl in vein_cls],
        by=s -> -s.femoral_score,
    )

    artery_points_path = write_centerline_points(joinpath(output_dir, "xcat_right_leg_artery_centerlines.csv"), artery_cls)
    artery_segments_path = write_segment_points(joinpath(output_dir, "xcat_right_leg_artery_segments.csv"), artery_cls)
    artery_summary_path = write_summaries(joinpath(output_dir, "xcat_right_leg_artery_summary_ranked.csv"), artery_summaries)
    vein_points_path = write_centerline_points(joinpath(output_dir, "xcat_right_leg_vein_centerlines.csv"), vein_cls)
    vein_segments_path = write_segment_points(joinpath(output_dir, "xcat_right_leg_vein_segments.csv"), vein_cls)
    vein_summary_path = write_summaries(joinpath(output_dir, "xcat_right_leg_vein_summary_ranked.csv"), vein_summaries)

    selected_names = Set(s.name for s in Iterators.take(artery_summaries, min(4, length(artery_summaries))))
    selected = [cl for cl in artery_cls if cl.name in selected_names]
    selected = sort(selected, by=cl -> findfirst(==(cl.name), [s.name for s in artery_summaries]))
    femoral_path = write_selected_femoral(joinpath(output_dir, "xcat_femoral_trunk_candidates_centerline.csv"), selected)
    raw_space_paths = String[]
    main_artery_stats = Any[]
    log_info = isfile(log_path) ? parse_xcat_log(log_path) : nothing
    if log_info !== nothing
        all_artery_raw_stats = sort(
            filter(!isnothing, [raw_space_path_stats(cl, log_info) for cl in artery_cls]),
            by=st -> -st.score,
        )
        # Main thigh arteries: enough XCAT structure for a realistic seed, but
        # not every short/small patch. With this dataset it selects six patches.
        eligible_main_arteries = [st for st in all_artery_raw_stats
            if (st.length_cm * 10.0 >= 80.0 || st.max_diameter_cm * 10.0 >= 2.5)]
        main_artery_stats = eligible_main_arteries[1:min(6, length(eligible_main_arteries))]
        push!(raw_space_paths, write_raw_space_centerlines(
            joinpath(output_dir, "xcat_right_leg_artery_centerlines_thigh_raw_space.csv"),
            artery_cls,
            log_info,
        ))
        push!(raw_space_paths, write_raw_space_centerlines(
            joinpath(output_dir, "xcat_femoral_trunk_candidates_thigh_raw_space.csv"),
            selected,
            log_info,
        ))
        push!(raw_space_paths, write_raw_space_path_stats(
            joinpath(output_dir, "xcat_right_leg_artery_thigh_raw_space_ranked.csv"),
            all_artery_raw_stats,
        ))
        push!(raw_space_paths, write_selected_raw_space_centerlines(
            joinpath(output_dir, "xcat_main_thigh_arteries_centerline.csv"),
            main_artery_stats,
        ))
        push!(raw_space_paths, write_raw_space_segments(
            joinpath(output_dir, "xcat_right_leg_artery_segments_thigh_raw_space.csv"),
            artery_cls,
            log_info,
        ))
        push!(raw_space_paths, write_raw_space_segments(
            joinpath(output_dir, "xcat_right_leg_vein_segments_thigh_raw_space.csv"),
            vein_cls,
            log_info,
        ))
    end

    summary_path = joinpath(output_dir, "README_xcat_femoral_geometry.md")
    open(summary_path, "w") do io
        println(io, "# XCAT Femoral Geometry Extraction")
        println(io)
        println(io, "- Date: $(Dates.now())")
        println(io, "- NRB: $nrb_path")
        println(io, "- XCAT log: $log_path")
        println(io, "- Parsed surfaces: $(length(surfaces))")
        println(io, "- Right-leg artery patches: $(length(artery_surfaces))")
        println(io, "- Right-leg vein patches: $(length(vein_surfaces))")
        println(io, "- Long leg axis used for ranking: $axis_name")
        @printf(io, "- `leg_right`/fallback bbox low: %.4f, %.4f, %.4f\n", leg_lo[1], leg_lo[2], leg_lo[3])
        @printf(io, "- `leg_right`/fallback bbox high: %.4f, %.4f, %.4f\n", leg_hi[1], leg_hi[2], leg_hi[3])
        println(io)
        println(io, "## Files")
        println(io)
        println(io, "- `xcat_group_counts.csv`: all parsed XCAT object groups and patch counts.")
        println(io, "- `xcat_right_leg_artery_centerlines.csv`: sampled centerline points for every `arteries_rleg` NRB patch.")
        println(io, "- `xcat_right_leg_artery_segments.csv`: adjacent centerline segments with diameters.")
        println(io, "- `xcat_right_leg_artery_summary_ranked.csv`: ranked artery patches; top rows are the femoral-trunk candidates.")
        println(io, "- `xcat_right_leg_vein_centerlines.csv`: sampled centerline points for every `veins_rleg` NRB patch.")
        println(io, "- `xcat_right_leg_vein_segments.csv`: adjacent vein centerline segments with diameters.")
        println(io, "- `xcat_right_leg_vein_summary_ranked.csv`: ranked vein patches.")
        println(io, "- `xcat_femoral_trunk_candidates_centerline.csv`: centerline points for the top artery candidates only.")
        if log_info !== nothing
            println(io, "- `xcat_femoral_trunk_candidates_thigh_raw_space.csv`: top artery candidate points transformed into the raw mask coordinate system and clipped to the exported thigh slab.")
            println(io, "- `xcat_right_leg_artery_centerlines_thigh_raw_space.csv`: all right-leg artery centerline points clipped to the exported thigh slab.")
            println(io, "- `xcat_right_leg_artery_thigh_raw_space_ranked.csv`: thigh-slab artery patch ranking by length and diameter.")
            println(io, "- `xcat_main_thigh_arteries_centerline.csv`: selected main artery seed patches for growth runs.")
            println(io, "- `xcat_right_leg_artery_segments_thigh_raw_space.csv`: all right-leg artery segments that fall inside the exported thigh slab.")
            println(io, "- `xcat_right_leg_vein_segments_thigh_raw_space.csv`: all right-leg vein segments that fall inside the exported thigh slab.")
        end
        println(io)
        println(io, "## Raw-Space Transform")
        println(io)
        if log_info === nothing
            println(io, "No XCAT log was found, so raw-space cropped files were not written.")
        else
            println(io, "The raw-space files use the XCAT image convention used by the color-code output:")
            println(io)
            println(io, "```")
            println(io, "raw_x_cm = xcat_x_cm + 0.5 * nx * pixel_width_cm")
            println(io, "raw_y_cm = xcat_y_cm + 0.5 * ny * pixel_width_cm")
            println(io, "raw_z_cm = (100.0 - xcat_z_cm) - start_slice * slice_width_cm")
            println(io, "```")
            println(io)
            println(io, "This maps the NRB world coordinates into the same index-origin coordinate system used by the current raw-mask viewer.")
        end
        println(io)
        println(io, "## Top Artery Candidates")
        println(io)
        println(io, "```")
        print_top(io, "Ranked by axial span, length, and diameter:", artery_summaries; n=10)
        println(io, "```")
        if log_info !== nothing
            println(io)
            println(io, "## Main Thigh Artery Seed Selection")
            println(io)
            println(io, "```")
            println(io, "Selected by thigh-slab length and diameter:")
            for st in main_artery_stats
                @printf(io, "  %-28s length=%8.2f mm meanD=%6.2f mm maxD=%6.2f mm\n",
                    st.centerline.name,
                    st.length_cm * 10.0,
                    st.mean_diameter_cm * 10.0,
                    st.max_diameter_cm * 10.0)
            end
            println(io, "```")
        end
        println(io)
        println(io, "These are XCAT NRB-derived centerlines, not the approximate seed path.")
    end

    print_top(stdout, "Top artery candidates:", artery_summaries; n=10)
    print_top(stdout, "Top vein candidates:", vein_summaries; n=6)
    println("Wrote:")
    for path in (
        group_counts_path, artery_points_path, artery_segments_path, artery_summary_path,
        vein_points_path, vein_segments_path, vein_summary_path, femoral_path, summary_path
    )
        println("  ", path)
    end
    for path in raw_space_paths
        println("  ", path)
    end
    println("Finished: $(Dates.now())")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
