"""
    CSV export with parent_segment_id column for deterministic topology reconstruction.
"""

function _parent_segment_id(tree::GrowthTree, s::Int)
    start_v = tree.segment_start[s]
    return 1 <= start_v <= length(tree.incoming_segment) ? tree.incoming_segment[start_v] : 0
end

function write_growth_csv(path::AbstractString, branch::String, tree::GrowthTree)
    open(path, "w") do io
        println(io, "branch,segment_id,parent_segment_id,x1_cm,y1_cm,z1_cm,x2_cm,y2_cm,z2_cm,xmid_cm,ymid_cm,zmid_cm,length_mm,diameter_um,label")
        for s in eachindex(tree.segment_start)
            a = tree.vertices[tree.segment_start[s]]
            b = tree.vertices[tree.segment_end[s]]
            length_cm = norm(b - a)
            diameter_cm = tree.segment_diameter_cm[s]
            println(io, join((branch, s, _parent_segment_id(tree, s),
                a[1], a[2], a[3],
                b[1], b[2], b[3],
                (a[1] + b[1]) / 2,
                (a[2] + b[2]) / 2,
                (a[3] + b[3]) / 2,
                10.0 * length_cm,
                1.0e4 * diameter_cm,
                tree.segment_label[s]), ","))
        end
    end
    return path
end
