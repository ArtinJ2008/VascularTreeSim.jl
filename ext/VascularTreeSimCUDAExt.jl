"""
    VascularTreeSimCUDAExt — CUDA acceleration for VascularTreeSim.

Provides GPU-accelerated distance kernels for the competitive growth engine.
Each GPU thread handles one coverage point and brute-forces over all segments,
leveraging massive parallelism instead of spatial indexing.

Activated automatically when `using CUDA` is called alongside VascularTreeSim.
"""
module VascularTreeSimCUDAExt

using VascularTreeSim
using CUDA
using StaticArrays

# ── GPU state container ──

"""
Holds device arrays that persist across growth rounds to minimize CPU↔GPU transfers.
Point arrays are uploaded once. Min distances and ownership live on GPU.
"""
mutable struct GPUDistanceState
    # Coverage point coordinates (uploaded once, read-only)
    d_px::CuVector{Float64}
    d_py::CuVector{Float64}
    d_pz::CuVector{Float64}
    # Persistent distance/ownership state
    d_min_dist::CuVector{Float64}
    d_owner::CuVector{Int32}
    # Optional component reachability state. When enabled, kernels ignore
    # target points whose route-graph component is not reachable from the
    # active tree seed.
    d_point_component::CuVector{Int32}
    d_tree_component_allowed::CuVector{UInt8}
    n_components::Int32
    reachability_enabled::Bool
    n_points::Int
end

# ── CUDA Kernels ──

"""
Brute-force minimum segment distance kernel.
Each thread = one point, iterates over ALL n_segs segments.
"""
function _kernel_min_seg_dist!(min_dist, owner,
                                px, py, pz,
                                point_component, tree_component_allowed,
                                ax, ay, az, bx, by, bz,
                                n_segs::Int32, tree_idx::Int32, inv_weight::Float64,
                                n_components::Int32, reachability_enabled::Bool)
    # min_dist stores EFFECTIVE distance = raw * (1/weight_owner); see
    # growth_engine.jl comment. `inv_weight` here is 1/weight of the tree being
    # added. A larger weight (smaller inv_weight) shrinks the effective
    # distance, letting the tree claim points up to `weight` times farther
    # (in raw distance) than a unit-weight competitor.
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing
    if reachability_enabled
        comp = @inbounds point_component[i]
        comp <= 0 && return nothing
        allowed_idx = comp + (tree_idx - Int32(1)) * n_components
        @inbounds tree_component_allowed[allowed_idx] == UInt8(0) && return nothing
    end

    best_d2 = Inf

    @inbounds for s in Int32(1):n_segs
        _abx = bx[s] - ax[s]
        _aby = by[s] - ay[s]
        _abz = bz[s] - az[s]
        _apx = px[i] - ax[s]
        _apy = py[i] - ay[s]
        _apz = pz[i] - az[s]

        denom = _abx * _abx + _aby * _aby + _abz * _abz

        if denom <= 1e-24
            d2 = _apx * _apx + _apy * _apy + _apz * _apz
        else
            t = (_apx * _abx + _apy * _aby + _apz * _abz) / denom
            t = max(0.0, min(1.0, t))
            dx = _apx - t * _abx
            dy = _apy - t * _aby
            dz = _apz - t * _abz
            d2 = dx * dx + dy * dy + dz * dz
        end

        d2 < best_d2 && (best_d2 = d2)
    end

    d_eff = sqrt(best_d2) * inv_weight
    @inbounds if d_eff < min_dist[i]
        min_dist[i] = d_eff
        owner[i] = tree_idx
    end
    return nothing
end

"""
Incremental kernel — only processes segments in range [seg_start, seg_end].
"""
function _kernel_min_seg_dist_range!(min_dist, owner,
                                      px, py, pz,
                                      point_component, tree_component_allowed,
                                      ax, ay, az, bx, by, bz,
                                      seg_start::Int32, seg_end::Int32, tree_idx::Int32,
                                      inv_weight::Float64, n_components::Int32,
                                      reachability_enabled::Bool)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing
    if reachability_enabled
        comp = @inbounds point_component[i]
        comp <= 0 && return nothing
        allowed_idx = comp + (tree_idx - Int32(1)) * n_components
        @inbounds tree_component_allowed[allowed_idx] == UInt8(0) && return nothing
    end

    best_d = @inbounds min_dist[i]
    best_owner = @inbounds owner[i]

    @inbounds for s in seg_start:seg_end
        _abx = bx[s] - ax[s]
        _aby = by[s] - ay[s]
        _abz = bz[s] - az[s]
        _apx = px[i] - ax[s]
        _apy = py[i] - ay[s]
        _apz = pz[i] - az[s]

        denom = _abx * _abx + _aby * _aby + _abz * _abz

        if denom <= 1e-24
            d2 = _apx * _apx + _apy * _apy + _apz * _apz
        else
            t = (_apx * _abx + _apy * _aby + _apz * _abz) / denom
            t = max(0.0, min(1.0, t))
            dx = _apx - t * _abx
            dy = _apy - t * _aby
            dz = _apz - t * _abz
            d2 = dx * dx + dy * dy + dz * dz
        end

        d_eff = sqrt(d2) * inv_weight
        if d_eff < best_d
            best_d = d_eff
            best_owner = tree_idx
        end
    end

    @inbounds min_dist[i] = best_d
    @inbounds owner[i] = best_owner
    return nothing
end

"""
Seed tree kernel — distance from all points to a single root vertex.
"""
function _kernel_seed_dist!(min_dist, owner,
                             px, py, pz,
                             point_component, tree_component_allowed,
                             rx::Float64, ry::Float64, rz::Float64,
                             tree_idx::Int32, inv_weight::Float64,
                             n_components::Int32, reachability_enabled::Bool)
    i = (blockIdx().x - Int32(1)) * blockDim().x + threadIdx().x
    i > length(px) && return nothing
    if reachability_enabled
        comp = @inbounds point_component[i]
        comp <= 0 && return nothing
        allowed_idx = comp + (tree_idx - Int32(1)) * n_components
        @inbounds tree_component_allowed[allowed_idx] == UInt8(0) && return nothing
    end

    @inbounds begin
        dx = px[i] - rx
        dy = py[i] - ry
        dz = pz[i] - rz
        d_eff = sqrt(dx * dx + dy * dy + dz * dz) * inv_weight
        if d_eff < min_dist[i]
            min_dist[i] = d_eff
            owner[i] = tree_idx
        end
    end
    return nothing
end

# ── Launch helpers ──

const CUDA_BLOCK_SIZE = 256

function _launch_config(n::Int)
    threads = min(CUDA_BLOCK_SIZE, n)
    blocks = cld(n, threads)
    return threads, blocks
end

# ── Interface implementations ──

function VascularTreeSim._gpu_init_distance_state(points_cm::Matrix{Float64})
    n = size(points_cm, 1)
    d_px = CuArray(points_cm[:, 1])
    d_py = CuArray(points_cm[:, 2])
    d_pz = CuArray(points_cm[:, 3])
    d_min_dist = CUDA.fill(Inf, n)
    d_owner = CUDA.zeros(Int32, n)
    d_point_component = CuArray(Int32[0])
    d_tree_component_allowed = CuArray(UInt8[0])
    println("[GPU] initialized: $(n) points on $(CUDA.name(CUDA.device()))")
    flush(stdout)
    return GPUDistanceState(d_px, d_py, d_pz, d_min_dist, d_owner,
        d_point_component, d_tree_component_allowed, Int32(0), false, n)
end

function VascularTreeSim._gpu_set_component_reachability!(state::GPUDistanceState,
                                                          point_components::Vector{Int},
                                                          tree_components::Vector{Set{Int}},
                                                          n_components::Int)
    n_components <= 0 && return nothing
    n_trees = length(tree_components)
    allowed = zeros(UInt8, n_components * n_trees)
    for ti in 1:n_trees
        offset = (ti - 1) * n_components
        for comp in tree_components[ti]
            1 <= comp <= n_components || continue
            allowed[offset + comp] = UInt8(1)
        end
    end
    CUDA.unsafe_free!(state.d_point_component)
    CUDA.unsafe_free!(state.d_tree_component_allowed)
    state.d_point_component = CuArray(Int32.(point_components))
    state.d_tree_component_allowed = CuArray(allowed)
    state.n_components = Int32(n_components)
    state.reachability_enabled = true
    println("[GPU] component reachability uploaded: $(n_components) components, $(n_trees) trees")
    flush(stdout)
    return nothing
end

function VascularTreeSim._gpu_full_distance_scan!(state::GPUDistanceState,
                                                    seg_idx::VascularTreeSim.SegmentSpatialIndex,
                                                    tree_idx::Int;
                                                    weight::Float64=1.0)
    nseg = length(seg_idx.ax)
    nseg == 0 && return nothing

    d_ax = CuArray(seg_idx.ax)
    d_ay = CuArray(seg_idx.ay)
    d_az = CuArray(seg_idx.az)
    d_bx = CuArray(seg_idx.bx)
    d_by = CuArray(seg_idx.by)
    d_bz = CuArray(seg_idx.bz)

    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_min_seg_dist!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        state.d_point_component, state.d_tree_component_allowed,
        d_ax, d_ay, d_az, d_bx, d_by, d_bz,
        Int32(nseg), Int32(tree_idx), 1.0 / weight,
        state.n_components, state.reachability_enabled)
    CUDA.synchronize()

    CUDA.unsafe_free!(d_ax); CUDA.unsafe_free!(d_ay); CUDA.unsafe_free!(d_az)
    CUDA.unsafe_free!(d_bx); CUDA.unsafe_free!(d_by); CUDA.unsafe_free!(d_bz)
    return nothing
end

function VascularTreeSim._gpu_seed_distance!(state::GPUDistanceState,
                                               root_vertex::SVector{3,Float64},
                                               tree_idx::Int;
                                               weight::Float64=1.0)
    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_seed_dist!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        state.d_point_component, state.d_tree_component_allowed,
        root_vertex[1], root_vertex[2], root_vertex[3],
        Int32(tree_idx), 1.0 / weight,
        state.n_components, state.reachability_enabled)
    CUDA.synchronize()
    return nothing
end

function VascularTreeSim._gpu_incremental_scan!(state::GPUDistanceState,
                                                  seg_idx::VascularTreeSim.SegmentSpatialIndex,
                                                  tree_idx::Int,
                                                  seg_start::Int, seg_end::Int;
                                                  weight::Float64=1.0)
    seg_end < seg_start && return nothing
    nseg_total = length(seg_idx.ax)
    nseg_total == 0 && return nothing

    d_ax = CuArray(@view seg_idx.ax[seg_start:seg_end])
    d_ay = CuArray(@view seg_idx.ay[seg_start:seg_end])
    d_az = CuArray(@view seg_idx.az[seg_start:seg_end])
    d_bx = CuArray(@view seg_idx.bx[seg_start:seg_end])
    d_by = CuArray(@view seg_idx.by[seg_start:seg_end])
    d_bz = CuArray(@view seg_idx.bz[seg_start:seg_end])
    nseg = length(d_ax)

    threads, blocks = _launch_config(state.n_points)
    @cuda threads=threads blocks=blocks _kernel_min_seg_dist_range!(
        state.d_min_dist, state.d_owner,
        state.d_px, state.d_py, state.d_pz,
        state.d_point_component, state.d_tree_component_allowed,
        d_ax, d_ay, d_az, d_bx, d_by, d_bz,
        Int32(1), Int32(nseg), Int32(tree_idx), 1.0 / weight,
        state.n_components, state.reachability_enabled)
    CUDA.synchronize()

    CUDA.unsafe_free!(d_ax); CUDA.unsafe_free!(d_ay); CUDA.unsafe_free!(d_az)
    CUDA.unsafe_free!(d_bx); CUDA.unsafe_free!(d_by); CUDA.unsafe_free!(d_bz)
    return nothing
end

function VascularTreeSim._gpu_download_distances(state::GPUDistanceState)
    min_dist = Array(state.d_min_dist)
    owner = Array(state.d_owner)
    return min_dist, owner
end

function VascularTreeSim._gpu_upload_distances!(state::GPUDistanceState,
                                                 min_dist::Vector{Float64},
                                                 owner::AbstractVector{<:Integer})
    length(min_dist) == state.n_points || error("min_dist length mismatch")
    length(owner) == state.n_points || error("owner length mismatch")
    copyto!(state.d_min_dist, min_dist)
    copyto!(state.d_owner, Int32.(owner))
    return nothing
end

function VascularTreeSim._gpu_free!(state::GPUDistanceState)
    CUDA.unsafe_free!(state.d_px); CUDA.unsafe_free!(state.d_py); CUDA.unsafe_free!(state.d_pz)
    CUDA.unsafe_free!(state.d_min_dist); CUDA.unsafe_free!(state.d_owner)
    CUDA.unsafe_free!(state.d_point_component); CUDA.unsafe_free!(state.d_tree_component_allowed)
    return nothing
end

# ── Extension initialization ──

function __init__()
    if CUDA.functional()
        VascularTreeSim._gpu_backend[] = :cuda
        dev = CUDA.device()
        mem_gb = round(CUDA.totalmem(dev) / 1024^3; digits=1)
        println("[VascularTreeSim] CUDA extension loaded: $(CUDA.name(dev)) ($(mem_gb) GB)")
        flush(stdout)
    else
        @warn "CUDA.jl loaded but no functional GPU detected. GPU acceleration disabled."
    end
end

end # module VascularTreeSimCUDAExt
