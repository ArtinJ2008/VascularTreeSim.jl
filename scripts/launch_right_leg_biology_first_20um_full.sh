#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453"
REPO="${WORK_ROOT}/repo/VascularTreeSim.jl"
INPUT="${WORK_ROOT}/input/both_legs_xcat_input"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${WORK_ROOT}/output/right_leg_biology_first_20um_explicit_${STAMP}"
LOG_DIR="${WORK_ROOT}/logs"
LOG="${LOG_DIR}/right_leg_biology_first_20um_explicit_${STAMP}.log"

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${REPO}"

export JULIA_DEPOT_PATH="/media/molloi-lab/2TB4/Artin/julia_depot"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-16}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VTS_CHECKPOINT_DIR="${OUT}/checkpoints"
export VTS_CHECKPOINT_INTERVAL_SECONDS="${VTS_CHECKPOINT_INTERVAL_SECONDS:-1800}"
export VTS_CHECKPOINT_KEEP="${VTS_CHECKPOINT_KEEP:-3}"

export VTS_BIOLOGY_FIRST="true"
export VTS_TARGET_BRANCH_COUNT_MODE="explicit"
export VTS_ALLOW_POST_GROWTH_SUBDIVISION="false"
export VTS_ADAPTIVE_BRANCH_CAPS="true"
export VTS_EXPLICIT_TERMINAL_UM="20.0"
export VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM="20.0"
export VTS_TOPOLOGY_AUDIT_MIN_DIAMETER_UM="20.0"
export VTS_SUBDIVIDE_XCAT_TERMINALS="false"

export VTS_XCAT_NRB_PATH="${VTS_XCAT_NRB_PATH:-${INPUT}/both_legs_1.nrb}"
export VTS_XCAT_VESSEL_SOURCE="nrb"
export VTS_NRB_DOMAIN_VOXEL_SPACING_CM="${VTS_NRB_DOMAIN_VOXEL_SPACING_CM:-0.1}"
export VTS_NRB_DOMAIN_SAMPLES="${VTS_NRB_DOMAIN_SAMPLES:-48,48}"
export VTS_ROUTE_DILATION_VOXELS="${VTS_ROUTE_DILATION_VOXELS:-0}"
export VTS_MAX_GRAPH_BLOCK_SIZE="${VTS_MAX_GRAPH_BLOCK_SIZE:-1}"
export VTS_MIN_GRAPH_LARGEST_COMPONENT_FRACTION="${VTS_MIN_GRAPH_LARGEST_COMPONENT_FRACTION:-0.95}"
export VTS_MAX_UNCLAIMED_TARGET_FRACTION="${VTS_MAX_UNCLAIMED_TARGET_FRACTION:-0.02}"

export VTS_TARGET_TISSUE_MODE="${VTS_TARGET_TISSUE_MODE:-soft}"
export VTS_TARGET_DEMAND_MODE="${VTS_TARGET_DEMAND_MODE:-weighted}"
export VTS_TARGET_DEMAND_WEIGHTS="${VTS_TARGET_DEMAND_WEIGHTS:-muscle=1.0,skin=0.05,fat=0.02,envelope=0.03,other_soft=0.05}"
export VTS_MAX_COVERAGE_POINTS="${VTS_MAX_COVERAGE_POINTS:-5000000}"

export VTS_DISTAL_MURRAY_GAMMA="${VTS_DISTAL_MURRAY_GAMMA:-3.0}"
export VTS_PROXIMAL_MURRAY_GAMMA="${VTS_PROXIMAL_MURRAY_GAMMA:-2.0}"
export VTS_MURRAY_TRANSITION_UM="${VTS_MURRAY_TRANSITION_UM:-200.0}"

export VTS_MAX_BRANCH_LENGTH_CM="${VTS_MAX_BRANCH_LENGTH_CM:-Inf}"
export VTS_MAX_ROUTE_TORTUOSITY="${VTS_MAX_ROUTE_TORTUOSITY:-4.0}"
export VTS_HYDRAULIC_COST_WEIGHT="${VTS_HYDRAULIC_COST_WEIGHT:-1.0}"
export VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR="${VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR:-4}"
export VTS_HYDRAULIC_REFERENCE_LENGTH_CM="${VTS_HYDRAULIC_REFERENCE_LENGTH_CM:-1.0}"
export VTS_HYDRAULIC_EXISTING_PATH_WEIGHT="${VTS_HYDRAULIC_EXISTING_PATH_WEIGHT:-1.0}"
export VTS_PATH_LENGTH_COST_WEIGHT="${VTS_PATH_LENGTH_COST_WEIGHT:-0.5}"
export VTS_BRANCHPOINT_DEPTH_COST_WEIGHT="${VTS_BRANCHPOINT_DEPTH_COST_WEIGHT:-0.02}"
export VTS_MIN_HYDRAULIC_SCORE_CM="${VTS_MIN_HYDRAULIC_SCORE_CM:-0.0}"
export VTS_MAX_NEW_BRANCH_RESISTANCE_REL="${VTS_MAX_NEW_BRANCH_RESISTANCE_REL:-Inf}"
export VTS_MAX_TERMINAL_PATH_RESISTANCE_REL="${VTS_MAX_TERMINAL_PATH_RESISTANCE_REL:-Inf}"
export VTS_MAX_TERMINAL_PATH_LENGTH_CM="${VTS_MAX_TERMINAL_PATH_LENGTH_CM:-Inf}"
export VTS_MAX_BRANCHPOINT_GENERATION="${VTS_MAX_BRANCHPOINT_GENERATION:-80}"
export VTS_BLOOD_VISCOSITY_POISE="${VTS_BLOOD_VISCOSITY_POISE:-0.035}"
export VTS_MIN_INITIAL_TERRITORY_FRACTION="${VTS_MIN_INITIAL_TERRITORY_FRACTION:-0.10}"
export VTS_MAX_INITIAL_TERRITORY_POINTS="${VTS_MAX_INITIAL_TERRITORY_POINTS:-8192}"

export VTS_EXPORT_FULL_GEOMETRY_CSV="${VTS_EXPORT_FULL_GEOMETRY_CSV:-false}"
export VTS_EXPORT_FULL_ARTERIAL_CSV="${VTS_EXPORT_FULL_ARTERIAL_CSV:-false}"
export VTS_TERMINAL_PATH_AUDIT_MAX_ROWS="${VTS_TERMINAL_PATH_AUDIT_MAX_ROWS:-500000}"
export VTS_TERMINAL_PATH_AUDIT_INCLUDE_SEGMENTS="${VTS_TERMINAL_PATH_AUDIT_INCLUDE_SEGMENTS:-false}"

TARGET="${VTS_TARGET_BRANCHES:-auto}"

{
    echo "run=right_leg_biology_first_20um_explicit"
    echo "started=$(date --iso-8601=seconds)"
    echo "repo=${REPO}"
    echo "input=${INPUT}"
    echo "xcat_nrb_path=${VTS_XCAT_NRB_PATH}"
    echo "output=${OUT}"
    echo "log=${LOG}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "terminal_um=20.0"
    echo "explicit_terminal_um=${VTS_EXPLICIT_TERMINAL_UM}"
    echo "target=${TARGET}"
    echo "target_branch_count_mode=${VTS_TARGET_BRANCH_COUNT_MODE}"
    echo "adaptive_branch_caps=${VTS_ADAPTIVE_BRANCH_CAPS}"
    echo "allow_post_growth_subdivision=${VTS_ALLOW_POST_GROWTH_SUBDIVISION}"
    echo "graph_block_size=1"
    echo "graph_neighbors=24"
    echo "frontier_batch=4096"
    echo "min_frontier_separation_cm=0.005"
    echo "max_segment_length_cm=0.05"
    echo "max_path_nodes=64"
    echo "graph_jitter_cm=0.005"
    echo "snap_terminal_to_target=true"
    echo "max_terminal_snap_cm=0.15"
    echo "coverage_multiplier=1.25"
    echo "use_indexed_anchor=true"
    echo "use_astar_routing=true"
    echo "frontier_candidate_factor=64"
    echo "fixed_vein_exports=6"
    echo "max_anchor_gap_cm=1.25"
    echo "growth_artery_seeds=0"
    echo "growth_artery_min_length_cm=3.0"
    echo "fixed_artery_min_length_cm=3.0"
    echo "fixed_vein_min_length_cm=8.0"
    echo "path_length_cost_weight=${VTS_PATH_LENGTH_COST_WEIGHT}"
    echo "branchpoint_depth_cost_weight=${VTS_BRANCHPOINT_DEPTH_COST_WEIGHT}"
    echo "max_terminal_path_length_cm=${VTS_MAX_TERMINAL_PATH_LENGTH_CM}"
    echo "max_branchpoint_generation=${VTS_MAX_BRANCHPOINT_GENERATION}"
} > "${OUT}/run_metadata.txt"

exec /home/molloi-lab/.juliaup/bin/julia --project=. \
    examples/right_leg_xcat_50um_gpu.jl \
    20.0 \
    "${OUT}" \
    "${VTS_XCAT_NRB_PATH}" \
    "${TARGET}" \
    4096 \
    1 \
    0.005 \
    0.05 \
    24 \
    64 \
    0.005 \
    true \
    0.15 \
    1.25 \
    true \
    true \
    64 \
    6 \
    1.25 \
    0 \
    3.0 \
    3.0 \
    8.0 \
    > "${LOG}" 2>&1
