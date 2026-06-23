#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453"
REPO="${WORK_ROOT}/repo/VascularTreeSim.jl"
INPUT="${WORK_ROOT}/input/both_legs_xcat_input"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${WORK_ROOT}/output/right_leg_active_xcat_30um_full_${STAMP}"
LOG_DIR="${WORK_ROOT}/logs"
LOG="${LOG_DIR}/right_leg_active_xcat_30um_full_${STAMP}.log"

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${REPO}"

export JULIA_DEPOT_PATH="/media/molloi-lab/2TB4/Artin/julia_depot"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export VTS_MAX_GRAPH_BLOCK_SIZE="${VTS_MAX_GRAPH_BLOCK_SIZE:-1}"
export VTS_TARGET_TISSUE_MODE="${VTS_TARGET_TISSUE_MODE:-soft}"
export VTS_TARGET_DEMAND_MODE="${VTS_TARGET_DEMAND_MODE:-uniform}"
export VTS_TARGET_DEMAND_WEIGHTS="${VTS_TARGET_DEMAND_WEIGHTS:-muscle=1.0,skin=0.2,fat=0.05,envelope=0.1,other_soft=0.1}"
export VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM="${VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM:-50.0}"
export VTS_MAX_BRANCH_LENGTH_CM="${VTS_MAX_BRANCH_LENGTH_CM:-Inf}"
export VTS_MAX_ROUTE_TORTUOSITY="${VTS_MAX_ROUTE_TORTUOSITY:-8.0}"
export VTS_HYDRAULIC_COST_WEIGHT="${VTS_HYDRAULIC_COST_WEIGHT:-0.25}"
export VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR="${VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR:-2}"
export VTS_HYDRAULIC_REFERENCE_LENGTH_CM="${VTS_HYDRAULIC_REFERENCE_LENGTH_CM:-1.0}"
export VTS_HYDRAULIC_EXISTING_PATH_WEIGHT="${VTS_HYDRAULIC_EXISTING_PATH_WEIGHT:-0.5}"
export VTS_MIN_HYDRAULIC_SCORE_CM="${VTS_MIN_HYDRAULIC_SCORE_CM:-0.0}"
export VTS_MAX_NEW_BRANCH_RESISTANCE_REL="${VTS_MAX_NEW_BRANCH_RESISTANCE_REL:-Inf}"
export VTS_MAX_TERMINAL_PATH_RESISTANCE_REL="${VTS_MAX_TERMINAL_PATH_RESISTANCE_REL:-Inf}"
export VTS_BLOOD_VISCOSITY_POISE="${VTS_BLOOD_VISCOSITY_POISE:-0.035}"
export VTS_XCAT_NRB_PATH="${VTS_XCAT_NRB_PATH:-${INPUT}/both_legs_1.nrb}"
export VTS_XCAT_VESSEL_SOURCE="${VTS_XCAT_VESSEL_SOURCE:-nrb}"
export VTS_NRB_DOMAIN_VOXEL_SPACING_CM="${VTS_NRB_DOMAIN_VOXEL_SPACING_CM:-0.1}"
export VTS_NRB_DOMAIN_SAMPLES="${VTS_NRB_DOMAIN_SAMPLES:-48,48}"
export VTS_EXPLICIT_TERMINAL_UM="${VTS_EXPLICIT_TERMINAL_UM:-200.0}"
export VTS_SUBDIVISION_MAX_LD_RATIO="${VTS_SUBDIVISION_MAX_LD_RATIO:-25.0}"
export VTS_SUBDIVISION_CLIP_BELOW_UM="${VTS_SUBDIVISION_CLIP_BELOW_UM:-0.0}"

{
    echo "run=right_leg_active_xcat_30um_full"
    echo "started=$(date --iso-8601=seconds)"
    echo "repo=${REPO}"
    echo "input=${INPUT}"
    echo "xcat_nrb_path=${VTS_XCAT_NRB_PATH}"
    echo "xcat_vessel_source=${VTS_XCAT_VESSEL_SOURCE}"
    echo "nrb_domain_voxel_spacing_cm=${VTS_NRB_DOMAIN_VOXEL_SPACING_CM}"
    echo "nrb_domain_samples=${VTS_NRB_DOMAIN_SAMPLES}"
    echo "explicit_terminal_um=${VTS_EXPLICIT_TERMINAL_UM}"
    echo "subdivision_max_ld_ratio=${VTS_SUBDIVISION_MAX_LD_RATIO}"
    echo "subdivision_clip_below_um=${VTS_SUBDIVISION_CLIP_BELOW_UM}"
    echo "output=${OUT}"
    echo "log=${LOG}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "terminal_um=30.0"
    echo "target=auto"
    echo "frontier_batch=4096"
    echo "graph_block_size=1"
    echo "max_graph_block_size=${VTS_MAX_GRAPH_BLOCK_SIZE}"
    echo "max_segment_length_cm=0.05"
    echo "route_dilation_voxels=0"
    echo "target_tissue_mode=${VTS_TARGET_TISSUE_MODE}"
    echo "target_demand_mode=${VTS_TARGET_DEMAND_MODE}"
    echo "target_demand_weights=${VTS_TARGET_DEMAND_WEIGHTS}"
    echo "flow_explicit_min_diameter_um=${VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM}"
    echo "max_branch_length_cm=${VTS_MAX_BRANCH_LENGTH_CM}"
    echo "max_route_tortuosity=${VTS_MAX_ROUTE_TORTUOSITY}"
    echo "hydraulic_cost_weight=${VTS_HYDRAULIC_COST_WEIGHT}"
    echo "hydraulic_rank_candidate_factor=${VTS_HYDRAULIC_RANK_CANDIDATE_FACTOR}"
    echo "hydraulic_reference_length_cm=${VTS_HYDRAULIC_REFERENCE_LENGTH_CM}"
    echo "hydraulic_existing_path_weight=${VTS_HYDRAULIC_EXISTING_PATH_WEIGHT}"
    echo "min_hydraulic_score_cm=${VTS_MIN_HYDRAULIC_SCORE_CM}"
    echo "max_new_branch_resistance_rel=${VTS_MAX_NEW_BRANCH_RESISTANCE_REL}"
    echo "max_terminal_path_resistance_rel=${VTS_MAX_TERMINAL_PATH_RESISTANCE_REL}"
    echo "blood_viscosity_poise=${VTS_BLOOD_VISCOSITY_POISE}"
    echo "fixed_vein_exports=6"
    echo "main_xcat_artery_seeds=all eligible"
} > "${OUT}/run_metadata.txt"

exec /home/molloi-lab/.juliaup/bin/julia --project=. \
    examples/right_leg_xcat_50um_gpu.jl \
    30.0 \
    "${OUT}" \
    "${INPUT}/both_legs_act_1.raw" \
    "${INPUT}/organ_ids.txt" \
    "${INPUT}/both_legs_log" \
    auto \
    4096 \
    1 \
    0.01 \
    0.05 \
    24 \
    48 \
    0.005 \
    true \
    0.15 \
    1.10 \
    true \
    true \
    32 \
    6 \
    1.25 \
    0 \
    > "${LOG}" 2>&1
