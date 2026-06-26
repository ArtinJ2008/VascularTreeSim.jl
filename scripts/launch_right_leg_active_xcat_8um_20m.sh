#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453"
REPO="${WORK_ROOT}/repo/VascularTreeSim.jl"
INPUT="${WORK_ROOT}/input/both_legs_xcat_input"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${WORK_ROOT}/output/right_leg_active_xcat_8um_20m_${STAMP}"
LOG_DIR="${WORK_ROOT}/logs"
LOG="${LOG_DIR}/right_leg_active_xcat_8um_20m_${STAMP}.log"

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${REPO}"

export JULIA_DEPOT_PATH="/media/molloi-lab/2TB4/Artin/julia_depot"
export JULIA_NUM_THREADS="${JULIA_NUM_THREADS:-16}"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export VTS_CHECKPOINT_DIR="${OUT}/checkpoints"
export VTS_CHECKPOINT_INTERVAL_SECONDS="${VTS_CHECKPOINT_INTERVAL_SECONDS:-14400}"
export VTS_CHECKPOINT_KEEP="${VTS_CHECKPOINT_KEEP:-2}"
export VTS_ROUTE_DILATION_VOXELS="${VTS_ROUTE_DILATION_VOXELS:-0}"
export VTS_MAX_GRAPH_BLOCK_SIZE="${VTS_MAX_GRAPH_BLOCK_SIZE:-1}"
export VTS_TARGET_TISSUE_MODE="${VTS_TARGET_TISSUE_MODE:-soft}"
export VTS_TARGET_DEMAND_MODE="${VTS_TARGET_DEMAND_MODE:-weighted}"
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
export VTS_MIN_INITIAL_TERRITORY_FRACTION="${VTS_MIN_INITIAL_TERRITORY_FRACTION:-0.05}"
export VTS_MAX_INITIAL_TERRITORY_POINTS="${VTS_MAX_INITIAL_TERRITORY_POINTS:-4096}"
export VTS_MIN_GRAPH_LARGEST_COMPONENT_FRACTION="${VTS_MIN_GRAPH_LARGEST_COMPONENT_FRACTION:-0.50}"
export VTS_MAX_UNCLAIMED_TARGET_FRACTION="${VTS_MAX_UNCLAIMED_TARGET_FRACTION:-0.05}"
export VTS_XCAT_NRB_PATH="${VTS_XCAT_NRB_PATH:-${INPUT}/both_legs_1.nrb}"
export VTS_XCAT_VESSEL_SOURCE="${VTS_XCAT_VESSEL_SOURCE:-nrb}"
export VTS_NRB_DOMAIN_VOXEL_SPACING_CM="${VTS_NRB_DOMAIN_VOXEL_SPACING_CM:-0.1}"
export VTS_NRB_DOMAIN_SAMPLES="${VTS_NRB_DOMAIN_SAMPLES:-48,48}"
export VTS_EXPLICIT_TERMINAL_UM="${VTS_EXPLICIT_TERMINAL_UM:-200.0}"
export VTS_DISTAL_MURRAY_GAMMA="${VTS_DISTAL_MURRAY_GAMMA:-3.0}"
export VTS_PROXIMAL_MURRAY_GAMMA="${VTS_PROXIMAL_MURRAY_GAMMA:-2.0}"
export VTS_MURRAY_TRANSITION_UM="${VTS_MURRAY_TRANSITION_UM:-200.0}"
export VTS_SUBDIVISION_MAX_LD_RATIO="${VTS_SUBDIVISION_MAX_LD_RATIO:-25.0}"
export VTS_SUBDIVISION_CLIP_BELOW_UM="${VTS_SUBDIVISION_CLIP_BELOW_UM:-0.0}"
export VTS_SUBDIVIDE_XCAT_TERMINALS="${VTS_SUBDIVIDE_XCAT_TERMINALS:-false}"
export VTS_EXPORT_FULL_GEOMETRY_CSV="${VTS_EXPORT_FULL_GEOMETRY_CSV:-false}"
export VTS_EXPORT_FULL_ARTERIAL_CSV="${VTS_EXPORT_FULL_ARTERIAL_CSV:-false}"
export VTS_TOPOLOGY_AUDIT_MIN_DIAMETER_UM="${VTS_TOPOLOGY_AUDIT_MIN_DIAMETER_UM:-${VTS_FLOW_EXPLICIT_MIN_DIAMETER_UM}}"
export VTS_TERMINAL_PATH_AUDIT_MAX_ROWS="${VTS_TERMINAL_PATH_AUDIT_MAX_ROWS:-250000}"
export VTS_TERMINAL_PATH_AUDIT_INCLUDE_SEGMENTS="${VTS_TERMINAL_PATH_AUDIT_INCLUDE_SEGMENTS:-false}"

{
    echo "run=right_leg_active_xcat_8um_20m"
    echo "started=$(date --iso-8601=seconds)"
    echo "repo=${REPO}"
    echo "input=${INPUT}"
    echo "xcat_nrb_path=${VTS_XCAT_NRB_PATH}"
    echo "xcat_vessel_source=${VTS_XCAT_VESSEL_SOURCE}"
    echo "nrb_domain_voxel_spacing_cm=${VTS_NRB_DOMAIN_VOXEL_SPACING_CM}"
    echo "nrb_domain_samples=${VTS_NRB_DOMAIN_SAMPLES}"
    echo "explicit_terminal_um=${VTS_EXPLICIT_TERMINAL_UM}"
    echo "distal_murray_gamma=${VTS_DISTAL_MURRAY_GAMMA}"
    echo "proximal_murray_gamma=${VTS_PROXIMAL_MURRAY_GAMMA}"
    echo "murray_transition_um=${VTS_MURRAY_TRANSITION_UM}"
    echo "subdivision_max_ld_ratio=${VTS_SUBDIVISION_MAX_LD_RATIO}"
    echo "subdivision_clip_below_um=${VTS_SUBDIVISION_CLIP_BELOW_UM}"
    echo "subdivide_xcat_terminals=${VTS_SUBDIVIDE_XCAT_TERMINALS}"
    echo "export_full_geometry_csv=${VTS_EXPORT_FULL_GEOMETRY_CSV}"
    echo "export_full_arterial_csv=${VTS_EXPORT_FULL_ARTERIAL_CSV}"
    echo "topology_audit_min_diameter_um=${VTS_TOPOLOGY_AUDIT_MIN_DIAMETER_UM}"
    echo "terminal_path_audit_max_rows=${VTS_TERMINAL_PATH_AUDIT_MAX_ROWS}"
    echo "terminal_path_audit_include_segments=${VTS_TERMINAL_PATH_AUDIT_INCLUDE_SEGMENTS}"
    echo "output=${OUT}"
    echo "log=${LOG}"
    echo "julia_num_threads=${JULIA_NUM_THREADS}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "terminal_um=8.0"
    echo "target=20000000"
    echo "frontier_batch=4096"
    echo "graph_block_size=1"
    echo "min_frontier_separation_cm=0.01"
    echo "max_segment_length_cm=0.05"
    echo "graph_neighbors=24"
    echo "max_path_nodes=48"
    echo "graph_jitter_cm=0.005"
    echo "snap_terminal_to_target=true"
    echo "max_terminal_snap_cm=0.15"
    echo "coverage_multiplier=1.10"
    echo "use_indexed_anchor=true"
    echo "use_astar_routing=true"
    echo "frontier_candidate_factor=32"
    echo "fixed_vein_exports=6"
    echo "max_anchor_gap_cm=1.25"
    echo "growth_artery_seeds=0"
    echo "growth_artery_min_length_cm=3.0"
    echo "fixed_artery_min_length_cm=3.0"
    echo "fixed_vein_min_length_cm=8.0"
    echo "route_dilation_voxels=${VTS_ROUTE_DILATION_VOXELS}"
    echo "max_graph_block_size=${VTS_MAX_GRAPH_BLOCK_SIZE}"
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
    echo "min_initial_territory_fraction=${VTS_MIN_INITIAL_TERRITORY_FRACTION}"
    echo "max_initial_territory_points=${VTS_MAX_INITIAL_TERRITORY_POINTS}"
    echo "min_graph_largest_component_fraction=${VTS_MIN_GRAPH_LARGEST_COMPONENT_FRACTION}"
    echo "max_unclaimed_target_fraction=${VTS_MAX_UNCLAIMED_TARGET_FRACTION}"
    echo "checkpoint_dir=${VTS_CHECKPOINT_DIR}"
    echo "checkpoint_interval_seconds=${VTS_CHECKPOINT_INTERVAL_SECONDS}"
    echo "checkpoint_keep=${VTS_CHECKPOINT_KEEP}"
} > "${OUT}/run_metadata.txt"

exec /home/molloi-lab/.juliaup/bin/julia --project=. \
    examples/right_leg_xcat_50um_gpu.jl \
    8.0 \
    "${OUT}" \
    "${VTS_XCAT_NRB_PATH}" \
    20000000 \
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
    3.0 \
    3.0 \
    8.0 \
    > "${LOG}" 2>&1
