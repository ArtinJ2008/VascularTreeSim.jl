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

{
    echo "run=right_leg_active_xcat_30um_full"
    echo "started=$(date --iso-8601=seconds)"
    echo "repo=${REPO}"
    echo "input=${INPUT}"
    echo "output=${OUT}"
    echo "log=${LOG}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "terminal_um=30.0"
    echo "target=auto"
    echo "frontier_batch=4096"
    echo "graph_block_size=1"
    echo "max_segment_length_cm=0.05"
    echo "route_dilation_voxels=0"
    echo "main_xcat_artery_seeds=6"
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
    6 \
    > "${LOG}" 2>&1
