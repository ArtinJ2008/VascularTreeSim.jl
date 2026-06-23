#!/usr/bin/env bash
set -euo pipefail

WORK_ROOT="/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453"
REPO="${WORK_ROOT}/repo/VascularTreeSim.jl"
INPUT="${WORK_ROOT}/input/both_legs_xcat_input"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="${WORK_ROOT}/output/right_leg_multiseed_100um_validation_${STAMP}"
LOG_DIR="${WORK_ROOT}/logs"
LOG="${LOG_DIR}/right_leg_multiseed_100um_validation_${STAMP}.log"

mkdir -p "${OUT}" "${LOG_DIR}"
cd "${REPO}"

export JULIA_DEPOT_PATH="/media/molloi-lab/2TB4/Artin/julia_depot"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export VTS_XCAT_NRB_PATH="${VTS_XCAT_NRB_PATH:-${INPUT}/both_legs_1.nrb}"
export VTS_XCAT_VESSEL_SOURCE="${VTS_XCAT_VESSEL_SOURCE:-nrb}"

exec /home/molloi-lab/.juliaup/bin/julia --project=. \
    examples/right_leg_xcat_50um_gpu.jl \
    100.0 \
    "${OUT}" \
    "${INPUT}/both_legs_act_1.raw" \
    "${INPUT}/organ_ids.txt" \
    "${INPUT}/both_legs_log" \
    1000 \
    64 \
    4 \
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
    16 \
    6 \
    1.25 \
    6 \
    > "${LOG}" 2>&1
