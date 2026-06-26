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
export VTS_NRB_DOMAIN_VOXEL_SPACING_CM="${VTS_NRB_DOMAIN_VOXEL_SPACING_CM:-0.1}"
export VTS_NRB_DOMAIN_SAMPLES="${VTS_NRB_DOMAIN_SAMPLES:-48,48}"
export VTS_EXPLICIT_TERMINAL_UM="${VTS_EXPLICIT_TERMINAL_UM:-200.0}"
export VTS_DISTAL_MURRAY_GAMMA="${VTS_DISTAL_MURRAY_GAMMA:-3.0}"
export VTS_PROXIMAL_MURRAY_GAMMA="${VTS_PROXIMAL_MURRAY_GAMMA:-2.0}"
export VTS_MURRAY_TRANSITION_UM="${VTS_MURRAY_TRANSITION_UM:-200.0}"
export VTS_SUBDIVISION_MAX_LD_RATIO="${VTS_SUBDIVISION_MAX_LD_RATIO:-25.0}"
export VTS_SUBDIVISION_CLIP_BELOW_UM="${VTS_SUBDIVISION_CLIP_BELOW_UM:-0.0}"
export VTS_MIN_INITIAL_TERRITORY_FRACTION="${VTS_MIN_INITIAL_TERRITORY_FRACTION:-0.05}"
export VTS_MAX_INITIAL_TERRITORY_POINTS="${VTS_MAX_INITIAL_TERRITORY_POINTS:-4096}"

exec /home/molloi-lab/.juliaup/bin/julia --project=. \
    examples/right_leg_xcat_50um_gpu.jl \
    100.0 \
    "${OUT}" \
    "${VTS_XCAT_NRB_PATH}" \
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
