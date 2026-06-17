#!/usr/bin/env bash
set -euo pipefail

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export JULIA_DEPOT_PATH=/media/molloi-lab/2TB4/Artin/julia_depot

RUN_ROOT=/media/molloi-lab/2TB4/Artin/right_leg_corrected_50um_full_20260615_1453
REPO="$RUN_ROOT/repo/VascularTreeSim.jl"
OUT="$RUN_ROOT/output/repair_validation_multiseed_route_tissue_grid4_50um_2000"
RAW="$RUN_ROOT/input/both_legs_xcat_input/both_legs_act_1.raw"
ORGANS="$RUN_ROOT/input/both_legs_xcat_input/organ_ids.txt"
LOG="$RUN_ROOT/input/both_legs_xcat_input/both_legs_log"

mkdir -p "$OUT"

/home/molloi-lab/.juliaup/bin/julia --project="$REPO" \
  "$REPO/examples/right_leg_xcat_50um_gpu.jl" \
  50.0 "$OUT" "$RAW" "$ORGANS" "$LOG" \
  2000 \
  256 \
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
  32 \
  6 \
  1.25 \
  6
