# Leg pipeline fixes — subdivision, Murray, flow-export, domain

**Author:** Yishu Li · **Date:** 2026-06-23 · **Base commit:** `d919e55` ("validated nrb shift")

A batch of correctness fixes to the right-leg pipeline (two-stage NRB **growth → subdivision**),
found by a systematic review of the terminal-count math, the Murray recompute, the seed-diameter
restore, and the flow-export accounting.

> **Status:** logic- and arithmetic-verified (+ `git diff`). **NOT yet compiled/run in Julia** —
> please `julia`-compile before a production run.

---

## 1. `subdivision_factor` undercounts → only ~half the requested terminals
**File:** `examples/right_leg_xcat_50um_gpu.jl`

The growth-branch budget divided the requested final terminal count by the **continuous**
estimate `ceil((growth/final)^3)` (= 25³ = 15625 for 200→8 µm). But the actual subdivision is a
**symmetric binary** bifurcation that yields `2^13 = 8192` leaves per terminal. So `target=20M`
produced ~10.5 M (≈52%). 50 µm runs were exactly 50%.

**Fix:** `subdivision_factor = 2^max(0, ceil(3*log2(growth/final)) - 1)` (the real binary leaf
count). Verified: 8 µm and 50 µm runs now reach ~100% of the requested count.

## 2. `subdivide_terminals!` subdivided real XCAT artery tips into fake capillary fans
**Files:** `src/growth_tree.jl`, `examples/right_leg_xcat_50um_gpu.jl`

The terminal loop decided to subdivide purely from diameter (no `is_xcat` check), so a mm-scale
XCAT artery distal tip (e.g. femoral end) that growth never anchored a child to was bifurcated all
the way to capillary scale — a physiologically meaningless self-similar fan off a major artery.

**Fix:** opt-in `skip_xcat::Bool=false` kwarg + `skip_xcat && tree.is_xcat[seg] && continue`; the
leg call passes `skip_xcat=true`. Default `false` keeps the coronary pipeline unchanged.

## 3. `restore_xcat_seed_diameters!` re-introduced a distal bottleneck
**File:** `examples/thigh_xcat_femoral_100um.jl`

`_recompute_all_murray!` widens a thin distal XCAT segment that now feeds a large grown subtree
(its documented anti-bottleneck `max()`). But this function then **unconditionally** reset every
XCAT segment to its raw NRB diameter, undoing that widening — a thin distal segment becomes a
bottleneck that dominates path resistance (R ∝ 1/r⁴).

**Fix:** `segment_diameter_cm[s] = max(current_post_Murray, measured)` — keep the larger of the
measured anatomy and the Murray demand. Affects all leg/thigh callers; the coronary pipeline does
not call it.

## 4. `_recompute_all_murray!` over-sized domain-clipped grown segments
**File:** `src/growth_tree.jl`

`max(existing, murray_d)` kept the **stale top-down creation diameter** for grown/subdivided
segments after domain clipping reduced their surviving terminal count → boundary vessels too thick
(Murray violation). The code comment already stated `murray_d` is authoritative for grown segments.

**Fix:** grown segments (`is_xcat=false`) use `murray_d` directly; only XCAT segments take the
`max()` (to preserve measured anatomy that may exceed the Murray demand).
**Note:** `_recompute_all_murray!` is shared with the coronary pipeline, so this also corrects
grown-segment sizing there — worth a sanity check on a coronary run.

## 5. `auto` target + low terminal diameter could OOM
**File:** `examples/right_leg_xcat_50um_gpu.jl`

The safety guard checked `coverage_count` only. The `auto` branch sizes `target_branches` to the
**growth** stage, so subdivision could then materialize hundreds of millions of segments
unchecked (e.g. an auto 8 µm run ≈ 5×10⁸ segments) while the guard saw a small coverage count.

**Fix:** also guard `projected_final_terminals = target_branches * subdivision_factor` against the
same limit. The explicit 20 M run passes; an auto low-diameter run is now rejected cleanly (use
`VTS_ALLOW_HUGE_COVERAGE`).

## 6. flow-export terminal-count non-conservation
**File:** `src/flow_export.jl`

The degree-2 chain collapse dropped sub-threshold (pruned) beds hanging off **interior** collapsed
vertices: `pruned_children` and `subtree_terminals` were read only at the chain end, so terminals
behind interior-pruned branches were lost (sum of children < parent).

**Fix:** accumulate pruned children at **every** vertex along the conduit; report
`subtree_terminals` at the conduit **entry** (`segment_end[first_seg]`), which includes all
downstream terminals → flow accounting conserves.

## 7. flow-export resistance columns mislabeled + wrong viscosity
**Files:** `src/flow_export.jl`, `examples/right_leg_xcat_50um_gpu.jl`

- Columns named `*_resistance_rel` actually held **absolute** Poiseuille resistance → renamed
  `*_resistance_abs`. **Heads up:** any downstream reader keying on the old column name must update.
- Flow-export calls didn't pass a viscosity, so they silently used `0.035 P` even when
  `VTS_BLOOD_VISCOSITY_POISE` was overridden → now thread `blood_viscosity_poise` into all five
  flow-export calls.

---

## Config change
- `scripts/launch_right_leg_active_xcat_8um_20m.sh` now defaults to
  `VTS_TARGET_TISSUE_MODE=muscle` (was `soft`). The growth/coverage domain is the right-leg
  **muscle** (union of `musc*` surfaces), not the whole soft-tissue (skin envelope) interior —
  the physiologically intended target. This also makes the currently-unused demand-weighting
  machinery irrelevant for the muscle-only run.

## Known / not addressed
- **Domain clipping** still reduces the realized terminal count somewhat below target
  (data-dependent; unavoidable for a static planner).
- The **demand-weighted sampler** (`random_points_in_mask_by_demand`) remains unused; only relevant
  if you run `VTS_TARGET_TISSUE_MODE=soft`.
- These changes are **logic-verified but not yet compiled/run in Julia.**
