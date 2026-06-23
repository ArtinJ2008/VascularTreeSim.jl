# VascularTreeSim.jl

Vascular tree generation by competitive constrained-constructive optimization
(CCO) in anatomical voxel domains. Input is either an XCAT NURBS phantom file
(`.nrb`) or a synthetic geometry; output is a per-tree segment CSV and an
interactive 3-D HTML viewer.

The package is **organ-agnostic** — every organ-specific value (surface names,
root anchors, root diameters, target flows) lives in a `.toml` config. No
LAD/LCX/RCA strings are hardcoded in the source; the same code grows brain or
skeletal-muscle trees by editing the config alone.

---

## Pipeline

```
NRB phantom + config.toml
        │
        ▼
┌─────────────────────────────┐
│ 1. Parse NRB → flood-fill   │
│    voxel-shell organ domain │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ 2. Init trees from XCAT     │
│    centerlines (or seeds)   │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ 3. Competitive round-robin  │
│    growth with weighted-    │
│    Voronoi territory + per- │
│    tree p95 stall detection │
└─────────────────────────────┘
        │
        ▼
┌─────────────────────────────┐
│ 4. Recursive bifurcation    │
│    of every 200 μm tip down │
│    to 6 μm capillaries      │
└─────────────────────────────┘
        │
        ▼
  <tree>_segments.csv  + viewer.html
        │
        ▼  (optional, post-growth)
┌─────────────────────────────┐
│ 5. scale_to_rest.jl: apply  │
│    diameter-dependent       │
│    arteriolar tone to get   │
│    the at-rest geometry     │
└─────────────────────────────┘
        │
        ▼
  <tree>_segments.csv (at-rest)
```

The downstream flow simulator
([FlowContrastSim.jl](../FlowContrastSim.jl)) consumes the CSVs directly.

---

## Key concepts

**Murray's law on every segment.** Both grown (round-robin) and subdivided
(recursive bifurcation) segments take their diameter from
`d = d_term · N^(1/γ)` where `N` is the count of capillary terminals in the
subtree below and γ = 3. XCAT-derived ostia keep `max(NRB-measured, Murray)`
so a clinically realistic root (e.g. 3.7 mm for LAD) survives even when the
subtree is small.

**Weighted-Voronoi territory.** Each tree claims tissue up to
`(d_i / d_j) ×` farther than a competitor with smaller root diameter,
producing territory volume ∝ d³ (consistent with Q ∝ d³). Without weights
all trees share territory by pure geometric proximity.

**Per-tree p95 saturation stop.** Growth halts independently per tree when
its own owned-points p95 distance stalls — a fast-territory tree won't keep
stretching into already-covered tissue while a slow-territory tree is still
genuinely improving.

**Domain-clipped subdivision.** When recursive bifurcation would put a
sub-arteriole tip outside the myocardial shell, up to 8 random plane rotations
are tried so the tree stays inside the organ. Clipping only applies below
`subdivision_clip_below_diameter_cm` (default 50 μm) because large epicardial
vessels naturally sit on or outside the wall.

**L/d safety net.** After Murray re-recompute, any residual segment with
length / diameter > `subdivision_max_ld_ratio` (default 25) is split into
equal-length pieces at the same diameter. Preserves Murray (d unchanged) and
keeps Poiseuille resistance physically reasonable.

**At-rest tree (`scripts/scale_to_rest.jl`).** Murray-optimal diameters
correspond to maximum vasodilation. To produce the at-rest state, apply
a hard diameter band — Wong & Molloi 2008 (*Phys Med Biol* 53:3995)
empirically dilate every arteriole in `[8, 400]` μm by a factor of 1.6
under max hyperemia. Reversing the factor gives the at-rest geometry:

```
d_at_rest = d_max_dilated × (1 − tone)   if D_LOW ≤ d_max_dilated ≤ D_HIGH
          = d_max_dilated                otherwise
```

Defaults: `tone = 0.375` (= 1 − 1/1.6 → 1.6× reserve), `D_LOW = 8 μm`
(Kassab pre-capillary), `D_HIGH = 400 μm` (resistance/conductance boundary).
Capillaries (no smooth muscle, < 8 μm) and conduits (> 400 μm) are unchanged.

The flow simulator treats the two CSV sets as the same anatomy in two
different smooth-muscle states. The Wong-Molloi paper also describes a
Cornelissen 2000 passive pressure-diameter curve (eq. 12) that adds local-
pressure dependence on top of the tone factor; not implemented here — would
require iterative coupling with the flow solver.

---

## Installation

Requires Julia ≥ 1.9. Not registered in General; develop from a local clone:

```julia
using Pkg
Pkg.develop(path="/path/to/VascularTreeSim.jl")
Pkg.instantiate()
```

GPU acceleration is via a `CUDA` weak dependency: `using CUDA, VascularTreeSim`
loads the GPU kernels; pure-CPU runs need only the base package.

---

## Quick start — coronary tree from XCAT

```bash
julia --project=. --threads=auto examples/run_coronary_growth.jl \
      configs/coronary.toml
```

For the vmale50 phantom this produces, in `output/`:
- `lad_segments.csv` (~28 GB, ~121 M segments, root 3.7 mm → 6 μm capillaries)
- `lcx_segments.csv` (~25 GB, ~110 M segments, root 3.4 mm → 6 μm)
- `rca_segments.csv` (~29 GB, ~127 M segments, root 3.9 mm → 6 μm)
- `xcat_coronary_viewer.html`
- ancillary point clouds (`domain_points.csv`, `chambers_points.csv`, …)

Full run on a 256 GB / RTX 4090 workstation: **~6 hours** dominated by the
6 μm subdivision pass.

To produce the at-rest tree (used by FlowContrastSim's baseline config):

```bash
for tree in lad lcx rca; do
  julia --project=. scripts/scale_to_rest.jl \
        output/${tree}_segments.csv \
        output_at_rest/${tree}_segments.csv
        # defaults: tone=0.375  d_low_um=8.0  d_high_um=400.0
        # (= Wong-Molloi 2008 spec: 1.6× reserve in [8, 400] μm band)
done
```

Each tree takes ~2.5 min (IO-bound, ~840 k rows/s).

---

## Synthetic / non-XCAT use

For a new organ without an XCAT phantom, use the seed-point mode:

```julia
using VascularTreeSim, StaticArrays
tree = growth_tree_from_seed("Brain_MCA", SVector(5.0, 5.0, 5.0))
# … build a VoxelShellDomain manually …
graph, _, stats = grow_trees_mcp!(Dict("Brain_MCA" => tree), domain; …)
```

`examples/synthetic_cube.jl` is a runnable demo on a cube-shell domain.
`examples/check_domain_only.jl` builds and visualizes the domain alone — useful
for sanity-checking `[surfaces]`/`[domain]` config changes without paying the
~6 h growth cost.

---

## OrganConfig TOML schema

```toml
[organ]
name = "coronary"
nrb_path = "/path/to/vmale50_heart.nrb"
phantom_path = "/path/to/vmale50_1600x1400x500_8bit_little_endian_act_1.raw"
phantom_dims = [1600, 1400, 500]
coordinate_scale = 0.1     # NRB mm → cm
embed_phantom_raw = false  # set true to also write a .raw with grown trees
                           # embedded — costs +5 min and +1 GB

[surfaces]
outer = "dias_pericardium"
reference = "dias_aorta"
cavities = [ "dias_lv_0", "dias_lv_1", … ]    # chambers + tubular cavities
                                              # (subtracted from organ shell)

[[vessel_trees]]
name = "LAD"
surface_names = ["dias_lad1", "dias_lad2", "dias_lad3"]
root_anchor_surface = "dias_aorta"
root_diameter_override_cm = 0.37    # Dodge 1992 anatomical prior, applied
                                    # AFTER growth and BEFORE subdivide
# target_flow_ml_min = 242          # optional; weights per-round batch size
# color = "#1f77ff"

[[vessel_trees]]
name = "LCX"
…

[domain]
voxel_spacing_cm = 0.02           # fine resolution (v5)
outer_samples = [96, 72]          # u/v NURBS sample grid for outer surface
cavity_samples = [56, 40]
dilation_radius = 1               # voxels
coarse_seed_cm = [14.085, 21.628, 26.966]   # midwall seed for flood-fill

[growth]
mode = "continue_from_xcat"       # or "seed_point"
effective_supply_radius_cm = 0.00125     # 12.5 μm — frontier suppression radius
terminal_diameter_cm = 0.02              # 200 μm — Murray-strict terminal during growth
subdivision_terminal_diameter_cm = 0.0006  # 6 μm — capillary target after subdivision
subdivision_clip_below_diameter_cm = 0.005 # 50 μm — only clip thin sub-branches
subdivision_max_ld_ratio = 25.0
max_new_branches_per_tree = 200000        # cap; real stop is Murray budget / p95 stall
graph_neighbors = 16
min_frontier_separation_cm = 0.015
max_path_nodes = 20
frontier_batch = 28                        # per-round frontier targets per tree
murray_gamma = 3.0
max_segment_length_cm = 0.1
smooth_passes = 20                         # Laplacian smoothing of paths
spline_density = 5
coverage_stride = 4                        # block size for coverage points
graph_stride = 12                          # block size for graph points
graph_jitter_cm = 0.08                     # break grid alignment
turn_penalty = 0.5                         # Dijkstra cost of changing direction
target_p95_distance_cm = 0.005             # stop when this is met
target_max_distance_cm = 0.01
hydraulic_cost_weight = 0.0                # 0 keeps pure coverage ordering
hydraulic_rank_candidate_factor = 0        # >0 ranks extra frontier candidates by flow-aware score
hydraulic_reference_length_cm = 1.0        # resistance normalization length
hydraulic_existing_path_weight = 0.0       # include upstream path in soft cost
min_hydraulic_score_cm = 0.0               # reject if coverage benefit/cost is below this
max_new_branch_resistance_rel = inf        # optional hard cap vs reference vessel
max_terminal_path_resistance_rel = inf     # optional hard cap including upstream path
blood_viscosity_poise = 0.035              # 3.5 cP
```

`[seed_points]` is only consulted when `growth.mode = "seed_point"`.

---

## CSV output format

Each segment is one row:

| column | unit | meaning |
|---|---|---|
| `branch` | — | tree name (e.g. LAD) |
| `segment_id` | int | unique within tree |
| `parent_segment_id` | int | id of the segment ending at this segment's start vertex (0 for root) — **the authoritative topology source** |
| `x1_cm, y1_cm, z1_cm` | cm | start coords |
| `x2_cm, y2_cm, z2_cm` | cm | end coords |
| `xmid_cm, ymid_cm, zmid_cm` | cm | midpoint (convenience) |
| `length_mm` | mm | Euclidean length |
| `diameter_um` | μm | Murray + (XCAT max-rule for is_xcat segments) |
| `label` | — | `dias_lad1` etc. for XCAT, `grown` for round-robin, `subdivided` for recursive-bifurcation |

A downstream consumer should rebuild the tree from `parent_segment_id`, **not**
from coordinate coincidence. Dense subdivided trees pack vertices tightly
enough (~100 M end-points in 60 g of myocardium) that 10 nm coordinate
rounding can collapse unrelated endpoints; FlowContrastSim handles this by
following the parent-id chain.

Flow audit CSVs report both raw `generation` and `branchpoint_generation`.
`generation` counts every geometry row, including degree-2 polyline samples
used to trace a curved vessel. `branchpoint_generation` increments only at
actual branchpoints, so it is the safer number for physiological depth checks.

---

## Testing

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

`test/` exercises the cube-shell, sphere-shell, and cylinder seed-point
workflows on synthetic domains. The XCAT-coronary pipeline is too expensive
to run in unit tests.

`examples/benchmark_gpu.jl` reports CPU vs GPU growth wall-time on a cube
shell.

---

## API quick reference

| function | purpose |
|---|---|
| `load_organ_config(path)` | parse TOML → `OrganConfig` |
| `parse_xcat_nrb(path)` | parse `.nrb` → `Vector{XCATNurbsSurface}` |
| `xcat_centerline_from_surface(surf)` | NURBS surface → centerline tree |
| `build_vessel_trees(centerlines, config)` | wire centerlines into trees per config |
| `build_voxel_shell_domain_floodfill(outer, cavities; …)` | NRB surfaces → `VoxelShellDomain` |
| `growth_tree_from_xcat(name, xcat_tree; terminal_diameter_cm)` | XCAT centerline → `GrowthTree` |
| `growth_tree_from_seed(name, point; terminal_diameter_cm)` | seed point → empty `GrowthTree` |
| `grow_trees_mcp!(trees, domain; …)` | competitive round-robin growth |
| `subdivide_terminals!(tree; target_diameter_cm, …)` | recursive bifurcation to capillaries |
| `write_growth_csv(path, branch, tree)` | export to CSV |
| `growth_viewer_html(path, domain, trees, stats, color_map; …)` | interactive 3-D viewer |
| `run_growth(config; output_dir)` | end-to-end pipeline (used by `examples/run_coronary_growth.jl`) |

See `src/VascularTreeSim.jl` for the full export list.

---

## Reproducing the canonical vmale50 coronary run

1. `julia --project=. --threads=auto examples/run_coronary_growth.jl configs/coronary.toml` (~6 h)
2. Three trees in `output/{lad,lcx,rca}_segments.csv`. Topology is deterministic given the same config + NRB; subdivision uses a name-keyed RNG so the recursive bifurcation orientations are also stable.
3. (Optional) `julia scripts/scale_to_rest.jl …` for each tree to produce `output_at_rest/`.
4. Hand the two CSV directories to `FlowContrastSim.jl/scripts/natural_flow_summary.jl` with `configs/coronary_baseline.toml` and `configs/coronary_hyperemic.toml`.

Expected flow numbers (Wong-Molloi 2008 alignment: scale_to_rest tone =
0.375, band [8, 400] μm, 1.6× reserve; baseline cap_R = 0.15 + hyperemic
cap_R = 0.12 mmHg·min/mL/100g for the literature plus a small autoregulatory
cap-bed relaxation):

| tree | baseline (mL/min) | hyperemic (mL/min) | CFR |
|---|---|---|---|
| LAD | 58.4 | 184.0 | 3.15× |
| LCX | 64.9 | 199.9 | 3.08× |
| RCA | 71.5 | 220.5 | 3.08× |
| **total** | **195** | **604** | — |

Per-tree baseline numbers slightly above the 30-60 mL/min textbook range
because the strict Wong-Molloi 1.6× reserve was empirically calibrated
to Pantely 1984 / Fearon 2004 swine data, not to our tree's exact total
resistance. CFR 3.0-3.2× sits squarely in clinical 3-5× range. Total
flow ~195 (rest) / 604 (max hyperemia) match population means.

For a closed-loop autoregulation simulation (arterioles dilate up to
1.6× to maintain target flow as stenosis grows), see FlowContrastSim's
`scripts/lad_stenosis_autoreg_sweep.jl`.

---

## Phantom voxelization scripts

For the perfusion-pipeline (CT virtual scans, real-patient AIF, etc.),
trees need to be rasterized into the XCAT digital phantom. Three companion
scripts live under `scripts/`:

| script | role |
|---|---|
| `voxelize_csv_into_phantom.jl` | no-contrast pass: sub-voxel Monte Carlo to accumulate per-voxel f_blood + UInt16 100-bin volume-fraction quantization. Replaces myocardium voxels (labels 15-18) with blood mixture labels 256-355. Pre-step before contrast voxelization |
| `apply_contrast_at_peak.jl` | peak-iodine pass: reads `peak_iodine.f32` per tree (from `FlowContrastSim/scripts/extract_peak_iodine.jl`) and writes a UInt16 cross-product (100 blood bins × 101 iodine bins) phantom for BasisSim. Single-time-point — for dynamic, see below |
| `voxelize_dynamic_contrast.jl` | 4D (X, Y, Z, T) dynamic-contrast pass: reads per-segment arrival_time.f32 + sub-voxel MC + accumulates f_blood (time-invariant) + f_iodine_w[V, ti] (T frames). Output: one Float32 .raw per frame (1600×1400×500 × 4 B = 4.48 GB each) at iodine mg/mL of blood phase, plus a clipped Float32 f_blood file and a manifest TOML. Uses per-thread Gaussian-dispersed gamma sampling at the per-segment arrival; takes T_END_S and N_FRAMES as CLI args |

The `voxelize_dynamic_contrast.jl` output (Float32 frames) is the closest
analog of the 4D `(x,y,z,t)` iodine concentration array a downstream
consumer would want; the perfusion-pipeline `make_basissim_phantoms.jl`
sibling instead encodes one specific time-point's voxelization as UInt16
cross-product labels for BasisSim physics simulation.
