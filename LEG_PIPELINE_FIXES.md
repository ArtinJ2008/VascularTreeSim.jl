# Leg pipeline fixes — subdivision budget + XCAT artery-tip guard

**Author:** Yishu Li · **Date:** 2026-06-23 · **Base commit:** `d919e55` ("validated nrb shift")

Two correctness fixes in the two-stage **growth → subdivision** path of the right-leg
pipeline (`examples/right_leg_xcat_50um_gpu.jl` + `src/growth_tree.jl`). Both were found
by tracing the terminal-count math end-to-end and reading the actual subdivision recursion.

---

## Fix 1 — `subdivision_factor` undercounts → only ~half the requested terminals

**Where:** `examples/right_leg_xcat_50um_gpu.jl`, the `subdivision_factor` line (~line 761).

**Problem.** The growth-branch budget back-computes how many 200 µm growth branches to
grow by dividing the requested final terminal count by `subdivision_factor`, which used the
**continuous** Murray estimate:

```julia
subdivision_factor = ceil(Int, (growth_terminal_cm / final_terminal_cm)^3)   # (200/8)^3 = 25^3 = 15625
```

But the actual post-growth subdivision (`_subdivide_recursive!` in `growth_tree.jl`) is a
**symmetric binary bifurcation** (`d_child = d_parent / 2^(1/γ)`) that returns as soon as the
next child would be `<= target`. From 200 µm to 8 µm that is only **13 whole levels**, so each
growth terminal yields `2^13 = 8192` leaves — **not** 15625.

The planner and the actual subdivision therefore used two different, unreconciled multipliers,
and nothing checked them against each other. End result for the 8 µm / 20 M run:

```
target_branches = ceil(20,000,000 / 15625) = 1280
final           = 1280 × 8192 ≈ 10.5 M   ≈ 52% of the requested 20 M
```

Every explicit-target run silently produced ~half the intended resolution.

**Fix.** Make `subdivision_factor` equal the *actual* binary leaf count:

```julia
subdivision_factor = final_terminal_cm < growth_terminal_cm ?
    2 ^ max(0, ceil(Int, 3 * log2(growth_terminal_cm / final_terminal_cm)) - 1) : 1
```

`3 * log2(...)` matches γ = 3 (the `^3` it replaces and the `gamma=3.0` used by the subdivision
call). The `- 1` accounts for the recursion stopping one level early (next child `<= target`),
which is also why the 50 µm case realizes 32 leaves rather than 64.

**Verification (factor → branches → final):**

| growth/final | old factor | old final (% of 20 M) | new factor | new final (% of 20 M) |
|---|---|---|---|---|
| 200/8  = 25 | 15625 | 10,485,760 (52%) | **8192** | 20,004,864 (**100%**) |
| 200/50 = 4  | 64    | 10,000,000 (50%) | **32**   | 20,000,000 (**100%**) |

**Note.** Domain clipping during subdivision still reduces the final count somewhat below the
target (out-of-domain sub-branches are pruned); that residual is data-dependent and unavoidable
for a static planner. This fix removes the systematic ~2× shortfall.

---

## Fix 2 — `subdivide_terminals!` subdivides real XCAT artery tips into fake capillary fans

**Where:** `src/growth_tree.jl` (`subdivide_terminals!`) + `examples/right_leg_xcat_50um_gpu.jl`
(the subdivision call).

**Problem.** `subdivide_terminals!` collects **every** leaf via
`_branch_terminals(tree) = [i for i in vertices if isempty(children[i])]` and decides whether to
subdivide **purely from the incoming segment's diameter** — there is no `is_xcat` check:

```julia
for tip_v in terminals
    seg = tree.incoming_segment[tip_v]
    seg == 0 && continue
    d_cm = tree.segment_diameter_cm[seg]
    d_cm <= target_diameter_cm && continue   # only filter is diameter
    ...
```

A distal tip of an XCAT arterial seed (e.g. the femoral distal end) is a leaf that keeps its
measured **mm-scale** diameter when growth did not anchor a child there. Since mm ≫ target, it is
subdivided like a grown terminal: a ~2–4 mm tip is recursively bifurcated down to the target
(~24–27 levels), sprouting a large self-similar symmetric fan (with random-rotation geometry) off
a major artery — physiologically meaningless. (Domain clipping prunes out-of-domain children, so
this is a geometry/realism bug rather than a guaranteed OOM, but it is still wrong.)

**Fix (opt-in, so the shared heart pipeline is unaffected).** Add a `skip_xcat` flag (default
`false`) to `subdivide_terminals!` and skip XCAT seed terminals when it is on:

```julia
# signature
skip_xcat::Bool=false,
# in the loop, right after `seg == 0 && continue`
skip_xcat && tree.is_xcat[seg] && continue
```

and turn it on only at the leg call:

```julia
subdivide_terminals!(trees[name]; ..., skip_xcat=true, domain=route_domain)
```

With this, only grown terminals are subdivided; real XCAT artery tips are left as-is (their
downstream perfusion comes from branches grown off their sides, not from a fan at the tip).
The heart pipeline (`run_coronary_growth.jl`) does not pass the flag → default `false` → unchanged.

---

## Still open (from the same review, not addressed here)

- **`auto` target branch** does not divide by `subdivision_factor`, and the coverage guard does
  not catch the post-subdivision blow-up → potential OOM on a low-diameter `auto` run.
- **`_recompute_all_murray!`** uses `max(existing, murray_d)`, which keeps oversized stale
  diameters for clipped boundary segments (Murray violation near the domain edge).
- **`restore_xcat_seed_diameters!`** unconditionally resets XCAT diameters after the Murray
  recompute, undoing the anti-bottleneck `max()` and reintroducing distal bottlenecks
  (dominates Poiseuille resistance, R ∝ 1/r⁴).
- **`flow_export.jl`**: degree-2 chain collapse drops interior-pruned beds (subtree-terminal
  undercount / non-conservation); `*_resistance_rel` columns hold absolute resistance; viscosity
  hardcoded 0.035 (ignores `VTS_BLOOD_VISCOSITY_POISE`).
- **demand weights** (`VTS_TARGET_DEMAND_MODE=weighted`) are parsed but never used — sampling is
  always uniform (skin/fat perfused like muscle).
