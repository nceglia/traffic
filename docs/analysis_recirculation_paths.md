# Recirculation reactive-path analysis

Decomposes T-cell trafficking into **recirculation routes** and their **fates** on the fitted
growth operator. Answers: seeded in a tissue, where does a clone's mass go, does it come back
(recirculate) or fade (decay), and which routes carry the traffic?

Scripts (reproduce): `scripts/analysis_recirculation_paths_setup.py` (compute) →
`scripts/analysis_recirculation_paths_visualize.py` (figures). Outputs to
`figures/analysis/recirculation_paths/`.

## What this is (prior art / naming)

There is no established "recirculation tree" for this system, but the construction is a
combination of standard tools:

- **Transition Path Theory (TPT) / reactive-pathway decomposition on a Markov State Model** —
  the closest match. The ranked routes home→…→home are TPT reactive pathways ranked by flux
  (source = sink = the home tissue).
- **Renewal / first-return decomposition** — "stop when you return to the seed tissue" is the
  first-return renewal point; excursions between renewals are i.i.d. (strong Markov property).
- **Absorbing Markov chains / phase-type distributions** — the recirculate-vs-decay fate split is
  an absorbing-chain computation (fundamental matrix `N=(I−Q)⁻¹`).
- **Multitype branching / matrix population models (Leslie/Lefkovitch)** — the growth/death
  (`g≠1`) makes it a birth–death process, the demography/ecology analog.
- The planned repeat roll-up (`(CT)*` → a loop) is **regular-expression / Kleene-star** compression
  and **strongly-connected-component (SCC) condensation** of the walk; "roll up to an infinite
  walk" recovers the stationary Markov *graph* from its unrolling.

## Method

1. `T = decompose(M)` is the conserving redistribution operator. Here we KEEP the growth:
   `G` = the ρ-weighted 3×3 tissue collapse of the growth matrix `M` (row sums `g ≠ 1`).
2. Per-step fates of mass at tissue `t`: **retain** (self-loop `G_tt`), **decay/loss** (`1−g_t`, the
   depth-normalization contraction — not literal death), **recirculate** (return home `G_{t,home}`),
   **continue** (migrate to the other non-home tissue). These four sum to 1.
3. **Recirculate vs decay** for mass leaving a home tissue = an absorbing Markov chain with the two
   non-home tissues transient and {home-return, decay} absorbing. `Q` = the 2×2 transient block;
   `N = (I−Q)⁻¹` the fundamental matrix; `start` = normalized first-hop; **occupation** `v = start·N`
   (expected visits to each non-home tissue per unit leaving mass, self-dwells included).
4. **Exact per-tissue split** (drives the loop figure): returning mass `ret_i = v_i·G_{i,home}`
   (Σ = the recirculate %, exactly — verified to 1e-9), decay `dec_i = v_i·(1−g_i)`, loop flux
   `v_i·G_{i,j}`; `ret + dec` partition the leaving mass into {home, decay}.
5. **Decay decomposition:** `1−g_t` is split into a shared **global contraction** (`1−λ`, `λ` =
   occupancy-weighted mean `g` ≈ 0.66) and the **tissue-specific excess** — separating the universal
   normalization artifact from tissue biology in the fate glyph.
6. **Ranked routes** = reactive paths home→…→home ranked by returning mass; consecutive dwell
   repeats collapsed. Direct bounces (`home→X→home`) vs via-loop (longer) are tagged.

## Key observations (baseline `off`)

- **Recirculation is asymmetric — blood is a leaky feeder, CSF⇄tumor is a tight loop.**
  Of mass leaving each tissue: **Blood 24%** recirculates (leakiest), **CSF 41%** (stickiest),
  **Tumor 29%**. Mass in CSF returns to blood only 19% of the time vs 35% for tumor — CSF is a
  one-way conduit onward to tumor.
- **Dominant routes:** CSF → `C→T→C` (the CSF⇄tumor bounce); Blood → `B→T→B`; Tumor → `T→C→T`.
  The recirculating core is the CSF⇄tumor pair; blood feeds in and drains out.
- **`stay6` makes the CSF–tumor compartment self-sustaining** (ρ(Q) ≥ 1: mass never fully returns
  or decays) for the Blood and Tumor seeds — a diagnostic that the strong persistence prior is too
  strong (corroborates the drainage-calibration finding).

## Caveats (read before interpreting)

- **"Decay" is NOT a literal death rate.** `G` is fit on the depth-normalized scale (source ÷
  `d^src`, destination × `d^dst`), so `g` is a per-sample-depth output multiplier — not absolute
  fecundity, and not comparable across tissues without an anchor. The global
  contraction (λ≈0.66, mean `g`) that drives most of the "decay" is a normalization artifact and cannot be
  separated from real cell death/turnover without an **abundance anchor** (per-sample T-cell
  counts / cellularity / spike-in). What is robust is the RELATIVE structure (CSF one-way, tumor
  bounces back, blood leaky, which routes dominate) — not the absolute recirculate/decay percentages.
- **Tissue-level only.** A phenotype-resolved version is confounded by 18–33% biologically-impossible
  cross-lineage flow in the operator (annotation/doublet artifact) — see `docs/DATA.md`
  ("cross-lineage leak"). Use the no-lineage differentiation view if going phenotype-resolved.
- **Source-tissue attribution** uses raw counts (`data.source_tissue`), not the depth-rescaled
  Xtilde — see `docs/DATA.md` ("Source-tissue attribution").
- **CSF rests on 2–3 informative patients** (DFCI3, DFCI5); MSK1 is a masking artifact. Report CSF
  routes with that `n` in mind (`docs/DATA.md`).
- Numbers are the posterior-mean operator (`off`); the fate table carries posterior CIs across
  configs in `recirc_summary.json`.

## Figures

- **`recirc_paths_trees.png`** — the UNROLLED excursion. Home-rooted chains that fan outward (each
  step a little higher/lower); each node draws a green **return-home arc** (out-and-around, sized by
  the returning mass `m·recirc`) and a per-step **fate glyph** (recirculate / retain / continue /
  decay, decay split into global vs excess). Because the unrolled chain folds out self-loops, the
  drawn arcs are the **no-dwell** returns only (the panel annotates that fraction); dwell-then-return
  paths re-enter in the loop view.
- **`recirc_loops.png`** — the COLLAPSED stationary loop (the roll-up). Home + the two non-home
  tissues as a mini-recirculation loop `A⇄B`, node size ∝ occupation `v`. Green **return arcs** sum
  **exactly** to the recirculate % (via `N`, dwells included); the `A⇄B` back-edges carry the loop
  with its per-cycle survival annotated (weak here: **1.4–3.3%/cycle**); decay stubs stay **dangling**.
  Compartments where `ρ(Q) ≥ 1` (self-sustaining, e.g. `stay6`) are flagged instead of drawn.
- **`recirc_paths_ranked.png`** — ranked reactive routes `home→…→home` by returning mass; direct
  bounces vs via-the-loop tagged by color.

The roll-up (`(AB)*` → loop back-edge + reconnection to the seed) that earlier lived here as a plan
is now `recirc_loops.png`. The mini-loops are real but weak — the CSF⇄tumor "tight loop" is carried
by high **self-retention** (CSF→CSF 0.33, TP→TP 0.46) and the direct exchange, not by many bounce
cycles (see the survival numbers).
