# Migratory vs resident phenotype analysis

Characterises every **(tissue, phenotype)** on two share-supported axes of the fitted operator
`T`, with full posterior uncertainty, to say which phenotypes are **resident** (sticky — stay in
tissue) vs **migratory** (higher propensity to leave), and — among those that stay — whether they
keep their phenotype (**stable**) or switch in place (**plastic**).

Script (single entry point): `scripts/analysis_migratory_vs_resident_phenotype.py`.
Outputs to `figures/analysis/migratory_resident_phenotypes/`.

```
FIT=/Users/ceglian/Data/misc/nb_global_putative.npz \
TRAFFIC_H5=<...>_putative.h5ad \                # optional — enables the calibration figure (E)
OUTDIR=figures/analysis/migratory_resident_phenotypes DRAWS=400 \
python3 scripts/analysis_migratory_vs_resident_phenotype.py
```

State space + phenotype labels are read **from the FIT object**, so a refit on any state space
reruns unchanged. Operator figures (A–D) need only `FIT`; the calibration figure (E) also needs
`TRAFFIC_H5`.

## What this is (prior art / naming)

The model-side, Bayesian counterpart of `gbm_trafficking/pipeline/figure_main2_trafficking.py`,
which classifies each *clone* as RESIDENT/MIGRATORY (did it appear in another tissue next step),
then computes a per-(tissue, phenotype) `Δ = frac_migratory − frac_resident` with a mixed-effects
logistic-GLM significance (patient random intercept, BH-FDR). That is a **clone-mixture,
occupancy-confounded, empirical** construction.

Here the quantities come from the fitted operator `T = decompose(M)[0]` (row-stochastic
redistribution; share-supported, audit-clean — see `traffic/readouts.py` CONTRACT), which resolves
per-*phenotype* leave rates the clone-mixture view cannot. Uncertainty is the posterior over the
saved draws. The primary metric is a per-phenotype **rate** (not the occupancy-confounded Δ), and
the analysis adds the switching/persistence axis to give a 2-D residence characterisation.

## Method

Helper `operator_quantities(T, ss, rho)` returns, per source state `z = (a, u)` = (tissue,
phenotype):

| quantity | definition | meaning |
|---|---|---|
| `go[a,u]` | off-block row mass of `T` = `1 − Σ_v T[(a,u),(a,v)]` | **migratory propensity** — P(leave tissue) |
| `stay` | `1 − go` | residence |
| `persist[a,u]` | `T[z,z] / stay` | **in-place stability** — P(keep phenotype \| stayed) |
| `delta[a,u]` | `P(u\|migratory) − P(u\|resident)`, `rho`-weighted | reference metric (occupancy-**confounded**) |
| `migdest[a,u,b]` | `Σ_v T[(a,u),(b,v))]` over `b≠a`, normalised | destination-tissue split among migrants |

`main()` evaluates these on `DRAWS` posterior draws → posterior mean + 95% CrI for each. Derived:

- **baseline** = occupancy(`rho`)-weighted mean `go` per tissue ("the average cell's leave rate").
- **deviation** `dev = go − baseline` (per draw), the PRIMARY metric.
- **resolved** = the 95% CrI of `dev` excludes 0 (the Bayesian migratory/resident call).

`rho` is the fit's pooled source-phenotype occupancy per tissue. Colors are the canonical
phenotype palette (`gbm_trafficking/pipeline/modules/style.py`, extended to the CD4 split).

## Figures produced

| file | what it shows |
|---|---|
| **`leave_propensity.png`** (PRIMARY) | `go − baseline` per phenotype, one panel per tissue, **sorted resident→migratory**, diverging bars, phenotype-coloured, 95% posterior CrI. **Bold+solid = resolved** (CrI excludes 0), faded = unresolved. **Hollow dot = phenotype-plastic** (`persist<0.5`) when it stays. Directional labels (← resident \| migratory →). |
| **`residence_stability_map.png`** | 2-D map per tissue: x = `go` (migratory propensity), y = `persist\|stay` (stability); each phenotype a point with 2-D posterior CrI. Quadrants **resident-stable** (low go, high persist) / **resident-plastic** (low go, low persist) / **migratory** (high go). Shaded split at the tissue-mean leave rate. |
| **`delta_migratory_resident.png`** | Reference recreation of `figure_main2`: occupancy-weighted `Δ = P(u\|mig) − P(u\|res)` per tissue, phenotype-coloured bars, CrI, `*` where CrI excludes 0. **Occupancy-confounded — kept only for comparison; use `leave_propensity` for the propensity call.** |
| **`migrant_destination.png`** | Where migrants go: per source tissue, the destination-tissue split among migrants for each phenotype (stacked bar, rows ordered by `go`). |
| **`calibration_by_phenotype.png`** (needs `TRAFFIC_H5`) | Correctness check. **A:** per-phenotype observed vs predicted leave-rate (on the diagonal = calibrated; size = n). **B:** per-phenotype leave/stay AUC with patient-cluster 95% CI. |

## Key observations

- **TP Treg is the archetypal resident** — `go≈0.16`, `persist≈0.91`, tightly resolved (stays *and*
  keeps identity).
- **Tumor exhausted/effector states are resident-*plastic*** — TEXeff/TEXterm/TEXprog/4QExh stay but
  switch phenotype in place (exhaustion-in-place).
- **Naive states (Naive, 4Naive) are migratory in every tissue.**
- **TEMRA is tissue-specific:** resident in blood, migratory in CSF/tumor.
- **The model is calibrated and discriminative per phenotype** (calibration fig: on-diagonal;
  AUC 0.78–0.96), so the calls are backed by a validated model.
- **Migrant routing:** blood migrants → predominantly CSF; CSF exhausted/effector → tumor (the
  drainage route).

## Two views (clone-mixture vs phenotype-resolved)

Residence can be a property of the **clone** or of the **phenotype**, and they are different objects
— this is the analysis's central point:

- **Clone-as-mixture** (traditional/empirical, = the `Δ` panel / `figure_main2`): a clone is a
  distribution over phenotypes; "does it move" is a property of the whole mixture. A clone's
  propensity is the mixture average `go_clone(c) = Σ_z θ_c(z)·go[z]`.
- **Phenotype-as-state** (model-enabled, = `leave_propensity`): `go[a,u]` is the leave rate of the
  *phenotype itself*, de-mixed. Only the operator resolves this.

The two can disagree, and the disagreement is informative, not a bug. **TEMRA** is the example:
blood clones that migrate are TEMRA-enriched → "TEMRA is migratory" (clone view / `Δ`), yet a TEMRA
cell's share tends to *stay* (`go=0.45`, below baseline) → "TEMRA is resident" (phenotype view).
Both true: TEMRA-heavy clones move, but it isn't the TEMRA cells doing the leaving. The model
de-mixes what the clone view cannot.

## Caveats (read before interpreting)

1. **Calibration validates a different estimand than the calls.** The calls use operator `go[a,u]`;
   `calibration_by_phenotype` computes predicted `P(leave)` from the per-clone tissue-marginal
   prediction (depth-weighted over the population), a *related but different* quantity. It certifies
   an aggregate proxy, not `go[a,u]`. **(open fix)**
2. **TRM reads migratory** (high `go`, low `persist`), which contradicts "tissue-resident memory" —
   especially TRM-in-blood. Likely a sparse-data / labeling artifact; treat that call as suspect.
   **(open — investigate)**
3. **Patient pooling.** "Resolved" (95% CrI excludes 0) is parameter uncertainty on a
   patient-*pooled* operator — **not** a cross-patient claim. A call can be resolved while driven by
   1–2 patients. The reference used a patient random effect precisely to guard this; per-patient
   resolution needs the `loPatient` / `nb_patient` fits. **(open — biggest gap)**
4. **Share-space, not absolute.** `go`/`persist` are redistribution of depth-normalized *share*
   off-tissue, not absolute cell emigration (compositional closure; see `readouts.py` CONTRACT).
   "Migratory" = "share redistributes off-tissue."
5. **`persist|stay` is conditional and noisy** exactly where the map plots it for high-`go`
   (migratory) phenotypes — little mass stays, so `T[z,z]/stay` is poorly determined there; the two
   map axes are not independent.
6. **Static occupancy.** The baseline and `Δ` use one pooled `rho`; the system is OVI-treated and
   temporal, so occupancy shifts across timepoints and the "average cell" reference is a snapshot.
7. **`migrant_destination` has no CrI**, and low-n phenotypes are not flagged in A/B/D.
8. **Thresholds are arbitrary** (baseline = tissue-mean; `persist>0.5`; resolved = CrI excludes 0) —
   the categorical labels can move; no sensitivity analysis yet.

## Open items / planned fixes

- Per-patient resolution (`loPatient`/`nb_patient`) → replace pooled-CrI significance with a
  patient-consistency test (caveat 3).
- Make `calibration_by_phenotype` validate `go[a,u]` itself, or relabel it as the aggregate proxy
  (caveat 1).
- Investigate the TRM anomaly (caveat 2).
- Add a **clone-level-vs-phenotype-level panel** — `go_clone` (mixture-weighted `go`) beside the
  phenotype-resolved `go`, flagging the phenotypes where they diverge (TEMRA). Needs `TRAFFIC_H5`
  for the clone compositions `θ_c`.
- Add CrI to `migrant_destination`; annotate `n` on A/B/D; threshold-sensitivity pass.
