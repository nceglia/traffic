# Data, sampling structure, and source-tissue attribution

This note documents how the GBM T-cell trafficking data is structured, what is and isn't
observable per patient (the **mask**), and a **source-tissue attribution fix** that affects every
source-conditioned read-out. Read this before interpreting any tissue-conditioned result
(traffic matrices, migration/switching validation, drainage).

The state space is `Z = S x K = 3 tissues x 13 phenotypes = 39` states
(`PBMC, CSF, TP(tumor)` x the T-cell phenotypes). A **clone-step** (row `j`) is one
`(patient, clonotype, source_timepoint)`; its source composition `x_j` spans all 39 states.

---

## 1. Raw sampling grid: TCR+ cells per patient / timepoint / tissue

Cohort: **6 patients** (DFCI1-5, MSK1). Sampling is highly uneven; several
(patient, timepoint, tissue) cells are simply absent. TCR+ cell counts:

| patient | timepoints | PBMC | CSF | TP (tumor) |
|---|---|---|---|---|
| DFCI1 | 3,4 | 1703, 2000 | — | — |
| DFCI2 | 1-6 | 1208 (t1 only) | — | 348,393,430,344,3212,4079 |
| DFCI3 | 1-6 | 940,1890,1393,—,2226,1895 | 37,135,256,—,84,— | 25,—,3424,5806,3764,1915 |
| DFCI4 | 1-6 | 1697,1259,—,1963,588,— | 1,4,—,—,8,— | 164,279,5346,—,3730,7309 |
| DFCI5 | 1-5 | 4968,4383,1789,4216,1995 | —,98,400,—,— | 84,780,652,1579,6333 |
| MSK1 | 1-4 | **never** | 108,821,1036,836 | **2514 (t1 only)** |

Regenerate with `TRAFFIC_H5=/path/to.h5ad python scripts/mask_audit.py`.

Key facts:
- **CSF is desperately sparse.** DFCI1 and DFCI2 have **zero** CSF cells ever; DFCI4 has 1-8
  cells at a couple of timepoints. Only **DFCI3, DFCI5, and MSK1** carry meaningful CSF.
- **PBMC is missing entirely for MSK1** and only at t1 for DFCI2.
- Depth varies ~100x across (patient, timepoint, tissue); tumor is generally sampled far
  deeper than CSF.

---

## 2. Forward-step construction and the destination mask

`traffic.data.build` turns the grid into `[J, L]` arrays. Two rules matter:

1. **Destination is strictly `tp + 1`** (`dtp = tp + 1`), and a step exists **only if the
   patient has any sample at `tp+1`**. Non-consecutive gaps (e.g. a patient with timepoints
   {1,3}) produce **no** step across the gap. (The manuscript describes this as "next valid
   timepoint" but the code uses `tp+1`; this is an implementation detail, not the model.)
2. **The destination-tissue mask is a property of `(patient, tp+1)` only** — it does *not*
   depend on the clone or the source tissue:
   ```
   mask[t] = 1  iff  depth[(patient, tp+1, t)] > 0     # tissue t sampled at the destination time
   ```
   For a masked destination tissue, `Y=0` and `D=0`, so it drops out of the likelihood.

**Consequence:** a clone-step can only be *observed* migrating to tissues that its patient
sampled at `tp+1`. A destination that wasn't collected looks like "stayed / vanished," not
"migrated there."

### Per-patient forward-step masks (`source tp -> tp+1 : observable dest tissues`)

- **DFCI1**: t3->t4 `{PBMC}` (one step; PBMC-only patient)
- **DFCI2**: every step `{TP}` (tumor-only from t2 on)
- **DFCI3**: t1->t2 `{PBMC,CSF}`, t2->t3 `{PBMC,CSF,TP}`, t3->t4 `{TP}`, t4->t5 `{PBMC,CSF,TP}`, t5->t6 `{PBMC,TP}`
- **DFCI4**: t1->t2 `{PBMC,CSF,TP}`, t2->t3 `{TP}`, t3->t4 `{PBMC}`, t4->t5 `{PBMC,CSF,TP}`, t5->t6 `{TP}`
- **DFCI5**: t1->t2 `{PBMC,CSF,TP}`, t2->t3 `{PBMC,CSF,TP}`, t3->t4 `{PBMC,TP}`, t4->t5 `{PBMC,TP}`
- **MSK1**: t1->t2 `{CSF}`, t2->t3 `{CSF}`, t3->t4 `{CSF}`

**`CSF -> TP` (drainage) is observable only in DFCI3, DFCI5, and (n=9) DFCI4.** The CSF-drainage
story rests on effectively **two patients (DFCI3, DFCI5)** — they agree (both drain strongly to
tumor). Report drainage with that `n` in mind.

### MSK1 is a masking trap
MSK1's tumor is sampled **only at t1**, and t1's destination (t2) has **only CSF**. So *every*
MSK1 forward step — CSF-source or tumor-source — can only be observed landing in CSF. MSK1
therefore contributes **zero `->TP` and zero `->PBMC`** observations. Its raw `CSF->CSF` "stay"
of 1.0 is a **missingness artifact, not residence**; it informs only `CSF->CSF` and `TP->CSF`.
Any per-patient "CSF persistence" tally that includes MSK1 (e.g. pooled CSF-stay ~= 0.23) is
inflated by this artifact; the informative-subset CSF-stay (drop MSK1) is ~0.04-0.10.

---

## 3. Source-tissue attribution — the `src_tis` fix

**Problem.** Source-conditioned read-outs label each clone-step with a single "source tissue"
via `argmax` over the tissue blocks. The buggy convention argmaxed the **depth-rescaled**
composition `Xtilde = x / d_src`:
```python
src_tis = Xt.reshape(J, S, K).sum(2).argmax(1)     # WRONG: Xtilde up-weights shallow tissues
```
Dividing by source depth **up-weights shallowly-sampled tissues (notably CSF)**. Small
multi-tissue clones with only a few CSF cells but a deep sequencing floor get their CSF share
inflated and are misassigned to "CSF-source." Measured impact on this cohort:

- CSF-source count: **1919 (Xtilde-argmax) vs 1796 (raw-count argmax)** — 123 steps flip in.
- Those 123 flippers are **76 tumor-dominated + 47 PBMC-dominated** clones, with a **median of
  only 25% of their raw cells in CSF** and ~6 cells total. Being mostly tumor clones that stay
  in tumor, they **inflate apparent `CSF->TP` drainage** and **understate CSF persistence**.

Effect on the observed CSF drainage (informative subset, tumor destination sampled):

| CSF-source definition | CSF stay | CSF->PBMC | CSF->TP |
|---|---|---|---|
| Xtilde-argmax (buggy) | 0.042 | 0.064 | 0.895 |
| raw-count argmax (fix) | 0.071 | 0.082 | 0.847 |
| CSF-pure (>50% raw cells) | 0.083 | 0.068 | 0.849 |
| CSF-pure (>80% raw cells) | 0.102 | 0.077 | 0.821 |

The **direction is robust** (CSF drains overwhelmingly to tumor under every definition), but the
buggy convention overstates the drainage magnitude by a few points and halves the persistence.

**Fix.** Attribution now uses **raw counts** via a single canonical helper:
```python
from traffic import data
src_tis = obs.src_tissue                 # Observations property: raw-count argmax
# or:  data.source_tissue(X, ss)         # module-level; X may be a masked subset, e.g. obs.X[mask]
```
`traffic.data.source_tissue` / `Observations.src_tissue` are the single source of truth. All
source-conditioned scripts were switched to it (`migration_validation*`, `switching_validation*`,
`migration_validation_ppc`, `prior_predictive_persistence`, `predictability_breakdown`,
`validation_figure`, `analysis_migratory_vs_resident_phenotype`). For a stricter label, threshold
on source purity `x_tissue / x_total` instead of argmax.

**Scope of the bug.** It affects only **data-side, source-conditioned summaries** (observed
traffic matrices, per-clone migration/switching labels, the PPC's observed statistic). It does
**not** touch the fitted operator `M`: the likelihood uses the full `Xtilde` composition with no
argmax, so operator-derived read-outs — `decompose(M)`, `tissue_traffic`, the steady-state
recirculation cycle, drainage magnitude from `M` — are **unaffected**.

---

## 4. Caveats to carry into any analysis

- **CSF results = 2-3 informative patients** (DFCI3, DFCI5, partial DFCI4). Patient-clustered
  bootstrap of CSF persistence spans nearly [0.03, 1.0]; pooled p-values that treat ~22k
  clone-steps as independent are **pseudo-replicated** and overstate precision. Use a
  patient cluster-bootstrap (n=6) for any CSF calibration claim.
- **Never read raw per-patient "stay" as biology without checking the mask** (MSK1 above).
- The persistence-prior (`a0_stay`) CSF knob is **not identifiable** from these data; treat it as
  prior-driven. PBMC persistence is under-predicted in every config (a structural gap, not a
  prior one). See `docs/persistence_prior_spec.md`.
