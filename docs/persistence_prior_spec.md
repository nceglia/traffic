# Persistence-aware operator prior (minimal spec)

## Problem

The operator prior is iid `M_{zz'} ~ Gamma(a0=1, b0=1)`. Normalizing a row of iid Gammas gives a
**uniform Dirichlet** over the `L = S·K = 39` destination states, so a priori

```
E[ P(descendant stays in source tissue) ] = K / L = 13/39 = 1/3
```

i.e. the prior expects a clone to **scatter uniformly** (⅓ stay, ⅔ leave). Biologically clones
persist. This is a strong **anti-persistence** prior, and it biases the fit: the model
**over-mobilizes** — it under-predicts within-tissue persistence and over-predicts efflux.

Confirmed by a prior-predictive check + the source-conditioned traffic PPC (data-weighted persistence):

| tissue | prior-predictive | empirical (data) | posterior (fit) | shrinkage (emp−post) |
|--------|------------------|------------------|-----------------|----------------------|
| PBMC   | 0.52 | 0.88 | 0.75 | +0.13 |
| CSF    | 0.09 | 0.27 | 0.19 | +0.08 |
| TP     | 0.50 | 0.86 | 0.76 | +0.10 |

The posterior is dragged 0.08–0.13 **below the data in every tissue** — prior-driven under-persistence.
(The source-conditioned PPC flags this as p=0.00 for CSF→CSF and TP→TP; the pooled marginal PPC hides it.)

## The change (prior only — same model everywhere else)

Break the prior's exchangeability across destinations: give the **within-tissue ("stay") block** a
larger Gamma shape than the **cross-tissue ("go")** entries. State index `z = tissue·K + phenotype`,
so tissue `a`'s within-block is rows/cols `[a·K : (a+1)·K]`.

```
a0_mat[z, z'] = a0_stay(tissue(z))   if tissue(z') == tissue(z)   # stay
             = a0                     otherwise                    # go
M_{zz'} ~ Gamma(a0_mat[z,z'], b0)
```

Nothing else changes: same NB2 likelihood, same depth exposure `d`, same global/tissue/patient
dispersion, same `L=39` joint tissue×phenotype state space, same NUTS inference. `M = diag(g)T` still
holds; we've only made the prior favour the diagonal blocks.

### Induced prior and calibration

Per row the transition is `Dirichlet` with concentration `a0_stay` on the `K` within-tissue states and
`a0` on the `(S−1)K` cross-tissue states, so

```
prior E[stay in source tissue] = a0_stay / (a0_stay + (S−1)·a0)
```

| `a0_stay` (with a0=1) | prior E[stay] |
|---|---|
| 1 (off) | 0.33 |
| 2 | 0.50 |
| 4 | 0.67 |
| 6 | 0.75 |
| 8 | 0.80 |

For **well-sampled tissues (PBMC, TP)** the likelihood dominates, so any non-anti-persistence prior
lets them reach the data (~0.86–0.88); the boost mainly stops the shrinkage. For **CSF** (sparse,
*transit* compartment, true persistence ≈0.27) the prior center matters — a strong global boost would
**overshoot** CSF. Hence the per-tissue knob.

## Modes / how to run both ways

`PriorConfig.a0_stay`:

- `None` → **off**: `a0` everywhere → reproduces the original uniform prior (baseline).
- `float` → **global**: all three within-tissue blocks share one stay-shape.
- `(S,)` tuple → **per source tissue** (PBMC, CSF, TP), e.g. `(6.0, 2.0, 6.0)` to let CSF stay low.

```python
PriorConfig()                          # baseline (unchanged)
PriorConfig(a0_stay=6.0)               # global persistence boost  (prior stay 0.75)
PriorConfig(a0_stay=(6.0, 2.0, 6.0))   # per-tissue: PBMC/TP high, CSF (transit) low
```

Implemented in `traffic/mcmc.py::a0_matrix` (built once in `fit_nuts`); a length mismatch raises.

## Recommended refit sweep

Refit the full data + the 14 holdouts under a small grid so we can pick by held-out score, not by eye:
`a0_stay ∈ {None (baseline), 4, 6}` global, plus one per-tissue `(6, 2, 6)`. Keep `b0=1`, NB global
dispersion (validated), x64.

## Validation after refit

Re-run the migration suite (esp. `ppc_traffic_matrix` and `prior_predictive_persistence`). Success =
CSF→CSF / TP→TP source-conditioned PPC p-values rise off 0.00 and the persistence diagonal moves up to
the empirical, **without** breaking the other axes (stay/go AUC, switching, the flat PIT/NB-dispersion
check) or overshooting CSF. Also confirm holdout elpd does not degrade.

## Notes / out of scope

- This targets **tissue retention** (within-tissue block). Boosting the exact self-phenotype diagonal
  would additionally favour phenotype *stability* (a switching knob) — separate, not included.
- If tissue-only trafficking proves too rigid (e.g. TRM vs TEMRA egress differs by phenotype), the
  next step is the factored `growth × π_a × Φ` operator; not needed for this fix.
