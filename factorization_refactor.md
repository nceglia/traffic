# Factorized Model Refactor — Specification & Implementation Plan

**Goal.** Replace the iid-Gamma, free-entry prior on `M` with the **factored operator**
`M = diag(g)·(π ⊗ Φ)` (expansion × tissue-trafficking × phenotype-switching) as the **only** model,
keeping the NB2 likelihood, NUTS, and every downstream read-out. This is a **clean roll-forward, not
a dual-mode addition**: the Gamma parameterization (`a0_matrix`, the Gamma `PriorConfig`) **and** the
per-tissue / per-patient dispersion modes are **deleted** — the CAVI path taught us not to keep dead
code runnable. There is no config switch and no way to run an "unfactorized" fit. Downstream
read-outs consume the assembled `M` (still exposed as a deterministic site) — that is a *data
interface*, not backward-compat with an old model.

**Authoritative spec:** `docs/model_methods.tex` (this plan implements exactly that model). Any
disagreement between this plan and `model_methods.tex` is resolved in favour of the tex.

**Audience:** an agent should be able to implement the rebuild from this document alone.

---

## 1. Target model (what we are building)

State space `Z = S × K = 3 tissues × 13 phenotypes`, `L = 39`. State index convention
(**critical, do not change**): `z = tissue*K + phenotype`. Source state `z=(a,u)`, destination
`z'=(b,v)`, destination column index `= b*K + v`.

Factors, drawn per source state `z=(a,u)`:

| factor | shape | prior | meaning |
|---|---|---|---|
| `g[z]` | `[L]` | `LogNormal(μ_g, σ_g²)` | growth / net-share change (centred ~1, heavy right tail) |
| `π[z]` | `[L, S]` | `Dir(α_a)` per source tissue `a`, persistence-favouring | destination-tissue distribution (trafficking) |
| `Φ[z, b]` | `[L, S, K]` | `Dir(β)` symmetric | destination-phenotype distribution per destination tissue `b` (switching) |
| `log φ` | scalar | `Normal(0, σ_φ²)` | NB2 concentration (global) |

**Assembly** (the one equation that must be exactly right):
```
M[z, b*K+v] = g[z] · π[z, b] · Φ[z, b, v]
# vectorized:  M3 = g[:,None,None] * π[:,:,None] * Φ        # [L,S,K]
#              M  = M3.reshape(L, L)                          # columns ordered b*K+v
```
Row sum `Σ_z' M[z,z'] = g[z]` (since π, Φ are simplices). `T = M/g` is row-stochastic.

**Likelihood (unchanged from current NB path):**
```
μ_j = x̃_j M                      # [J, L], x̃ = depth-rescaled source (fixed input)
mean_j(z') = d_{j,s'} · μ_j(z')
y_j(z') ~ NB2(mean, φ)            # only where m_{j,s'}=1 (masked otherwise)
```

**Inference:** NUTS over `{g, π, Φ, log φ}` in the unconstrained reparameterization (log for `g`,
NumPyro's default simplex bijections for `π, Φ`). `M` is a deterministic function of the factors.

---

## 2. Main challenges & how we solve them (read first)

1. **Assembly correctness (reshape ordering).** The destination column must be `b*K+v` to match
   `statespace.z = tissue*K+phenotype`. *Solution:* build `M3 [L,S,K]` then `.reshape(L,L)`; add a
   unit test on a tiny `S=2,K=2` case asserting `M[z, b*K+v] == g[z]π[z,b]Φ[z,b,v]` for all indices.
   This is the single highest-risk line; test it before anything else.

2. **Persistence prior on `π`.** `α_a` is per source tissue with the stay entry (`b=a`) boosted.
   *Solution:* an `alpha_pi_matrix(prior, L, S) -> [L,S]` helper mirroring the existing
   `a0_matrix`: off-tissue entries `= alpha_off`, and rows in tissue `a` get `alpha_stay[a]` on
   column `a`. Verify the induced prior `E[π(stay)] = α_stay/(α_stay+(S-1)α_off)` and reuse
   `scripts/prior_predictive_persistence.py` to confirm.

3. **Dirichlet geometry / NUTS efficiency.** ~1.6k parameters (`g`:39, `π`:39×2 free, `Φ`:117×12
   free, `logφ`:1). Off-route `Φ[z,b]` with small `π[z,b]` is weakly identified (funnel risk); the
   LogNormal `g` has a heavy tail. *Solution:* rely on proper Dirichlet priors (already regularize
   off-route directions), `target_accept ≥ 0.9`, and monitor divergences/ESS. Fallbacks if the
   sampler struggles: raise `target_accept` to 0.95, tighten `β`/`σ_g`, or increase warmup. Record
   which was needed. NumPyro's simplex transform is already "non-centered"; no manual reparam needed
   first pass.

4. **Support restrictions (implausible transitions).** `model_methods.tex` allows sparsifying `M`
   by removing destinations from `π`/`Φ` support — but a full Dirichlet cannot carry structural
   zeros. *Solution:* **v1 uses full support** (no structural zeros); this is a stated scope choice.
   Future options (documented, not implemented now): (a) per-source ragged Dirichlets over retained
   support, or (b) a near-zero pseudo-count on disallowed entries. Do **not** block the refactor on
   this.

5. **Downstream interface (a data interface, NOT backward-compat with an old model).** `scoring.py`,
   `io.py`, `readouts.py`, and figure scripts consume **`M` draws** (`fit.samples` `[N,L,L]`) and the
   NB concentration. *Solution:* expose `M` as `numpyro.deterministic("M", ...)` (so
   `get_samples()["M"]` = `fit.samples` as today) and keep the NB site name `log_r`; those consumers
   work unchanged against the one model. Factors (`g,π,Φ`) are additive. Confirm `decompose(M)`
   recovers `(T, g)` consistent with the sampled `π, g`. There is exactly one model behind this
   interface — no Gamma fallback.

6. **Recovery testing.** *Solution:* add `simulate.sample_M_factored` (draw `g,π,Φ`, assemble `M`)
   and a `recovery_check_factored` that fits and checks `M_hat≈M_true`, factor recovery, and
   convergence.

7. **Clean replacement — no coexistence (the CAVI lesson).** *Solution:* **delete** the Gamma
   `numpyro_model`, `a0_matrix`, the Gamma `PriorConfig` (`a0`/`b0`/`a0_stay`), and any CAVI/`fit`
   remnants. `fit_nuts` calls the factored model **unconditionally** — no `isinstance` dispatch, no
   `parameterization` flag. The existing callers switch to constructing a `FactoredPriorConfig`;
   nothing can run an unfactorized fit.

8. **Dispersion scope — global only.** The model is NB2 with a **single global** concentration.
   *Solution:* **delete** the `none` / `tissue` / `patient` dispersion branches from the likelihood,
   from `LikelihoodConfig`, and from `scoring._r_per_col` / `_DISP_SITES` (they become dead code once
   the factored model only ever registers `log_r`). Poisson is recoverable as the `φ→∞` limit; no
   separate `family` knob is kept. With one model there is nothing to share, so the likelihood is
   inlined in `numpyro_model_factored` (no `_nb_likelihood` extraction needed).

---

## 3. Current codebase inventory

| file | role | change |
|---|---|---|
| `traffic/config.py` | `PriorConfig` (Gamma), `MCMCConfig`, `LikelihoodConfig` | **replace** `PriorConfig`→`FactoredPriorConfig`; **delete** the `LikelihoodConfig` dispersion knob (global-only) |
| `traffic/mcmc.py` | `numpyro_model` (Gamma), `a0_matrix`, dispersion modes, `fit_nuts`, `MCMCResult` | **delete** Gamma model + `a0_matrix` + tissue/patient dispersion; **add** `assemble`, `alpha_pi_matrix`, `numpyro_model_factored`, `MCMCResult.factors`; `fit_nuts` unconditional |
| `traffic/io.py` | `SavedFit`, `save_fit`, `load_fit` | **extend** to persist/reload `factors` |
| `traffic/simulate.py` | Gamma `sample_M`, `sample_counts`, `make_synthetic` | **add** `sample_M_factored` + a factored synthetic maker |
| `traffic/model.py` | `intensity`, `expected_counts` (pure JAX) | unchanged (operate on `M`) |
| `traffic/readouts.py`, `dynamics.py`, `scoring.py`, `posterior.py` | consume `M` draws | unchanged; add optional factor read-outs later |
| `scripts/fit_nuts*.py`, `recovery_check.py`, `mcmc_check.py`, `temporal_fit.py` | call `fit_nuts` | unchanged; add a factored fit script |

---

## 4. File-by-file implementation

### 4.1 `traffic/config.py` — **replace** `PriorConfig` with `FactoredPriorConfig`
Delete `PriorConfig` (`a0`/`b0`/`a0_stay`) and the `dispersion` field of `LikelihoodConfig`
(global-only now; `family`/`alpha_scale`/`sigma_scale`/tissue/patient knobs go). `MCMCConfig` stays.
```python
@dataclass(frozen=True)
class FactoredPriorConfig:
    """Priors for the factored operator M = diag(g)·(π ⊗ Φ)."""
    mu_g: float = 0.0            # LogNormal location for g (centred ~1)
    sigma_g: float = 0.5        # LogNormal scale (heavy right tail)
    alpha_off: float = 1.0      # Dirichlet conc. on off-tissue destinations of π
    alpha_stay: "float | tuple" = 4.0   # stay (b=a) conc.; scalar=global, len-S=per source tissue
    beta: float = 1.0           # symmetric Dirichlet conc. for Φ
    sigma_phi: float = 1.0      # Normal scale on log φ
```

### 4.2 `traffic/mcmc.py`
```python
def alpha_pi_matrix(prior: FactoredPriorConfig, L: int, S: int = 3) -> np.ndarray:
    """[L,S] Dirichlet concentration for π: off-tissue = alpha_off; stay (col=tissue(z)) = alpha_stay."""
    K = L // S
    A = np.full((L, S), float(prior.alpha_off))
    stay = prior.alpha_stay
    stay = [float(stay)] * S if np.isscalar(stay) else [float(x) for x in stay]
    if len(stay) != S:
        raise ValueError(f"alpha_stay must be scalar or length-{S}")
    for a in range(S):
        A[a*K:(a+1)*K, a] = stay[a]
    return A

def assemble(g, pi, Phi, S=3):
    """M[z, b*K+v] = g[z]·pi[z,b]·Phi[z,b,v]. Shared by the model AND simulate (one reshape rule)."""
    L = pi.shape[0]
    return (g[:, None, None] * pi[:, :, None] * Phi).reshape(L, L)

def numpyro_model_factored(Xtilde, Y, D, alpha_pi, beta, mu_g, sigma_g, sigma_phi, S=3):
    import numpyro, numpyro.distributions as dist
    L = alpha_pi.shape[0]; K = L // S
    g   = numpyro.sample("g",  dist.LogNormal(mu_g, sigma_g).expand([L]).to_event(1))          # [L]
    pi  = numpyro.sample("pi", dist.Dirichlet(alpha_pi).to_event(1))                           # [L,S]
    Phi = numpyro.sample("Phi", dist.Dirichlet(beta*jnp.ones(K)).expand([L, S]).to_event(2))   # [L,S,K]
    M   = numpyro.deterministic("M", assemble(g, pi, Phi, S))                                  # [L,L]
    observed  = D > 0
    safe_mean = jnp.where(observed, D * (Xtilde @ M), 1.0)
    log_r = numpyro.sample("log_r", dist.Normal(0.0, sigma_phi))   # NB concentration; site name log_r (compat)
    numpyro.sample("Y", dist.NegativeBinomial2(safe_mean, jnp.exp(log_r)).mask(observed), obs=Y)
```
Global NB2 only — inlined, no `_nb_likelihood`/`family`/`dispersion`/`patient_idx` args. `assemble`
is shared with `simulate` (one reshape convention, tested once).
`fit_nuts` (unconditional, no dispatch): `L = Xtilde.shape[1]`; build
`alpha_pi = jnp.asarray(alpha_pi_matrix(prior, L))`; run NUTS on `numpyro_model_factored`; extract
`samples["M"]` → `MCMCResult.samples`, `samples["g"/"pi"/"Phi"]` → `MCMCResult.factors`, and
`samples["log_r"]` → the dispersion field. Diagnostics on the **free sites** (§5.7). The Gamma
`numpyro_model` and `a0_matrix` are removed from the file.

`MCMCResult`: add `factors: dict | None = None` (`{"g","pi","Phi"}` numpy arrays).

**Site-name compat (load-bearing):** register the NB concentration under the site name `log_r`
(global) — **not** `log_phi` — so `scoring._r_per_col`'s global branch resolves it (the tex writes
the symbol `φ`; the *site name* stays `log_r`). The `log_r_s`/`mu_s` (tissue/patient) sites are
**deleted** along with those modes. Enable float64 once (`jax.config.update("jax_enable_x64", True)`)
before any NB fit. `L = Xtilde.shape[1]`, `K = L // S`; the shared `assemble()` (above) is the single
reshape convention used by both model and simulate. Factored fit scripts must still save `rho`
(persistence read-outs fall back to uniform `1/K` without it).

### 4.3 `traffic/io.py`
`save_fit`: if `result.factors`, write `factor_g/factor_pi/factor_Phi` arrays + `meta["parameterization"]="factored"`.
`load_fit`: reload them into `SavedFit.factors` (default `None` → Gamma fits load exactly as today).

### 4.4 `traffic/simulate.py`
```python
def sample_M_factored(key, mu_g, sigma_g, alpha_pi, beta, L, S=3):
    K = L // S
    k1, k2, k3 = jax.random.split(key, 3)
    g   = jnp.exp(mu_g + sigma_g * jax.random.normal(k1, (L,)))
    pi  = jax.random.dirichlet(k2, jnp.asarray(alpha_pi))                 # [L,S]
    Phi = jax.random.dirichlet(k3, beta*jnp.ones(K), shape=(L, S))        # [L,S,K]
    M = (g[:, None, None] * pi[:, :, None] * Phi).reshape(L, L)
    return M, {"g": g, "pi": pi, "Phi": Phi}
```
A `make_synthetic_factored` mirrors `make_synthetic` but uses `sample_M_factored`.

### 4.5 Tests / scripts
- `tests/test_assembly.py`: the `S=2,K=2` reshape assertion (Challenge 1).
- `tests/test_alpha_pi.py`: `alpha_pi_matrix` shape/placement + induced stay-prior mean.
- `scripts/recovery_check_factored.py`: simulate → fit → assert corr(`M_hat`,`M_true`)>0.95, factor
  recovery within 95% CI, `r_hat_max<1.01`, `ess_min>400`, few divergences.
- `scripts/fit_nuts_factored.py`: real-data fit entry (mirrors `fit_nuts.py`, passes `FactoredPriorConfig`).

---

## 5. Testing & acceptance criteria

**Unit / correctness (fast):**
1. **Assembly index test** — `S=2,K=2` exact-index assertion (Challenge 1).
2. **Basis-equivalence test** — a *single* shared `assemble(g,π,Φ)→M` helper is used by **both**
   `simulate` and the model; a test asserts the factored `M` lands in the **same basis** as a Gamma
   `M` (reuse `statespace.tissue_of`/`col_tissue`; assert `row_sum(M)=g` and equality against a
   hand-built reference). Catches a shared-but-wrong reshape convention that recovery alone can't.
3. **`alpha_pi_matrix`** placement + induced `E[π_stay]=α_stay/(α_stay+(S-1)α_off)` — a *cheap prior
   identity*, kept as a unit test, **not** the acceptance gate (it is true by construction).
4. **float64 enabled** — assert `jax.config.read("jax_enable_x64")` in the fit entrypoint.

**Acceptance gates (the refactor must clear these):**
5. **Data-weighted persistence — the actual objective.** Push posterior `M` through the *real* clones
   (depth-weighted), row-normalize, and require the diagonal (stay) persistence to match the
   empirical ~0.75–0.88 per tissue — i.e. the "biologically-backwards ~1/3" is gone. The prior
   marginal (#3) does *not* test this; this is the thing the refactor exists to fix.
6. **Route-mass-stratified recovery.** On factored synthetic data, bin routes by mass `g·π`; require
   `corr(M_hat,M_true)>0.95` and factor recovery **only where route mass exceeds a floor**.
   Separately assert **starved routes** (`π[z,b]→0`) have `Φ[z,b]` posterior ≈ its Dirichlet prior
   (unidentified, as expected), and add a read-out flag marking such Φ entries "prior, not inferred"
   so downstream switching claims never trust them.
7. **Convergence on the free sites, not on `M`.** Compute `r_hat`/ESS on `{g,π,Φ,log_r}` (the funnel
   lives in the factor geometry, masked when collapsed into `M`). Gate `r_hat<1.01` on all sites and
   ESS on the **identified block** (row-sums=g and on-route entries above the mass floor); do **not**
   gate a single global `ess_min` — prior-dominated off-route directions have intrinsically low ESS
   (~35–70) and would fail correct fits.
8. **Three-way head-to-head (not two-way).** The factored model has ≈ the same parameter count as the
   free-Gamma `M` (~1.5k) — **not** a dimensionality reduction, so any benefit is prior structure. On
   identical holdouts/seed compare (i) vanilla Gamma, (ii) **Gamma + persistence prior** (`a0_stay`,
   already on `fb65cb1`), (iii) factored; justify the refactor only by what it adds **over (ii)**.
   State the param-count parity in the spec.
9. **Backward compat:** `io.load_fit` on a factored `.npz` returns `.samples`/`.dispersion` usable by
   `scoring.score_holdout` and `readouts.summary` with **no code changes**; `decompose(M)` matches
   the sampled `(g, T=π⊗Φ)`.
10. **Real-data smoke fit** completes; `M_hat` sane (row sums ≈ g).

---

## 6. Rollout

Order (each step green before the next): **config (replace `PriorConfig`) → mcmc (delete Gamma model
+ `a0_matrix` + tissue/patient dispersion; add `assemble`, `alpha_pi_matrix`,
`numpyro_model_factored`, `MCMCResult.factors`; `fit_nuts` unconditional) → assembly+alpha+basis unit
tests → io → simulate → recovery_check_factored → real-data smoke fit → downstream-interface check.**
The plan stops at a validated local build; committing/pushing and launching the GPU refit (global +
holdouts, `FactoredPriorConfig`) are the user's calls. Keep the Gamma path runnable throughout
(dispatch on prior type). All work stays on the `factorized-model` branch.

---

## 7. Multi-agent audit (outcome)

A 5-lens audit (math correctness, JAX/NumPyro feasibility, reproducibility, backward-compat, testing)
plus synthesis was run over this plan (39 raw findings).

**Scope caveat — the "blocking" tier is invalid.** The audit subagents could only reach the
`gbm_trafficking` session source, so their filesystem-verification lens audited the *old* manuscript
package (`gbm_trafficking/trafficking/` — Pyro/SVI/Multinomial, K=11/L=33) and the *old*
`docs/model_methods.tex` (Poisson) — **not** this plan's target, the `traffic/` repo (NumPyro/NUTS/NB2,
K=13/L=39, with `fit_nuts`/`a0_matrix`/`MCMCResult`/`NegativeBinomial2` all present, verified directly
this session). Their four "critical" findings ("fictional codebase", "Pyro not NumPyro", "Poisson not
NB2", "K=11/L=33") are artifacts of that repo mismatch and are **dismissed** after verification.

**The design findings are repo-independent and valid — incorporated above:**
- data-weighted persistence as the real acceptance gate (§5.5), not the by-construction prior marginal;
- route-mass-stratified recovery + a "prior, not inferred" flag on starved Φ (§5.6);
- convergence diagnostics on the free sites `{g,π,Φ,log_r}`, not on `M`; no global ESS gate (§5.7);
- three-way comparison vs the persistence-prior Gamma, with param-count parity stated (§5.8);
- one shared `assemble()` helper + a basis-equivalence test (§5.2);
- keep the NB dispersion site names (`log_r`) for scoring compat, save `rho`, enable float64 (§4.2);
- executable rollout with `L`/`col_tissue` provenance; commit/push step removed (§4.2, §6).

**Verdict: PROCEED.** No structural blocker survives verification: the plan targets the correct
codebase, and the incorporated findings harden the test/acceptance layer. Ready to implement on
`factorized-model`.

_Process note: re-running the codebase-verification lens would need the `traffic/` repo added as a
session source (subagents can't see it or the worktree); the design lenses did not depend on it._
