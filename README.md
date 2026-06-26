# traffic

Bayesian **Gamma–Poisson population-dynamics model** for clonal cell trafficking
across tissue × phenotype states, in **JAX + NumPyro**. Inference is gradient-based
MCMC (NUTS): the per-source allocations marginalize exactly (Poisson superposition),
leaving `M` as the only unknown, sampled directly.

```
M_{zz'}   ~ Gamma(a_z, b_z)                  non-negative mean matrix (free row sums)
mu_j      = xtilde_j @ M                       destination intensity
y_j(z')   ~ Poisson(d_{j,z'} * mu_j(z'))       over non-missing states
```

`M_{zz'}` = expected destination-`z'` cells per source-`z` cell over one forward
step; the row sum `g_z` is net growth, and `M = diag(g) T` splits growth from the
row-stochastic transition matrix `T`.

## Install
```bash
pip install -e .            # then `import traffic` anywhere
```
Runs on CPU and is **device-agnostic** — install a CUDA jaxlib
(`pip install -U "jax[cuda12]"`) to run NUTS on a GPU node with no code changes.

## Quickstart (synthetic)
```python
import jax; jax.config.update("jax_enable_x64", True)
import traffic as tm

# simulate from the generative model, then recover M
Xt, Y, D, mask, M_true = tm.simulate.make_synthetic(jax.random.PRNGKey(0), J=2000)
mc   = tm.fit_nuts(Xt, Y, D)            # NUTS -> posterior draws + mc.M_hat
T, g = tm.readouts.decompose(mc.M_hat)
```

## Fit your own data
A dataset is three `[J, L]` arrays — `Xtilde` (depth-rescaled source), `Y`
(destination counts), `D` (destination depth); `Y`,`D` are zeroed at missing
states. An AnnData loader builds them from per-cell obs columns
`patient, timepoint, tissue, phenotype, <clone-id>`:
```python
import os
obs = tm.data.from_h5ad(os.environ["TRAFFIC_H5"], tm.statespace.default())
mc  = tm.fit_nuts(obs.Xtilde, obs.Y, obs.D)
```

## Save / reload a fit  (e.g. GPU node → local)
Run the heavy NUTS fit on a GPU node, save it, copy it down, reload locally
(**numpy only** — no JAX/GPU needed to read it):
```bash
# on the GPU node (full data, f32 for speed)
TRAFFIC_H5=/path/to/data.h5ad TRAFFIC_X64=0 python scripts/fit_nuts.py   # -> nuts_fit.npz
```
```python
# locally, after copying nuts_fit.npz down
fit   = tm.io.load_fit("nuts_fit.npz")
draws = fit.draws(2000)                  # [2000, L, L] posterior M draws
T, g  = tm.readouts.decompose(fit.M_hat) # row-stochastic transitions + growth
```

## Modules
`statespace` · `config` · `data` (obs → arrays) · `model` · `simulate` ·
`mcmc` (NUTS) · `posterior` (draws/CIs) · `io` (save/load) · `readouts` · `dynamics`.

## Scripts
- `scripts/recovery_check.py` — synthetic recovery (proves model + inference).
- `scripts/fit_nuts.py` — fit full-data NUTS and save (`TRAFFIC_H5`; GPU-ready).

## Test
```bash
pip install -e . && python -m pytest -q
```

## License
[MIT](LICENSE) © 2026 Nicholas Ceglia
