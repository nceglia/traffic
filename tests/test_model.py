"""Core correctness tests. Run: PYTHONPATH=. python -m pytest -q"""
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from traffic import MCMCConfig, PriorConfig, fit_nuts, model, readouts, simulate, statespace


def test_nuts_recovery():
    """NUTS recovers M_true from synthetic data."""
    Xt, Y, D, m, Mt = simulate.make_synthetic(jax.random.PRNGKey(3), J=400)
    res = fit_nuts(Xt, Y, D, PriorConfig(),
                   MCMCConfig(num_warmup=120, num_samples=120, num_chains=2))
    Mt = np.asarray(Mt)
    assert np.isfinite(res.samples).all()
    assert res.num_divergences == 0
    assert np.corrcoef(res.M_hat.ravel(), Mt.ravel())[0, 1] > 0.9            # recovers truth
    assert np.corrcoef(res.M_hat.sum(1), Mt.sum(1))[0, 1] > 0.9             # recovers growth


def test_decompose_identity():
    ss = statespace.default()
    M = np.abs(np.random.default_rng(0).normal(size=(ss.L, ss.L))) + 0.1
    T, g = readouts.decompose(M)
    assert np.allclose(T.sum(1), 1.0)            # T row-stochastic
    assert np.allclose(np.diag(g) @ T, M)         # M = diag(g) T


def test_tissue_traffic_rows_sum_to_one():
    ss = statespace.default()
    M = np.abs(np.random.default_rng(0).normal(size=(ss.L, ss.L))) + 0.1
    T, _ = readouts.decompose(M)
    P = readouts.tissue_traffic(T, ss)
    assert np.allclose(P.sum(1), 1.0)


def test_masking_invariance():
    """Missing destination states (D=0) must not enter the likelihood."""
    Xt, Y, D, m, Mt = simulate.make_synthetic(jax.random.PRNGKey(2), J=200)
    M = jnp.asarray(Mt)
    ll1 = model.poisson_loglik(Y, D, Xt, M)
    Y2 = jnp.where(D == 0, 999.0, Y)             # corrupt masked entries
    ll2 = model.poisson_loglik(Y2, D, Xt, M)
    assert np.isclose(float(ll1), float(ll2))


def test_save_load_roundtrip(tmp_path):
    """io.save_fit/load_fit round-trips a NUTS fit; reload is numpy-only."""
    from traffic import io
    from traffic.mcmc import MCMCResult
    ss = statespace.default()
    rng = np.random.default_rng(0)
    samp = rng.gamma(2.0, size=(40, ss.L, ss.L)).astype(np.float32)
    fake = MCMCResult(samples=samp, M_hat=samp.mean(0), M_median=np.median(samp, 0),
                      sd=samp.std(0), num_divergences=0, r_hat_max=1.01, ess_min=123.0, n_draws=40)
    sn = io.load_fit(io.save_fit(tmp_path / "nuts", fake, ss=ss, meta={"note": "test"}))
    assert sn.samples.shape == (40, ss.L, ss.L)
    assert sn.draws(20).shape == (20, ss.L, ss.L)
    assert sn.meta.get("note") == "test"
    assert np.allclose(sn.M_hat, fake.M_hat, atol=1e-5)
