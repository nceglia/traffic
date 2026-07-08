"""Generative-consistency tests: the simulator (traffic.simulate) and the fitter's likelihood
(traffic.mcmc.numpyro_model_factored) must be the IDENTICAL generative process. This is the
non-negotiable precondition for simulation-based calibration (SBC) to mean anything -- if the data
are not drawn from the same law the fit assumes, rank uniformity tests nothing.

Run: python tests/test_generative_consistency.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from traffic import model, statespace
from traffic.config import FactoredPriorConfig
from traffic.mcmc import assemble
from traffic.simulate import make_synthetic_factored, prior_sample, sample_counts


def _rand_factors(rng, L, S, K):
    g = rng.random(L) + 0.1
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    return g, pi, Phi


def test_nb2_parity_simulator_vs_model():
    """sample_counts' Gamma-Poisson mixture == dist.NegativeBinomial2 in mean, variance AND full pmf.

    Pins the dispersion CONVENTION (which arg is phi), not just the first two moments -- the single
    load-bearing invariant for SBC validity. sample_counts uses rate = D*(Xtilde @ M); choosing
    M=[[mu]], Xtilde=D=1 makes rate == mu on every cell so we can compare the marginal count law.
    """
    import numpyro.distributions as dist
    from scipy.stats import nbinom
    mu, phi, n = 12.0, 4.0, 400_000
    km, ks = jax.random.split(jax.random.PRNGKey(1))
    y_model = np.asarray(dist.NegativeBinomial2(mu, phi).sample(km, (n,)))
    y_sim = np.asarray(sample_counts(ks, jnp.array([[mu]]), jnp.ones((n, 1)),
                                     jnp.ones((n, 1)), jnp.ones((n, 1)), phi=phi)).ravel()
    av = mu + mu ** 2 / phi                                        # NB2 variance = mu + mu^2/phi
    assert abs(y_model.mean() - mu) < 0.15 and abs(y_sim.mean() - mu) < 0.15, "mean mismatch"
    assert abs(y_model.var() - av) < 0.03 * av and abs(y_sim.var() - av) < 0.03 * av, "var mismatch"
    kgrid = np.arange(0, 40)
    pmf = nbinom.pmf(kgrid, phi, phi / (phi + mu))                 # analytic NB2 pmf, r=phi, p=phi/(phi+mu)
    emp_sim = np.array([(y_sim == k).mean() for k in kgrid])
    emp_model = np.array([(y_model == k).mean() for k in kgrid])
    assert np.abs(emp_sim - pmf).max() < 0.01, "simulator pmf must match analytic NB2"
    assert np.abs(emp_sim - emp_model).max() < 0.01, "simulator and model pmf must match"
    print("PASS test_nb2_parity_simulator_vs_model")


def test_mean_map_parity():
    """The mean map D*(Xtilde @ M) is one convention across simulate's rate, the numpyro model's
    safe_mean, and model.expected_counts. Pin them together so a transpose/broadcast change can't
    silently drift the simulator away from the likelihood."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(3)
    g, pi, Phi = _rand_factors(rng, L, S, K)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    J = 6
    Xt = rng.random((J, L))
    tissue_of = np.repeat(np.arange(S), K)
    D = rng.integers(0, 50, (J, S)).astype(float)[:, tissue_of]   # per-dest-tissue depth, some zeros
    ec = np.asarray(model.expected_counts(jnp.asarray(Xt), jnp.asarray(M), jnp.asarray(D)))
    rate_sim = D * (Xt @ M)                                       # the simulator's rate
    observed = D > 0
    safe_mean = np.where(observed, D * (Xt @ M), 1.0)            # the model's safe_mean
    assert np.allclose(ec, rate_sim, atol=1e-9), "model.expected_counts must equal simulate's rate"
    assert np.allclose(ec[observed], safe_mean[observed], atol=1e-9), "must equal model safe_mean"
    print("PASS test_mean_map_parity")


def test_prior_sample_shapes_and_prior():
    """prior_sample returns the full theta INCLUDING log_r, with valid simplex factors, M==assemble,
    and log_r ~ Normal(0, sigma_phi). This is the shared truth-generator for recovery and SBC."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(sigma_phi=1.0)
    th = prior_sample(jax.random.PRNGKey(0), prior, L, S)
    assert th["g"].shape == (L,) and th["pi"].shape == (L, S) and th["Phi"].shape == (L, S, K)
    assert np.asarray(th["log_r"]).shape == (), "log_r must be a scalar"
    assert np.allclose(np.asarray(th["pi"]).sum(1), 1.0, atol=1e-6), "pi rows must be simplices"
    assert np.allclose(np.asarray(th["Phi"]).sum(2), 1.0, atol=1e-6), "Phi rows must be simplices"
    assert np.allclose(np.asarray(th["M"]),
                       np.asarray(assemble(th["g"], th["pi"], th["Phi"])), atol=1e-9)
    keys = jax.random.split(jax.random.PRNGKey(7), 400)
    lr = np.array([float(prior_sample(k, prior, L, S)["log_r"]) for k in keys])
    assert abs(lr.mean()) < 0.2 and abs(lr.std() - prior.sigma_phi) < 0.2, "log_r ~ Normal(0,sigma_phi)"
    print("PASS test_prior_sample_shapes_and_prior")


def test_factors_injection_roundtrip():
    """make_synthetic_factored(factors=...) uses the injected truth VERBATIM (full and partial
    injection) rather than a prior draw -- the hook the boundary experiment depends on."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig()
    rng = np.random.default_rng(5)
    g, pi, Phi = _rand_factors(rng, L, S, K)
    inj = {"g": g, "pi": pi, "Phi": Phi}
    _, _, _, _, M_true, used = make_synthetic_factored(
        jax.random.PRNGKey(0), prior, L=L, J=50, S=S, phi=10.0, factors=inj)
    assert np.allclose(np.asarray(used["g"]), g, atol=1e-12), "g must be injected verbatim"
    assert np.allclose(np.asarray(used["pi"]), pi, atol=1e-12), "pi must be injected verbatim"
    assert np.allclose(np.asarray(used["Phi"]), Phi, atol=1e-12), "Phi must be injected verbatim"
    assert np.allclose(np.asarray(M_true),
                       np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi))), atol=1e-9)
    # partial injection: only pi fixed, g/Phi still drawn from the prior
    _, _, _, _, _, used2 = make_synthetic_factored(
        jax.random.PRNGKey(1), prior, L=L, J=50, S=S, phi=10.0, factors={"pi": pi})
    assert np.allclose(np.asarray(used2["pi"]), pi, atol=1e-12), "partial: pi still injected"
    assert not np.allclose(np.asarray(used2["g"]), g), "partial: g must be drawn, not injected"
    print("PASS test_factors_injection_roundtrip")


if __name__ == "__main__":
    test_nb2_parity_simulator_vs_model()
    test_mean_map_parity()
    test_prior_sample_shapes_and_prior()
    test_factors_injection_roundtrip()
    print("\nALL GENERATIVE-CONSISTENCY TESTS PASSED")
