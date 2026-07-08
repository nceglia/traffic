"""traffic.recovery.plant_pi_zeros: exact/near-zero route planting preserves the simplex and, once
injected through the simulator, produces exactly-zero M route-mass on the closed routes (so the
operator is well-defined there while Phi on those routes is unidentified -- the boundary the
degeneracy experiment probes).

Run: python tests/test_recovery.py
"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from traffic import statespace
from traffic.config import FactoredPriorConfig
from traffic.mcmc import assemble
from traffic.recovery import (plant_pi_zeros, prior_variance, route_mass, score_recovery,
                              shrinkage_zscore)
from traffic.simulate import make_synthetic_factored


def test_plant_pi_zeros_simplex_and_exact_zero():
    """Planting exact zeros sets pi[z,b]=0 and keeps every source row a valid simplex."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(0)
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    routes = [(0, 1), (K, 2), (2 * K, 0)]                         # off-stay routes, one per source tissue
    pi0, got = plant_pi_zeros(pi, routes, eps=0.0)
    assert got == sorted(routes), "returns the sorted unique route list"
    assert np.allclose(pi0.sum(1), 1.0, atol=1e-12), "rows must remain simplices"
    for z, b in routes:
        assert pi0[z, b] == 0.0, "exact-zero planting must set pi[z,b] to 0"
    # relative proportions of the untouched columns in a planted row are preserved after renorm
    z, b = routes[0]
    keep = [c for c in range(S) if c != b]
    assert np.allclose(pi0[z, keep] / pi0[z, keep].sum(),
                       pi[z, keep] / pi[z, keep].sum(), atol=1e-12)
    print("PASS test_plant_pi_zeros_simplex_and_exact_zero")


def test_plant_pi_zeros_guards():
    """Out-of-bounds routes and a fully-closed row are rejected."""
    rng = np.random.default_rng(2)
    pi = rng.random((6, 3)); pi /= pi.sum(1, keepdims=True)
    for bad in [(6, 0), (0, 3)]:
        try:
            plant_pi_zeros(pi, [bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for out-of-bounds route {bad}")
    try:
        plant_pi_zeros(pi, [(0, 0), (0, 1), (0, 2)])              # closes the whole row 0
    except ValueError:
        print("PASS test_plant_pi_zeros_guards")
        return
    raise AssertionError("expected ValueError when a row is fully closed")


def test_planted_zero_gives_zero_route_mass():
    """Inject a planted-zero pi through the simulator: assembled M route-mass on the closed routes is
    exactly 0, while an open route on the same source row still carries mass."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig()
    rng = np.random.default_rng(1)
    g = rng.random(L) + 0.1
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    routes = [(3, 1), (K + 2, 0)]
    pi0, _ = plant_pi_zeros(pi, routes, eps=0.0)
    _, _, _, _, M_true, _ = make_synthetic_factored(
        jax.random.PRNGKey(0), prior, L=L, J=10, S=S, phi=10.0,
        factors={"g": g, "pi": pi0, "Phi": Phi})
    M = np.asarray(M_true)
    for z, b in routes:
        block = M[z, b * K:(b + 1) * K]                          # route (z -> tissue b) mass over phenotypes
        assert np.allclose(block, 0.0, atol=1e-12), f"closed route {(z, b)} must carry 0 mass"
    z0 = 3
    open_b = next(b for b in range(S) if (z0, b) not in routes)
    assert M[z0, open_b * K:(open_b + 1) * K].sum() > 0, "an open route must still carry mass"
    print("PASS test_planted_zero_gives_zero_route_mass")


def test_route_mass_equals_M_block_sum():
    """route_mass(g,pi)[z,b] equals the assembled M row-block sum M[z, b*K:(b+1)*K]."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(4)
    g = rng.random(L) + 0.2
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    w = route_mass(g, pi)
    for z in range(L):
        for b in range(S):
            assert abs(w[z, b] - M[z, b * K:(b + 1) * K].sum()) < 1e-9
    print("PASS test_route_mass_equals_M_block_sum")


def test_score_recovery_perfect():
    """A synthetic fit whose draws are tightly centered on the truth scores near-perfect recovery
    (validates score_recovery's math with no NUTS run)."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(3)
    g = rng.random(L) + 0.5
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    N = 300
    gd = g[None] + 0.01 * rng.standard_normal((N, L))
    pid = np.clip(pi[None] + 0.005 * rng.standard_normal((N, L, S)), 1e-6, None); pid /= pid.sum(2, keepdims=True)
    Phid = np.clip(Phi[None] + 0.005 * rng.standard_normal((N, L, S, K)), 1e-6, None); Phid /= Phid.sum(3, keepdims=True)
    Md = M[None] + 0.01 * rng.standard_normal((N, L, L))
    log_r = np.log(10.0) + 0.05 * rng.standard_normal(N)
    res = types.SimpleNamespace(factors={"g": gd, "pi": pid, "Phi": Phid}, samples=Md,
                                dispersion={"params": {"log_r": log_r}})
    m = score_recovery(res, {"g": g, "pi": pi, "Phi": Phi, "M": M, "phi": 10.0}, ss)
    assert m["g"]["corr"] > 0.99 and m["pi"]["corr"] > 0.95 and m["M"]["corr"] > 0.99
    assert m["Phi_on"]["corr"] > 0.90 and m["g"]["coverage"] > 0.8 and m["phi"]["covered"]
    print("PASS test_score_recovery_perfect")


def test_prior_variance_shapes_and_values():
    """prior_variance returns the right shapes and matches the analytic LogNormal/Normal variances."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, sigma_phi=1.0)
    pv = prior_variance(prior, ss, n=8000, seed=0)
    assert pv["g"].shape == (L,) and pv["pi"].shape == (L, S) and pv["Phi"].shape == (L, S, K)
    assert abs(pv["log_r"] - prior.sigma_phi ** 2) < 0.1, "log_r var ~ sigma_phi^2"
    ana_g = (np.exp(prior.sigma_g ** 2) - 1) * np.exp(2 * prior.mu_g + prior.sigma_g ** 2)
    assert abs(pv["g"].mean() - ana_g) < 0.15 * ana_g, "g var ~ analytic LogNormal variance"
    print("PASS test_prior_variance_shapes_and_values")


def test_shrinkage_zscore_perfect():
    """Draws tightly centered on truth give |z|~0 (on-truth) and shrinkage~1 (post var << prior var)."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(3)
    g = rng.random(L) + 0.5
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    N = 300
    gd = g[None] + 0.01 * rng.standard_normal((N, L))
    pid = np.clip(pi[None] + 0.005 * rng.standard_normal((N, L, S)), 1e-6, None); pid /= pid.sum(2, keepdims=True)
    Phid = np.clip(Phi[None] + 0.005 * rng.standard_normal((N, L, S, K)), 1e-6, None); Phid /= Phid.sum(3, keepdims=True)
    log_r = np.log(10.0) + 0.05 * rng.standard_normal(N)
    res = types.SimpleNamespace(factors={"g": gd, "pi": pid, "Phi": Phid},
                                samples=M[None] + np.zeros((N, L, L)),
                                dispersion={"params": {"log_r": log_r}})
    pv = prior_variance(FactoredPriorConfig(), ss, n=4000)
    sz = shrinkage_zscore(res, {"g": g, "pi": pi, "Phi": Phi, "M": M, "phi": 10.0}, pv)
    for f in ("g", "pi", "Phi"):
        assert np.nanmedian(np.abs(sz[f]["z"])) < 3.0, f"{f} |z| should be small"
        assert np.nanmedian(sz[f]["s"]) > 0.9, f"{f} shrinkage should be high"
    assert abs(sz["log_r"]["z"][0]) < 3.0 and sz["log_r"]["s"][0] > 0.5
    print("PASS test_shrinkage_zscore_perfect")


if __name__ == "__main__":
    test_plant_pi_zeros_simplex_and_exact_zero()
    test_plant_pi_zeros_guards()
    test_planted_zero_gives_zero_route_mass()
    test_route_mass_equals_M_block_sum()
    test_score_recovery_perfect()
    test_prior_variance_shapes_and_values()
    test_shrinkage_zscore_perfect()
    print("\nALL RECOVERY-HELPER TESTS PASSED")
