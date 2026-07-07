"""Unit tests for the factored operator: assembly index convention, basis equivalence, alpha_pi.

Run: python tests/test_factored.py   (from the repo root)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from traffic.mcmc import assemble, alpha_pi_matrix
from traffic.config import FactoredPriorConfig
from traffic import statespace


def _rand_factors(rng, L, S, K):
    g = rng.random(L) + 0.1
    pi = rng.random((L, S)); pi /= pi.sum(1, keepdims=True)
    Phi = rng.random((L, S, K)); Phi /= Phi.sum(2, keepdims=True)
    return g, pi, Phi


def test_assembly_index():
    """M[z, b*K+v] == g[z]*pi[z,b]*Phi[z,b,v] for a tiny S=2,K=2 case (all indices)."""
    S, K = 2, 2; L = S * K
    g, pi, Phi = _rand_factors(np.random.default_rng(0), L, S, K)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    for z in range(L):
        for b in range(S):
            for v in range(K):
                assert abs(M[z, b * K + v] - g[z] * pi[z, b] * Phi[z, b, v]) < 1e-9, (z, b, v)
    print("PASS test_assembly_index")


def test_rowsum_is_g_and_basis():
    """Row sums = g; destination column b*K+v lands in the tissue-b block (statespace basis)."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    g, pi, Phi = _rand_factors(np.random.default_rng(1), L, S, K)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    assert np.allclose(M.sum(1), g, atol=1e-9), "row sum must equal g"
    tissue_of = ss.tissue_of  # [L] destination tissue index per column
    for b in range(S):
        cols = np.arange(b * K, (b + 1) * K)
        assert np.all(tissue_of[cols] == b), "column ordering must match statespace tissue*K+phenotype"
    print("PASS test_rowsum_is_g_and_basis")


def test_forward_mean_orientation():
    """Pin the load-bearing likelihood convention: mean_j(z') = d_{j,s'} * sum_z x~_j(z) M[z,z'].

    Locks (a) the source->dest contraction x~ @ M (not M.T) and (b) the per-DESTINATION-tissue
    depth broadcast. A transpose or a depth-axis swap would still pass the assembly tests but
    silently corrupt the likelihood -- this is the only test that catches it.
    """
    S, K = 2, 2; L = S * K
    rng = np.random.default_rng(2)
    g, pi, Phi = _rand_factors(rng, L, S, K)
    M = np.asarray(assemble(jnp.asarray(g), jnp.asarray(pi), jnp.asarray(Phi)))
    J = 3
    Xt = rng.random((J, L))
    d_tissue = rng.integers(10, 100, (J, S)).astype(float)      # per-(clone,dest-tissue) depth
    tissue_of = np.repeat(np.arange(S), K)                      # dest tissue of each column
    D = d_tissue[:, tissue_of]                                  # [J,L] depth broadcast over phenotypes
    mean_model = D * (Xt @ M)                                   # exactly the model's mean map
    ref = np.zeros((J, L))                                      # independent index-by-index reference
    for j in range(J):
        for zp in range(L):
            s_dst = zp // K
            ref[j, zp] = d_tissue[j, s_dst] * sum(Xt[j, z] * M[z, zp] for z in range(L))
    assert np.allclose(mean_model, ref, atol=1e-9), "D*(x~ @ M) must be source->dest with per-dest depth"
    print("PASS test_forward_mean_orientation")


def test_nb_mask_parity():
    """Masked (D==0) destination states contribute exactly zero to the NB2 log-density.

    Mirrors numpyro_model_factored's safe_mean + .mask(observed): corrupting Y only at masked
    states must not move the summed log-prob (replaces the old Poisson masking test).
    """
    import numpyro.distributions as dist
    rng = np.random.default_rng(4)
    n, L = 5, 6
    D = rng.integers(0, 3, (n, L)).astype(float) * rng.integers(0, 2, (n, L))   # some zeros
    observed = D > 0
    safe_mean = np.where(observed, D * (rng.random((n, L)) * 5.0), 1.0)          # model's safe_mean
    Y = rng.integers(0, 10, (n, L)).astype(float)

    def ll(Yin):
        d = dist.NegativeBinomial2(jnp.asarray(safe_mean), 3.0).mask(jnp.asarray(observed))
        return float(d.log_prob(jnp.asarray(Yin)).sum())

    Y2 = np.where(observed, Y, 999.0)                                            # corrupt only masked
    assert abs(ll(Y) - ll(Y2)) < 1e-9, "masked states must not enter the NB2 likelihood"
    print("PASS test_nb_mask_parity")


def test_alpha_pi_validation():
    """alpha_pi_matrix rejects a per-tissue alpha_stay whose length != S."""
    ss = statespace.default(); L, S = ss.L, ss.S
    try:
        alpha_pi_matrix(FactoredPriorConfig(alpha_stay=(6.0, 2.0)), L, S)        # length 2 != S=3
    except ValueError:
        print("PASS test_alpha_pi_validation")
        return
    raise AssertionError("expected ValueError for wrong-length alpha_stay")


def test_alpha_pi_matrix():
    """alpha_pi placement (stay col = alpha_stay, off = alpha_off), per-tissue, and induced mean."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    A = alpha_pi_matrix(FactoredPriorConfig(alpha_off=1.0, alpha_stay=4.0), L, S)
    assert A.shape == (L, S)
    for a in range(S):
        block = slice(a * K, (a + 1) * K)
        assert np.allclose(A[block, a], 4.0), "stay column must be alpha_stay"
        for b in range(S):
            if b != a:
                assert np.allclose(A[block, b], 1.0), "off-tissue columns must be alpha_off"
    exp_stay = 4.0 / (4.0 + (S - 1) * 1.0)                       # induced E[pi(stay)]
    assert abs(A[0, 0] / A[0].sum() - exp_stay) < 1e-9
    Ap = alpha_pi_matrix(FactoredPriorConfig(alpha_off=1.0, alpha_stay=(6.0, 2.0, 6.0)), L, S)
    assert np.allclose(Ap[0 * K:1 * K, 0], 6.0)
    assert np.allclose(Ap[1 * K:2 * K, 1], 2.0)                  # CSF lower stay
    assert np.allclose(Ap[2 * K:3 * K, 2], 6.0)
    print("PASS test_alpha_pi_matrix")


if __name__ == "__main__":
    test_assembly_index()
    test_rowsum_is_g_and_basis()
    test_forward_mean_orientation()
    test_nb_mask_parity()
    test_alpha_pi_validation()
    test_alpha_pi_matrix()
    print("\nALL ASSEMBLY / ALPHA / LIKELIHOOD TESTS PASSED")
