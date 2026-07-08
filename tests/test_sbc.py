"""traffic.sbc.rank_statistic zero-fit self-check: under the null -- 'posterior' == prior, i.e. the
draws and the truth are i.i.d. from the SAME law -- the SBC rank is Uniform{0..L}. This validates
the ranker independently of any MCMC fit, so a broken ranker fails in milliseconds rather than
masquerading as sampler miscalibration in a long run.

Run: python tests/test_sbc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from traffic import statespace
from traffic.sbc import (ecdf_diff_band, interval_coverage, rank_hist_expected_band,
                         rank_statistic, sbc_scalars)


def _chi2_uniform_pvalue(ranks, L):
    """Chi-square goodness-of-fit p-value of `ranks` against Uniform{0..L} (dof = L)."""
    from scipy.stats import chi2 as chi2dist
    counts = np.bincount(ranks, minlength=L + 1)
    exp = ranks.size / (L + 1)
    chi2 = ((counts - exp) ** 2 / exp).sum()
    return float(chi2dist.sf(chi2, L))


def test_rank_statistic_uniform_continuous():
    """i.i.d. continuous draws (no ties): rank of the truth among L peers is Uniform{0..L}."""
    rng = np.random.default_rng(0)
    R, L = 20000, 63
    x = rng.standard_normal((R, L + 1))                          # col 0 = truth, cols 1: = 'posterior'
    ranks = np.array([rank_statistic(x[r, 1:], x[r, 0], rng) for r in range(R)])
    assert ranks.min() >= 0 and ranks.max() <= L, "ranks must lie in {0..L}"
    p = _chi2_uniform_pvalue(ranks, L)
    assert p > 1e-3, f"continuous ranks non-uniform: p={p:.2e}"
    print("PASS test_rank_statistic_uniform_continuous")


def test_rank_statistic_tiebreak_uniform():
    """Heavy exact ties (discrete draws sharing the truth's law): the uniform random tie-break keeps
    ranks in {0..L} and still Uniform -- the property that lets SBC handle degenerate scalars (e.g.
    a starved Phi direction whose posterior collapses onto the prior)."""
    rng = np.random.default_rng(1)
    R, L = 20000, 15
    ranks = np.empty(R, dtype=int)
    for r in range(R):
        draws = rng.integers(0, 5, L)                            # few distinct values -> many ties
        truth = int(rng.integers(0, 5))
        ranks[r] = rank_statistic(draws, truth, rng)
    assert ranks.min() >= 0 and ranks.max() <= L, "ranks must lie in {0..L}"
    p = _chi2_uniform_pvalue(ranks, L)
    assert p > 1e-3, f"tie-broken ranks non-uniform: p={p:.2e}"
    print("PASS test_rank_statistic_tiebreak_uniform")


def test_rank_statistic_endpoints():
    """A truth below/above all draws ranks 0 / L; ties with no rng are handled deterministically."""
    assert rank_statistic([1.0, 2.0, 3.0], -5.0) == 0
    assert rank_statistic([1.0, 2.0, 3.0], 99.0) == 3
    # all draws equal the truth -> tie-break spans {0..L}; with a fixed rng it is reproducible
    r = rank_statistic([2.0, 2.0, 2.0], 2.0, np.random.default_rng(0))
    assert 0 <= r <= 3
    print("PASS test_rank_statistic_endpoints")


def test_sbc_scalars_fixed_functional():
    """sbc_scalars applies the SAME fixed functional to batched draws and the single-theta truth."""
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    rng = np.random.default_rng(0)
    g = rng.random((5, L)) + 0.1
    pi = rng.random((5, L, S)); pi /= pi.sum(2, keepdims=True)
    Phi = rng.random((5, L, S, K)); Phi /= Phi.sum(3, keepdims=True)
    lr = rng.standard_normal(5)
    sc = sbc_scalars(g, pi, Phi, lr, ss)
    assert sc["log_r"].shape == (5,)
    assert np.allclose(sc["g[0]"], g[:, 0])
    assert np.allclose(sc["pi_stay[0]"], pi[:, 0, 0])
    assert np.allclose(sc["Phi_stay0[0]"], Phi[:, 0, 0, 0])
    assert np.allclose(sc["mean_pi_stay"], pi[:, np.arange(L), ss.tissue_of].mean(1))
    one = sbc_scalars(g[:1], pi[:1], Phi[:1], lr[:1], ss)          # the N=1 truth path
    assert one["g[0]"].shape == (1,) and np.isclose(one["g[0]"][0], g[0, 0])
    print("PASS test_sbc_scalars_fixed_functional")


def test_interval_coverage_uniform():
    """Rank-implied coverage of a uniform rank distribution equals the nominal level."""
    rng = np.random.default_rng(0)
    ranks01 = rng.random(50000)
    cov = interval_coverage(ranks01, [0.5, 0.8, 0.9, 0.95])
    for q in (0.5, 0.8, 0.9, 0.95):
        assert abs(cov[q] - q) < 0.02, f"coverage@{q}={cov[q]:.3f}"
    # 2-D form returns per-column coverage
    cov2 = interval_coverage(rng.random((5000, 3)), [0.9])
    assert np.asarray(cov2[0.9]).shape == (3,)
    print("PASS test_interval_coverage_uniform")


def test_ecdf_band_calibrated():
    """The simultaneous band is calibrated: ~(1-alpha) of uniform null ensembles stay fully inside."""
    Lp, R, alpha = 99, 1000, 0.05
    rng = np.random.default_rng(0)
    grid, _, lo, hi = ecdf_diff_band(rng.integers(0, Lp + 1, R), Lp, n_sim=1000, alpha=alpha, seed=7)
    inside, T = 0, 400
    for _ in range(T):
        obs = np.searchsorted(np.sort(rng.integers(0, Lp + 1, R) / Lp), grid, side="right") / R - grid
        if np.all((obs >= lo - 1e-9) & (obs <= hi + 1e-9)):
            inside += 1
    frac = inside / T
    assert (1 - alpha) - 0.06 <= frac <= 1.0, f"band coverage {frac:.2f} vs nominal {1 - alpha}"
    print(f"PASS test_ecdf_band_calibrated (null coverage {frac:.2f})")


def test_ecdf_band_flags_bias():
    """A grossly non-uniform (biased-low) rank sample pushes the ECDF difference outside the band."""
    Lp, R = 99, 1500
    rng = np.random.default_rng(3)
    biased = rng.integers(0, (Lp + 1) // 2, R)                    # ranks in lower half -> ecdf bulges up
    _, obs, lo, hi = ecdf_diff_band(biased, Lp, n_sim=500, alpha=0.05, seed=1)
    assert np.any(obs > hi + 1e-6), "biased ranks must exit the simultaneous band"
    print("PASS test_ecdf_band_flags_bias")


def test_rank_hist_expected_band():
    """Expected count and a binomial band around it (lo <= expected <= hi, band widens as ci grows)."""
    exp, lo, hi = rank_hist_expected_band(1000, 20, ci=0.99)
    assert abs(exp - 50) < 1e-9 and lo < exp < hi
    _, lo95, hi95 = rank_hist_expected_band(1000, 20, ci=0.95)
    assert lo95 >= lo and hi95 <= hi                              # 95% band nested inside 99%
    print("PASS test_rank_hist_expected_band")


if __name__ == "__main__":
    test_rank_statistic_uniform_continuous()
    test_rank_statistic_tiebreak_uniform()
    test_rank_statistic_endpoints()
    test_sbc_scalars_fixed_functional()
    test_interval_coverage_uniform()
    test_ecdf_band_calibrated()
    test_ecdf_band_flags_bias()
    test_rank_hist_expected_band()
    print("\nALL SBC RANKER + DIAGNOSTIC TESTS PASSED")
