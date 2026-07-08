"""Simulation-Based Calibration + interval coverage (Goal 2 of the simulation validation).

For many rounds: draw a full truth theta* ~ prior (traffic.simulate.prior_sample, which includes
log_r), simulate data, fit, and record the RANK of each true scalar among the fit's posterior draws.
Under a correct sampler those ranks are Uniform{0..L_post} (Talts et al. 2018, arXiv:1804.06788);
systematic non-uniformity is a proof of a bug in the model, sampler, or simulator. Central-interval
coverage is the coarse, reader-facing cousin computed from the same rounds.

Recovery / identifiability (Goal 1) lives in traffic.recovery. The scalar spec, thinning, ECDF
bands, and the round driver are Phase 4; this module currently holds the rank primitive that the
zero-fit harness self-check exercises.
"""
from __future__ import annotations

import numpy as np


def rank_statistic(post_draws, true_val, rng=None):
    """SBC rank of a scalar truth among L posterior draws, an integer in {0, ..., L}.

    rank = #{draws < true} + a uniform random tie-break over the #{draws == true} equal draws
    (Talts et al. 2018): breaking ties uniformly at random keeps the rank exactly Uniform{0..L}
    under the null even for discrete or degenerate scalars. For continuous scalars ties have measure
    zero, so the tie-break never fires and rank = #{draws < true}.

    post_draws : 1-D array of L posterior draws of the scalar.
    true_val   : the ground-truth scalar (theta*).
    rng        : numpy Generator, int seed, or None -- SBC rounds should pass a per-round Generator
                 for reproducibility. Only consulted when there are exact ties.
    """
    d = np.asarray(post_draws, dtype=float).ravel()
    lt = int(np.count_nonzero(d < true_val))
    eq = int(np.count_nonzero(d == true_val))
    if eq == 0:
        return lt
    gen = rng if hasattr(rng, "integers") else np.random.default_rng(rng)
    return lt + int(gen.integers(0, eq + 1))


# --------------------------------------------------------------------------------------------------
# Ranked scalars: fixed functionals of theta, applied identically to the truth and posterior draws
# --------------------------------------------------------------------------------------------------

def sbc_scalars(g, pi, Phi, log_r, ss):
    """The ranked SBC scalars as a fixed ordered dict name -> array over the leading (sample) axis.

    g [N,L], pi [N,L,S], Phi [N,L,S,K], log_r [N]. The SAME function is applied to the truth (pass
    N=1) and to the posterior draws, so the ranked functional is identical every round -- a hard SBC
    requirement. The set spans the three model axes plus the global dispersion, using FIXED indices
    (a representative source state per tissue) and two global test quantities (Modrak et al. 2023):

      log_r                     global NB2 log-concentration
      g[z]                      expansion at a representative state of each tissue
      pi_stay[z] = pi[z,a]      stay/persistence probability (always on-route -> identified)
      Phi_stay0[z] = Phi[z,a,0] a switching probability on the stay route (identified)
      mean_pi_stay              mean stay probability over all states (global trafficking)
      mean_log_g                mean log expansion (global)
    """
    g = np.asarray(g); pi = np.asarray(pi); Phi = np.asarray(Phi); log_r = np.asarray(log_r).ravel()
    S, K, L = ss.S, ss.K, ss.L
    tissue_of = ss.tissue_of
    out = {"log_r": log_r}
    for a in range(S):
        z = a * K                                        # representative source state of tissue a
        out[f"g[{z}]"] = g[:, z]
        out[f"pi_stay[{z}]"] = pi[:, z, a]               # a == tissue_of[z]
        out[f"Phi_stay0[{z}]"] = Phi[:, z, a, 0]
    out["mean_pi_stay"] = pi[:, np.arange(L), tissue_of].mean(1)
    out["mean_log_g"] = np.log(g).mean(1)
    return out


def _ess_1d(x):
    """Effective sample size of a 1-D chain via the initial-positive-sequence autocorrelation sum.
    Approximate (draws are chain-flattened), used only to check thinning adequacy -- not a gate."""
    x = np.asarray(x, float); n = x.size
    if n < 8 or x.std() < 1e-12:
        return float(n)
    x = x - x.mean()
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, n=m)
    acf = np.fft.irfft(f * np.conj(f))[:n].real
    acf /= acf[0]
    tau = 1.0
    for t in range(1, n):
        if acf[t] <= 0:
            break
        tau += 2 * acf[t]
    return float(n / max(tau, 1.0))


# --------------------------------------------------------------------------------------------------
# One SBC round: draw theta* ~ prior -> simulate -> fit -> rank each true scalar among the draws
# --------------------------------------------------------------------------------------------------

def sbc_round(key, prior, cfg, ss, J, L_post, depth_lo=50, depth_hi=4000, p_missing=0.4, sparsity=0.3):
    """One SBC round. Draws theta* from the FULL prior (incl. log_r, so phi ~ prior -- never fixed),
    simulates, fits by NUTS (cfg carries the per-round seed), thins to a FIXED L_post evenly-spaced
    draws, and returns the rank of each true scalar among them. Lazy-imports jax/numpyro.

    Returns {ranks:{name:int}, ess:{name:float}, L_post:int, phi:float, r_hat_max, num_divergences,
    ess_min_identified}. Per-scalar `ess` is computed IN-HARNESS from the ranked functionals' own
    chains (res.ess_min excludes Phi, so it must not gate Phi thinning).
    """
    import jax
    from .mcmc import fit_nuts
    from .simulate import make_synthetic_factored, prior_sample

    L, S = ss.L, ss.S
    k_truth, k_sim = jax.random.split(key)
    th = prior_sample(k_truth, prior, L, S)
    phi = float(np.exp(np.asarray(th["log_r"])))
    Xt, Y, D, _, _, _ = make_synthetic_factored(
        k_sim, prior, L=L, J=J, S=S, depth_lo=depth_lo, depth_hi=depth_hi,
        p_missing=p_missing, sparsity=sparsity, phi=phi,
        factors={"g": np.asarray(th["g"]), "pi": np.asarray(th["pi"]), "Phi": np.asarray(th["Phi"])})
    res = fit_nuts(np.asarray(Xt), np.asarray(Y), np.asarray(D), prior, cfg)

    gd = np.asarray(res.factors["g"]); pid = np.asarray(res.factors["pi"])
    Phid = np.asarray(res.factors["Phi"]); lrd = np.asarray(res.dispersion["params"]["log_r"])
    n = gd.shape[0]
    full = sbc_scalars(gd, pid, Phid, lrd, ss)                     # full chain, for ESS
    ess = {k: _ess_1d(v) for k, v in full.items()}
    idx = np.unique(np.linspace(0, n - 1, min(L_post, n)).round().astype(int))   # systematic thin
    thin = sbc_scalars(gd[idx], pid[idx], Phid[idx], lrd[idx], ss)
    tru = sbc_scalars(np.asarray(th["g"])[None], np.asarray(th["pi"])[None],
                      np.asarray(th["Phi"])[None], np.asarray([float(th["log_r"])]), ss)
    rng = np.random.default_rng(int(np.asarray(k_truth)[0]))       # reproducible per-round tie-break
    ranks = {k: rank_statistic(thin[k], float(tru[k][0]), rng) for k in thin}
    return {"ranks": ranks, "ess": ess, "L_post": int(idx.size), "phi": phi,
            "r_hat_max": float(res.r_hat_max), "num_divergences": int(res.num_divergences),
            "ess_min_identified": float(res.ess_min)}


# --------------------------------------------------------------------------------------------------
# Diagnostics over many rounds (all derived from the ranks -- no extra fits/storage)
# --------------------------------------------------------------------------------------------------

def interval_coverage(ranks01, levels):
    """Rank-implied central credible-interval coverage. ranks01 = rank/L_post in [0,1], shape [R] or
    [R,P]. The central-`q` interval contains the truth iff its normalized rank is in [(1-q)/2,(1+q)/2],
    so coverage is exactly a function of the ranks -- same rounds, no extra work. Returns
    {q: coverage} (scalar per q if [R], else [P])."""
    ranks01 = np.asarray(ranks01, float)
    out = {}
    for q in levels:
        hit = (ranks01 >= (1.0 - q) / 2.0) & (ranks01 <= (1.0 + q) / 2.0)
        out[q] = hit.mean(0) if ranks01.ndim == 2 else float(hit.mean())
    return out


def ecdf_diff_band(ranks, L_post, n_sim=2000, alpha=0.05, seed=0, grid_n=201):
    """ECDF-of-normalized-ranks minus the uniform CDF, with a SIMULTANEOUS confidence band under the
    discrete-uniform null (Sailynoja et al. 2022, simultaneous-KS form). The band half-width is the
    (1-alpha) quantile of the max absolute ECDF-difference over `n_sim` simulated null ensembles of
    the SAME size R and discreteness L_post, so a curve straying outside it ANYWHERE is a calibrated
    (1-alpha simultaneous) rejection of uniformity. A constant band is used rather than a variance-
    standardized one because the standardized form degenerates at the grid endpoints for small R
    (sd -> 0 while the discrete ECDF keeps 1/R jumps), blowing the band up.

    ranks: array of integer ranks in {0..L_post}. Returns (grid, ecdf_diff, lo, hi).
    """
    r = np.asarray(ranks); R = r.size
    grid = np.linspace(0.0, 1.0, grid_n)

    def ediff(vals):
        return np.searchsorted(np.sort(vals), grid, side="right") / vals.size - grid

    obs = ediff(r / L_post)
    rng = np.random.default_rng(seed)
    maxdev = np.array([np.max(np.abs(ediff(rng.integers(0, L_post + 1, size=R) / L_post)))
                       for _ in range(n_sim)])
    c = float(np.quantile(maxdev, 1.0 - alpha))
    band = np.full_like(grid, c)
    return grid, obs, -band, band


def rank_hist_expected_band(R, n_bins, ci=0.99):
    """Expected per-bin count R/n_bins and a `ci` binomial(R, 1/n_bins) interval for a rank histogram."""
    from scipy.stats import binom
    p = 1.0 / n_bins
    return R / n_bins, float(binom.ppf((1 - ci) / 2, R, p)), float(binom.ppf((1 + ci) / 2, R, p))
