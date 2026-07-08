"""Parameter recovery + identifiability/boundary checks (Goal 1 of the simulation validation).

Simulate from the factored model at a KNOWN truth (g, pi, Phi, phi), fit, and check the posterior
lands on the truth -- and, at the pi->0 boundary, that the unidentified Phi correctly reverts to its
Dir(beta) prior mean 1/K while everything identified recovers. This is the clean version of the
identifiability panel: contraction shows "posterior != prior" (the data moved it), recovery shows
"posterior == truth" (it moved to the RIGHT place).

The known-truth INJECTION hook lives in traffic.simulate (sample_M_factored(..., factors=...) and
make_synthetic_factored(..., factors=...)); this module holds the zero-planting helper and (Phase 2)
the recovery scoring. SBC / interval calibration (Goal 2) lives in traffic.sbc.
"""
from __future__ import annotations

import numpy as np


def plant_pi_zeros(pi_base, zero_routes, eps=0.0):
    """Plant (near-)zero mass on chosen trafficking routes, keeping each source row a simplex.

    pi_base     : [L,S] destination-tissue distribution to start from (rows sum to 1).
    zero_routes : iterable of (z, b) -- set pi[z,b] = eps for each, then renormalize row z to 1.
    eps=0.0  -> an EXACT closed route: M route-mass on (z,b) is exactly 0 and Phi[z,b,:] is fully
                unidentified (its posterior must revert to the Dir(beta) prior). This is the anchor
                of the boundary experiment.
    eps>0    -> a near-closed route (e.g. 1e-4, 1e-2) for the "how much mass identifies switching"
                sweep, where identifiability returns as the route reopens.

    The removed mass is absorbed by the rest of row z in proportion (renormalization); when the stay
    column dominates the row it soaks up most of it, so report planting as "mass moved toward stay",
    not "spread evenly". Returns (pi [L,S] new array, routes sorted unique list of (z,b)) so callers
    can hand the same route list to the recovery gates that must treat those Phi[z,b] as prior-
    reverting and EXCLUDE them from the identified-block metrics.
    """
    pi = np.array(pi_base, dtype=float, copy=True)
    if pi.ndim != 2:
        raise ValueError(f"pi_base must be [L,S]; got shape {pi.shape}")
    L, S = pi.shape
    routes = sorted({(int(z), int(b)) for z, b in zero_routes})
    for z, b in routes:
        if not (0 <= z < L and 0 <= b < S):
            raise ValueError(f"route {(z, b)} out of bounds for pi_base [L={L}, S={S}]")
        pi[z, b] = eps
    row = pi.sum(1, keepdims=True)
    if np.any(row <= 0):
        raise ValueError("a source row summed to <=0 after planting; leave at least one open route")
    pi /= row
    return pi, routes


# --------------------------------------------------------------------------------------------------
# Recovery scoring
# --------------------------------------------------------------------------------------------------

def route_mass(g, pi):
    """Expansion-weighted trafficking mass into each destination tissue: w[z,b] = g[z] * pi[z,b].

    This -- not raw pi -- is the quantity to split identified (on-route) from starved (off-route) Phi:
    a route with tiny pi but large g can still carry identifying counts, and vice versa. Equals the
    assembled M row-block sum M[z, b*K:(b+1)*K]. Shape [L,S].
    """
    return np.asarray(g)[:, None] * np.asarray(pi)


def _corr(a, b):
    a = np.asarray(a).ravel(); b = np.asarray(b).ravel()
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _coverage(draws, truth, ci):
    """Fraction of truth entries inside the central `ci` posterior interval. draws [N, ...], truth
    [...] broadcastable to draws[0]."""
    lo = np.quantile(draws, (1.0 - ci) / 2.0, axis=0)
    hi = np.quantile(draws, (1.0 + ci) / 2.0, axis=0)
    t = np.asarray(truth)
    return float(np.mean((lo <= t) & (t <= hi)))


def _tv(P, Q):
    """Mean total-variation distance between last-axis distributions: mean over rows of 0.5*sum|P-Q|."""
    return float(np.mean(0.5 * np.abs(np.asarray(P) - np.asarray(Q)).sum(-1)))


def _entropy(P):
    """Shannon entropy along the last axis (nats)."""
    P = np.asarray(P)
    return -np.sum(np.where(P > 0, P * np.log(P), 0.0), axis=-1)


def score_recovery(res, truth, ss, ci=0.90, tau_on=0.02):
    """Score one fit against a known truth, per factor, split identified vs starved.

    res   : MCMCResult (reads res.factors {g,pi,Phi}, res.samples [N,L,L], res.dispersion.log_r).
    truth : dict {"g":[L], "pi":[L,S], "Phi":[L,S,K], "M":[L,L], "phi": scalar}.
    tau_on: route-mass threshold (w = g*pi) separating on-route (identified) from starved Phi.

    Every factor gets a point metric (Pearson corr of posterior mean vs truth), a calibration metric
    (fraction of truths inside the central `ci` interval), and an error metric. Phi is reported twice:
    on-route (must recover) and off-route/starved (must revert to the Dir(beta) prior mean 1/K, i.e.
    d_prior -> 0 and entropy -> log K). Returns a nested dict of floats.
    """
    S, K, L = ss.S, ss.K, ss.L
    g_t = np.asarray(truth["g"]); pi_t = np.asarray(truth["pi"]); Phi_t = np.asarray(truth["Phi"])
    M_t = np.asarray(truth["M"]); phi_t = float(truth["phi"])
    gd = np.asarray(res.factors["g"]); pid = np.asarray(res.factors["pi"]); Phid = np.asarray(res.factors["Phi"])
    Md = np.asarray(res.samples); lr = np.asarray(res.dispersion["params"]["log_r"])
    g_h, pi_h, Phi_h, M_h = gd.mean(0), pid.mean(0), Phid.mean(0), Md.mean(0)

    on = route_mass(g_t, pi_t) >= tau_on                  # [L,S] identified routes
    off = ~on
    unif = np.full(K, 1.0 / K)
    out = {}
    out["g"] = {"corr": _corr(g_h, g_t), "coverage": _coverage(gd, g_t, ci),
                "rmse_log": float(np.sqrt(np.mean((np.log(g_h) - np.log(g_t)) ** 2)))}
    out["pi"] = {"corr": _corr(pi_h, pi_t), "coverage": _coverage(pid, pi_t, ci),
                 "tv": _tv(pi_h, pi_t)}
    if on.any():
        out["Phi_on"] = {"corr": _corr(Phi_h[on], Phi_t[on]),
                         "coverage": _coverage(Phid[:, on, :], Phi_t[on], ci),
                         "tv": _tv(Phi_h[on], Phi_t[on]), "n": int(on.sum())}
    else:
        out["Phi_on"] = {"corr": float("nan"), "coverage": float("nan"), "tv": float("nan"), "n": 0}
    if off.any():
        out["Phi_off"] = {"d_prior": float(np.mean(0.5 * np.abs(Phi_h[off] - unif).sum(-1))),
                          "entropy_ratio": float(_entropy(Phi_h[off]).mean() / np.log(K)),
                          "d_truth": _tv(Phi_h[off], Phi_t[off]), "n": int(off.sum())}
    else:
        out["Phi_off"] = {"d_prior": float("nan"), "entropy_ratio": float("nan"),
                          "d_truth": float("nan"), "n": 0}
    mt, mh = M_t.ravel(), M_h.ravel()
    hi = mt >= np.quantile(mt, 0.9)
    out["M"] = {"corr": _corr(mh, mt), "corr_top": _corr(mh[hi], mt[hi]),
                "coverage": _coverage(Md, M_t, ci),
                "rel_fro": float(np.linalg.norm(M_h - M_t) / max(np.linalg.norm(M_t), 1e-12))}
    phi_draws = np.exp(lr)
    lo, hi_ = np.quantile(phi_draws, (1.0 - ci) / 2.0), np.quantile(phi_draws, (1.0 + ci) / 2.0)
    out["phi"] = {"true": phi_t, "post_mean": float(phi_draws.mean()),
                  "covered": bool(lo <= phi_t <= hi_)}
    out["route"] = {"tau_on": tau_on, "n_on": int(on.sum()), "n_off": int(off.sum())}
    return out


def recover_once(key, prior, cfg, ss, J=800, phi=10.0, factors=None, depth_lo=50, depth_hi=4000,
                 p_missing=0.4, sparsity=0.3, ci=0.90, tau_on=0.02):
    """Simulate one synthetic dataset (optionally at injected `factors`), fit by NUTS, and score
    recovery against the known truth. Lazy-imports jax/numpyro so importing this module stays light.

    Returns (res, truth, metrics, data) with truth={g,pi,Phi,M,phi} and data=(Xtilde,Y,D,mask).
    """
    import jax  # noqa: F401  (side-effect-free; caller enables x64)
    from .simulate import make_synthetic_factored
    from .mcmc import fit_nuts

    L, S = ss.L, ss.S
    Xt, Y, D, mask, M_true, fu = make_synthetic_factored(
        key, prior, L=L, J=J, S=S, depth_lo=depth_lo, depth_hi=depth_hi,
        p_missing=p_missing, sparsity=sparsity, phi=phi, factors=factors)
    Xt, Y, D, mask = np.asarray(Xt), np.asarray(Y), np.asarray(D), np.asarray(mask)
    res = fit_nuts(Xt, Y, D, prior, cfg)
    truth = {"g": np.asarray(fu["g"]), "pi": np.asarray(fu["pi"]), "Phi": np.asarray(fu["Phi"]),
             "M": np.asarray(M_true), "phi": float(phi)}
    metrics = score_recovery(res, truth, ss, ci=ci, tau_on=tau_on)
    return res, truth, metrics, (Xt, Y, D, mask)


# --------------------------------------------------------------------------------------------------
# Shrinkage / z-score (the canonical "learned + calibrated" diagnostic)
# --------------------------------------------------------------------------------------------------

def prior_variance(prior, ss, n=4000, seed=0):
    """Monte-Carlo marginal prior variance per parameter entry for {g, pi, Phi, log_r}, drawn from
    the SAME distributions as the model prior (mirrors simulate.prior_sample). Returns
    {"g":[L], "pi":[L,S], "Phi":[L,S,K], "log_r": float}. Used as the denominator of the posterior
    shrinkage 1 - var_post/var_prior."""
    import jax
    import jax.numpy as jnp
    from .mcmc import alpha_pi_matrix

    L, S, K = ss.L, ss.S, ss.K
    kg, kpi, kphi, kr = jax.random.split(jax.random.PRNGKey(seed), 4)
    alpha_pi = jnp.asarray(alpha_pi_matrix(prior, L, S))
    g = jnp.exp(prior.mu_g + prior.sigma_g * jax.random.normal(kg, (n, L)))
    pi = jax.random.dirichlet(kpi, alpha_pi, shape=(n, L))                     # [n,L,S]
    Phi = jax.random.dirichlet(kphi, prior.beta * jnp.ones(K), shape=(n, L, S))  # [n,L,S,K]
    log_r = prior.sigma_phi * jax.random.normal(kr, (n,))
    return {"g": np.asarray(g.var(0)), "pi": np.asarray(pi.var(0)),
            "Phi": np.asarray(Phi.var(0)), "log_r": float(np.asarray(log_r.var()))}


def shrinkage_zscore(res, truth, prior_var):
    """Per-parameter posterior z-score and shrinkage for the sampled parameters {g, pi, Phi, log_r}.

      z = (mean_post - truth) / sd_post      (signed: sign shows bias direction)
      s = 1 - var_post / var_prior           (1 = data highly informative; 0 = posterior==prior)

    The canonical Schad/Betancourt/Vasishth (2021) recovery diagnostic: the ideal cloud sits at
    s->1, |z|->0 (informative AND on-truth); s->0 is prior-limited (not broken -- e.g. starved Phi);
    large |z| at high s is confident-but-wrong. Returns {factor: {"z": array, "s": array}} with arrays
    in each factor's natural shape (g [L], pi [L,S], Phi [L,S,K], log_r scalar-as-[1]).
    """
    out = {}
    for name in ("g", "pi", "Phi"):
        d = np.asarray(res.factors[name])                     # [N, ...]
        sd = d.std(0); vpost = d.var(0)
        t = np.asarray(truth[name])
        z = np.where(sd > 1e-12, (d.mean(0) - t) / np.where(sd > 1e-12, sd, 1.0), np.nan)
        s = np.where(prior_var[name] > 1e-12, 1.0 - vpost / np.where(prior_var[name] > 1e-12, prior_var[name], 1.0), np.nan)
        out[name] = {"z": z, "s": s}
    lr = np.asarray(res.dispersion["params"]["log_r"])
    z = float((lr.mean() - np.log(truth["phi"])) / max(lr.std(), 1e-12))
    s = float(1.0 - lr.var() / max(prior_var["log_r"], 1e-12))
    out["log_r"] = {"z": np.array([z]), "s": np.array([s])}
    return out
