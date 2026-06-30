"""Predictive scoring of held-out fits -- magnitude + shape, with figures.

Each held-out transition j predicts a 33-vector of destination counts
    mean_j = D_j (x~_j M),   Y_j(z) ~ NegBinomial2(mean_j(z), r)
scored against the observed Y_j. Two regimes:

  predict   the holdout has a genuine never-seen test set (into-t transitions,
            left-out patient, rare clones, held-out clone hash). We score
            predictive accuracy, decomposed into:
              - magnitude: the total destination mass  N_j = sum_z Y_j(z)
                           (proper score: Dawid-Sebastiani on the predictive
                           total; readout: predicted-vs-observed, interval
                           coverage).
              - shape:     the destination composition  c_j = Y_j / N_j
                           (proper score: posterior-predictive multinomial
                           log-density; readout: abundance-weighted JS).
              - joint:     elpd_j = log p(Y_j | held-out data), the strictly
                           proper log score (magnitude + shape together).

  influence boundary holdouts (e.g. drop T1) have NO predictable target -- the
            held-out transitions need the dropped timepoint as a *source*
            feature. There we measure how much withholding moves the fit
            (`compare_fits`): predictive delta on the shared transitions plus
            the operator/read-out delta vs the full fit.

The scores fold in posterior uncertainty: every quantity averages over the
saved M (and dispersion) draws, not just M_hat.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom

from .io import load_fit
from .readouts import _js, decompose

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# dispersion: per-draw concentration r over the L destination columns
# --------------------------------------------------------------------------- #
def _r_per_col(fit, idx, ss, patient_idx=None):
    """Concentration r for each posterior draw, broadcast to [n_draws, L].

    Returns r evaluated on the subset of draws `idx`. For the `patient` mode the
    held-out patient has no fitted effect, so we fall back to the population mean
    (eps=0) -- flagged by the caller.
    """
    L, K = ss.L, ss.K
    col_tissue = np.repeat(np.arange(ss.S), K)          # [L] destination tissue
    disp = fit.dispersion
    if disp is None:                                    # Poisson -> r = inf
        return np.full((len(idx), L), np.inf)
    mode = disp["mode"]
    p = disp["params"]
    if mode == "global":
        r = np.exp(np.asarray(p["log_r"])[idx])[:, None]        # [n,1]
        return np.broadcast_to(r, (len(idx), L)).copy()
    if mode == "tissue":
        log_r_s = np.asarray(p["log_r_s"])[idx]                 # [n,3]
        return np.exp(log_r_s)[:, col_tissue]                   # [n,L]
    if mode == "patient":
        mu_s = np.asarray(p["mu_s"])[idx]                       # [n,3]
        # population-level (eps=0); per-patient effect unavailable out-of-sample
        return np.exp(mu_s)[:, col_tissue]                      # [n,L]
    raise ValueError(f"unknown dispersion mode {mode!r}")


# --------------------------------------------------------------------------- #
# test-set reconstruction from a fit's meta (mirrors the fit scripts)
# --------------------------------------------------------------------------- #
def _clone_bucket(clones):
    return np.array([int(hashlib.md5(str(c).encode()).hexdigest(), 16) % 100 for c in clones])


def test_mask(meta, obs):
    """(mask [J] bool, mode str) for the held-out set implied by `meta`.

    mode is 'predict' (a scorable test set) or 'influence' (empty test set, e.g.
    the T1 boundary holdout -- score via compare_fits instead).
    """
    if "split" in meta:
        split = meta["split"]
        if split == "patient":
            m = obs.patient == meta["held_out_patient"]
        elif split == "abundance":
            m = obs.n_src <= 2                                  # test_rule n_src<=2
        elif split == "clone8020":
            m = _clone_bucket(obs.clone) >= 80
        else:
            raise ValueError(f"unknown split {split!r}")
        return m, "predict"
    # timepoint holdout: test = into-t transitions (src_tp == t-1)
    tsrc = int(meta["test_src_tp"])
    m = obs.src_tp.astype(int) == tsrc
    return m, ("influence" if m.sum() == 0 else "predict")


# --------------------------------------------------------------------------- #
# core scoring on an explicit (X~, Y, D) test slice
# --------------------------------------------------------------------------- #
@dataclass
class Scores:
    name: str
    mode: str
    n: int
    # per-clone arrays (length n)
    elpd: np.ndarray = field(default_factory=lambda: np.empty(0))      # joint log score
    shape_ll: np.ndarray = field(default_factory=lambda: np.empty(0))  # multinomial log score
    dss: np.ndarray = field(default_factory=lambda: np.empty(0))       # magnitude Dawid-Sebastiani
    jsd: np.ndarray = field(default_factory=lambda: np.empty(0))       # shape JS divergence
    n_obs: np.ndarray = field(default_factory=lambda: np.empty(0))     # observed total
    n_pred: np.ndarray = field(default_factory=lambda: np.empty(0))    # predicted total (mean)
    covered: np.ndarray = field(default_factory=lambda: np.empty(0))   # in central 90% interval
    weight: np.ndarray = field(default_factory=lambda: np.empty(0))    # = n_obs (shape weight)
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        # shape is undefined for extinct clones (N_obs==0); average JS over the
        # survivors only, abundance-weighted.
        fin = np.isfinite(self.jsd)
        w = self.weight
        wsum = max(w[fin].sum(), _EPS)
        jsd_w = float((self.jsd[fin] * w[fin]).sum() / wsum) if fin.any() else np.nan
        lo = np.log1p(self.n_obs); lp = np.log1p(self.n_pred)
        spear = _spearman(lp, lo) if self.n > 2 else np.nan
        return {
            "name": self.name, "mode": self.mode, "n": self.n,
            "n_survive": int(fin.sum()), "frac_extinct": float((~fin).mean()) if self.n else np.nan,
            "elpd_mean": float(self.elpd.mean()) if self.n else np.nan,
            "shape_ll_mean": float(self.shape_ll.mean()) if self.n else np.nan,
            "dss_mean": float(self.dss.mean()) if self.n else np.nan,
            "jsd_wmean": jsd_w,
            "magnitude_spearman": float(spear),
            "coverage90": float(self.covered.mean()) if self.n else np.nan,
        }


def _spearman(a, b):
    a = np.asarray(a); b = np.asarray(b)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def score_slice(name, mode, Xt, Y, D, r_draws, M_draws, *, extra=None):
    """Score a test slice given posterior draws.

    Xt,Y,D : [n,L]   r_draws,M_draws : [S,L]/[S,L,L] aligned posterior subset.
    """
    n, L = Y.shape
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    N_obs = Yt.sum(1)                                          # observed totals

    joint = np.empty((len(M_draws), n))                       # joint loglik per draw
    shp = np.empty((len(M_draws), n))                         # shape loglik per draw
    mu_tot = np.empty((len(M_draws), n))                      # predicted total per draw
    var_tot = np.empty((len(M_draws), n))                     # predictive total var per draw
    p_bar = np.zeros((n, L))                                  # mean predicted composition

    for s, M in enumerate(M_draws):
        mean = D * (Xt @ M)                                   # [n,L]
        mean = np.where(obsv, mean, 0.0)
        r = r_draws[s]                                        # [L]
        # NB2 logpmf at Y over observed states
        with np.errstate(divide="ignore", invalid="ignore"):
            pp = r / (r + mean)                               # nbinom p
            lp = nbinom.logpmf(Yt, r, pp)
        joint[s] = np.where(obsv, lp, 0.0).sum(1)
        tot = mean.sum(1)
        mu_tot[s] = tot
        var_tot[s] = (mean + mean**2 / r).sum(1)              # Var of total | draw
        p = mean / np.maximum(tot[:, None], _EPS)             # composition | draw
        shp[s] = (Yt * np.log(p + _EPS)).sum(1)
        p_bar += p
    p_bar /= len(M_draws)

    elpd = logsumexp(joint, axis=0) - np.log(len(M_draws))    # joint log score
    shape_ll = logsumexp(shp, axis=0) - np.log(len(M_draws))  # shape (multinomial) log score

    # magnitude: predictive total moments (law of total variance over the draws)
    E_N = mu_tot.mean(0)
    Var_N = var_tot.mean(0) + mu_tot.var(0)
    dss = (N_obs - E_N) ** 2 / np.maximum(Var_N, _EPS) + np.log(np.maximum(Var_N, _EPS))

    # interval coverage: moment-match an NB (or Poisson) to (E_N, Var_N)
    covered = _interval_cover(N_obs, E_N, Var_N)

    # shape readout: JS at posterior-mean composition
    c_obs = Yt / np.maximum(N_obs[:, None], _EPS)
    jsd = np.array([_js(c_obs[j], p_bar[j]) if N_obs[j] > 0 else np.nan for j in range(n)])

    sc = Scores(name=name, mode=mode, n=n, elpd=elpd, shape_ll=shape_ll, dss=dss,
                jsd=jsd, n_obs=N_obs, n_pred=E_N, covered=covered,
                weight=N_obs.astype(float), meta=extra or {})
    return sc


def _interval_cover(N_obs, E_N, Var_N, level=0.90):
    lo_q = (1 - level) / 2
    over = Var_N > E_N + _EPS
    q_lo = np.empty_like(E_N); q_hi = np.empty_like(E_N)
    # NB where over-dispersed
    if over.any():
        m, v = E_N[over], Var_N[over]
        r = m**2 / (v - m)
        p = r / (r + m)
        q_lo[over] = nbinom.ppf(lo_q, r, p)
        q_hi[over] = nbinom.ppf(1 - lo_q, r, p)
    if (~over).any():                                          # Poisson fallback
        from scipy.stats import poisson
        m = np.maximum(E_N[~over], _EPS)
        q_lo[~over] = poisson.ppf(lo_q, m)
        q_hi[~over] = poisson.ppf(1 - lo_q, m)
    return (N_obs >= q_lo) & (N_obs <= q_hi)


# --------------------------------------------------------------------------- #
# top-level: score one saved holdout against rebuilt observations
# --------------------------------------------------------------------------- #
def score_holdout(npz_path, obs, *, n_draws=400, seed=0, mask=None):
    """Load a holdout fit, reconstruct its test set, return Scores (mode='predict').

    `obs` is a built Observations over the *full* dataset (data.build). Pass an
    explicit `mask` to override the meta-derived test set.
    """
    fit = load_fit(npz_path)
    ss = obs.ss
    name = _short(npz_path)
    if mask is None:
        mask, mode = test_mask(fit.meta, obs)
    else:
        mode = "predict"
    if mask.sum() == 0:
        return Scores(name=name, mode="influence", n=0, meta=dict(fit.meta))

    idx = _draw_idx(fit.samples.shape[0], n_draws, seed)
    M_draws = fit.samples[idx].astype(np.float64)
    r_draws = _r_per_col(fit, idx, ss)
    Xt, Y, D = obs.Xtilde[mask], obs.Y[mask], obs.D[mask]
    sc = score_slice(name, mode, Xt, Y, D, r_draws, M_draws,
                     extra={"n_src": obs.n_src[mask], "patient": obs.patient[mask],
                            "src_tp": obs.src_tp[mask].astype(int), **fit.meta})
    return sc


def _draw_idx(n_total, n, seed):
    if n is None or n >= n_total:
        return np.arange(n_total)
    return np.sort(np.random.default_rng(seed).choice(n_total, n, replace=False))


def _short(path):
    base = str(path).split("/")[-1].replace(".npz", "")
    return base.replace("nuts_fit_", "").replace("holdout_", "").replace("loPatient_", "")


# --------------------------------------------------------------------------- #
# influence: compare a withheld fit to the full fit on shared transitions
# --------------------------------------------------------------------------- #
@dataclass
class Influence:
    """Contribution of a withheld timepoint/group to the fitted operator.

    Sign convention: operator quantities are full - withheld, i.e. what the
    withheld data *adds* to the dynamics. For the T1 boundary holdout this is the
    pre-injection baseline's contribution: M_full mixes the naive baseline step
    (T1->T2) with the established post-injection dynamics, M_withheld is the pure
    post-injection operator, so (full - withheld) isolates how the baseline bends
    each trafficking route and growth rate.
    """
    name: str
    delta: dict                 # full - withheld predictive score on shared eval set
    rel_frobenius: float        # ||M_hat_full - M_hat_w|| / ||M_hat_full||
    z: np.ndarray               # [L,L] entrywise (M_full - M_w)/sd_full  (baseline contribution)
    t_row_jsd: np.ndarray       # [L] per-source-state JS(T_full row, T_w row)  (symmetric)
    g_log_ratio: np.ndarray     # [L] log(g_full / g_w)  (>0 => baseline raises growth)
    eval_summary_full: dict
    eval_summary_withheld: dict


def compare_fits(npz_withheld, npz_full, obs, *, eval_mask=None, n_draws=400, seed=0):
    """Influence panel for a boundary holdout: predictive delta + operator delta.

    eval_mask selects the shared transitions both fits trained on (default: the
    withheld fit's *training* set, i.e. everything not in its excluded source
    timepoints) -- a symmetric in-sample comparison isolating the withheld info.
    """
    fw = load_fit(npz_withheld); ff = load_fit(npz_full)
    ss = obs.ss
    if eval_mask is None:
        excl = set(int(x) for x in fw.meta.get("excluded_src_tp", []))
        eval_mask = ~np.isin(obs.src_tp.astype(int), list(excl)) if excl else np.ones(obs.Xtilde.shape[0], bool)

    Xt, Y, D = obs.Xtilde[eval_mask], obs.Y[eval_mask], obs.D[eval_mask]
    iw = _draw_idx(fw.samples.shape[0], n_draws, seed)
    iff = _draw_idx(ff.samples.shape[0], n_draws, seed)
    sw = score_slice("withheld", "predict", Xt, Y, D, _r_per_col(fw, iw, ss), fw.samples[iw].astype(np.float64))
    sf = score_slice("full", "predict", Xt, Y, D, _r_per_col(ff, iff, ss), ff.samples[iff].astype(np.float64))
    ssum_w, ssum_f = sw.summary(), sf.summary()
    delta = {k: ssum_f[k] - ssum_w[k] for k in
             ("elpd_mean", "shape_ll_mean", "dss_mean", "jsd_wmean", "coverage90")}

    # operator delta -- signed as the withheld timepoint's *contribution*
    # (full - withheld): for noT1 this is the pre-injection baseline's imprint.
    Mw, Mf = fw.M_hat, ff.M_hat
    sd_f = ff.samples.std(0)
    rel_fro = float(np.linalg.norm(Mf - Mw) / max(np.linalg.norm(Mf), _EPS))
    z = (Mf - Mw) / np.maximum(sd_f, _EPS)
    Tw, gw = decompose(Mw); Tf, gf = decompose(Mf)
    t_row_jsd = np.array([_js(Tf[i], Tw[i]) for i in range(ss.L)])
    g_log_ratio = np.log(np.maximum(gf, _EPS) / np.maximum(gw, _EPS))
    return Influence(name=_short(npz_withheld), delta=delta, rel_frobenius=rel_fro,
                     z=z, t_row_jsd=t_row_jsd, g_log_ratio=g_log_ratio,
                     eval_summary_full=ssum_f, eval_summary_withheld=ssum_w)


# --------------------------------------------------------------------------- #
# baselines, skill, and aggregate-marginal feasibility
#
# Absolute scores are unanchored -- "usable" is a *relative* property (Schiebinger
# 2019; forecast skill scores). We score two reference predictors and a noise
# ceiling, then report skill = how far the model moves from a null toward the
# best attainable:
#   static  M = identity: the depth-normalized source density is carried
#                forward unchanged (no trafficking, no growth) -- the "no change"
#                reference.
#   pooled  clone-agnostic: every clone predicted with the pooled population
#                destination composition, scaled to its source mass -- the
#                "marginal average" reference.
#   saturated    per-clone Poisson MLE (mean = observed): the achievable elpd
#                ceiling given count noise.
# Feasibility is also assessed at the aggregate-marginal level (the scientifically
# relevant scale) vs a replicate noise floor (JS between two random halves).
# --------------------------------------------------------------------------- #
def pooled_pi(obs, mask=None):
    """Pooled destination-state composition over `mask` (default all rows), [L]."""
    Y = obs.Y if mask is None else obs.Y[mask]
    s = Y.sum(0)
    return s / max(s.sum(), _EPS)


def baseline_mean(kind, Xt, D, pi=None, *, floor=1e-6):
    """Deterministic predicted-mean [n,L] for a null predictor (eps-floored).

    static  mean = D x~                      (M = identity; source carried forward unchanged)
    pooled  mean = D * (sum_z x~)(z) * pi    (pooled population-average destination shape)
    The floor keeps a null from being -inf-penalized at a state it assigns zero.
    """
    obsv = D > 0
    if kind == "static":
        nu = Xt
    elif kind == "pooled":
        if pi is None:
            raise ValueError("pooled needs pi")
        nu = Xt.sum(1, keepdims=True) * pi[None, :]
    else:
        raise ValueError(f"unknown baseline {kind!r}")
    return np.where(obsv, D * nu + floor, 0.0)


def _saturated_elpd(Y, D, r):
    """Per-clone NB saturated log-likelihood (mean = observed, concentration r): the ceiling.

    The best a NegBinomial(., r) model can score -- perfect mean aim under the SAME
    overdispersion the data carry, scored with the same r as the model and the nulls.
    (The Poisson saturated ceiling assumes only Poisson noise, so it is unreachable by an
    NB model and understates skill.) `r` broadcasts to Y's shape; y=0 states score 0.
    """
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    r = np.broadcast_to(np.asarray(r, float), Yt.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        lp = nbinom.logpmf(Yt, r, r / (r + Yt))          # log NB2(y; mean=y, r); = NB ceiling
    return np.where((Yt > 0) & obsv, lp, 0.0).sum(1)


def score_baseline(name, Y, D, mean, r, *, extra=None):
    """Score a deterministic predicted mean with the same metrics as the model."""
    n, L = Y.shape
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    N_obs = Yt.sum(1)
    r = np.broadcast_to(np.asarray(r), mean.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        pp = r / (r + mean)
        lp = nbinom.logpmf(Yt, r, pp)
    elpd = np.where(obsv, lp, 0.0).sum(1)
    tot = mean.sum(1)
    p = mean / np.maximum(tot[:, None], _EPS)
    shape_ll = (Yt * np.log(p + _EPS)).sum(1)
    Var_N = (mean + mean**2 / np.maximum(r, _EPS)).sum(1)
    dss = (N_obs - tot) ** 2 / np.maximum(Var_N, _EPS) + np.log(np.maximum(Var_N, _EPS))
    covered = _interval_cover(N_obs, tot, Var_N)
    c_obs = Yt / np.maximum(N_obs[:, None], _EPS)
    jsd = np.array([_js(c_obs[j], p[j]) if N_obs[j] > 0 else np.nan for j in range(n)])
    return Scores(name=name, mode="predict", n=n, elpd=elpd, shape_ll=shape_ll, dss=dss,
                  jsd=jsd, n_obs=N_obs, n_pred=tot, covered=covered,
                  weight=N_obs.astype(float), meta=extra or {})


def skill(model, static, pooled, oracle_elpd):
    """Skill of the model relative to each null (>0 => beats the null).

    skill_shape = 1 - JS_model / JS_null            (1 = perfect, 0 = no better)
    skill_elpd  = (elpd_model - elpd_null) / (elpd_oracle - elpd_null)
    """
    m, p, c = model.summary(), static.summary(), pooled.summary()
    em, ep, ec = m["elpd_mean"], p["elpd_mean"], c["elpd_mean"]
    jm, jp, jc = m["jsd_wmean"], p["jsd_wmean"], c["jsd_wmean"]

    def _se(null_e):
        d = oracle_elpd - null_e
        return float((em - null_e) / d) if d > _EPS else np.nan

    def _ss(null_j):
        return float(1 - jm / null_j) if null_j and np.isfinite(null_j) and null_j > 0 else np.nan

    return {
        "name": m["name"],
        "skill_shape_vs_static": _ss(jp), "skill_shape_vs_pooled": _ss(jc),
        "skill_elpd_vs_static": _se(ep), "skill_elpd_vs_pooled": _se(ec),
        "d_elpd_static": float(em - ep), "d_elpd_pooled": float(em - ec),
        "jsd_model": jm, "jsd_static": jp, "jsd_pooled": jc,
        "elpd_model": em, "elpd_static": ep, "elpd_pooled": ec,
        "elpd_oracle": float(oracle_elpd),
    }


def score_aggregate(name, Xt, Y, D, M_hat, pi, *, seed=0):
    """Aggregate-marginal feasibility: pooled predicted vs observed destination
    composition, model vs nulls vs a replicate noise floor (JS of random halves).
    """
    obsv = D > 0
    Yt = np.where(obsv, Y, 0.0)
    obs_agg = Yt.sum(0)
    obs_comp = obs_agg / max(obs_agg.sum(), _EPS)

    def _agg(mean):
        a = mean.sum(0)
        return a / max(a.sum(), _EPS), a.sum()

    cm, tm = _agg(np.where(obsv, D * (Xt @ M_hat), 0.0))
    cp, tp = _agg(baseline_mean("static", Xt, D))
    cc, tc = _agg(baseline_mean("pooled", Xt, D, pi))

    # replicate noise floor: JS between two random halves of the observed clones
    rng = np.random.default_rng(seed)
    perm = rng.permutation(Yt.shape[0])
    h1, h2 = perm[: len(perm) // 2], perm[len(perm) // 2:]
    a1 = Yt[h1].sum(0); a2 = Yt[h2].sum(0)
    js_floor = _js(a1 / max(a1.sum(), _EPS), a2 / max(a2.sum(), _EPS))

    obs_tot = obs_agg.sum()
    return {
        "name": name,
        "js_model": _js(obs_comp, cm), "js_static": _js(obs_comp, cp),
        "js_pooled": _js(obs_comp, cc), "js_noise_floor": float(js_floor),
        "total_ratio_model": float(tm / max(obs_tot, _EPS)),
        "total_ratio_static": float(tp / max(obs_tot, _EPS)),
    }


def evaluate_holdout(fit_path, obs, *, n_draws=400, seed=0):
    """Full feasibility evaluation of one predict-mode holdout.

    Returns None for influence-only (boundary) holdouts. Otherwise a dict with the
    model Scores, both null Scores, the skill dict, and the aggregate-marginal dict.
    """
    fit = load_fit(fit_path)
    ss = obs.ss
    name = _short(fit_path)
    mask, mode = test_mask(fit.meta, obs)
    if mask.sum() == 0:
        return None
    idx = _draw_idx(fit.samples.shape[0], n_draws, seed)
    M = fit.samples[idx].astype(np.float64)
    r = _r_per_col(fit, idx, ss)
    r_bar = r.mean(0)
    Xt, Y, D = obs.Xtilde[mask], obs.Y[mask], obs.D[mask]

    model = score_slice(name, "predict", Xt, Y, D, r, M,
                        extra={"n_src": obs.n_src[mask], "patient": obs.patient[mask],
                               "src_tp": obs.src_tp[mask].astype(int), **fit.meta})
    pi = pooled_pi(obs, ~mask)                       # population marginal, non-test
    static = score_baseline(name + "/static", Y, D, baseline_mean("static", Xt, D), r_bar)
    pooled = score_baseline(name + "/pooled", Y, D, baseline_mean("pooled", Xt, D, pi), r_bar)
    sk = skill(model, static, pooled, float(_saturated_elpd(Y, D, r_bar).mean()))
    agg = score_aggregate(name, Xt, Y, D, fit.M_hat, pi, seed=seed)
    return {"model": model, "static": static, "pooled": pooled,
            "skill": sk, "aggregate": agg}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_predictive(scores, path, *, title=None, skills=None):
    """4-panel predictive read-out across holdouts. `scores`: list[Scores].

    `skills` (optional): {name -> skill dict from `skill()`}. When provided, panel C
    shows the joint-score *skill toward the oracle* -- (elpd_model - null)/(elpd_oracle
    - null) -- instead of the raw elpd, since the raw log-density is always negative
    (even the oracle is) and is uninterpretable without the baseline/oracle reference.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sc = [s for s in scores if s.mode == "predict" and s.n > 0]
    names = [s.name for s in sc]
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(names)}

    fig, ax = plt.subplots(2, 2, figsize=(13, 11))

    # A) magnitude: predicted vs observed total (log-log), pooled, colored by holdout
    a = ax[0, 0]
    for s in sc:
        a.scatter(s.n_obs + 1, s.n_pred + 1, s=8, alpha=0.35, color=colors[s.name], label=s.name)
    lim = [1, max(2, max((s.n_obs.max() for s in sc), default=1) + 1)]
    a.plot(lim, lim, "k--", lw=1)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("observed destination total  N_j (+1)")
    a.set_ylabel("predicted total  E[N_j] (+1)")
    a.set_title("A. Magnitude: predicted vs observed total")
    a.legend(fontsize=7, markerscale=1.5, framealpha=0.6)

    # B) shape: abundance-weighted JS per holdout (box), survivors only
    b = ax[0, 1]
    bnames = [s.name for s in sc if np.isfinite(s.jsd).any()]
    data = [s.jsd[np.isfinite(s.jsd)] for s in sc if np.isfinite(s.jsd).any()]
    bp = b.boxplot(data, tick_labels=bnames, showfliers=False, patch_artist=True)
    for patch, n in zip(bp["boxes"], bnames):
        patch.set_facecolor(colors[n]); patch.set_alpha(0.5)
    b.set_ylabel("JS divergence (obs vs pred composition)")
    b.set_title("B. Shape: surviving-clone composition divergence (lower=better)")
    b.tick_params(axis="x", rotation=45, labelsize=8)

    # C) joint-score skill toward the oracle (vs nulls); raw elpd is always negative
    #    (so is the oracle), so without skills it is uninterpretable.
    c = ax[1, 0]
    have_skill = bool(skills) and all(skills.get(s.name) for s in sc)
    if have_skill:
        xs = np.arange(len(sc)); wbar = 0.4
        sp = [skills[s.name]["skill_elpd_vs_static"] for s in sc]
        scl = [skills[s.name]["skill_elpd_vs_pooled"] for s in sc]
        c.bar(xs - wbar / 2, sp, wbar, color="#3b6fb0", alpha=0.85, label="vs static (no change)")
        c.bar(xs + wbar / 2, scl, wbar, color="#e8820c", alpha=0.85, label="vs pooled pop-avg")
        c.axhline(0, color="k", lw=0.8)
        c.axhline(1, color="grey", ls=":", lw=1, label="oracle (perfect)")
        c.set_xticks(xs); c.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        c.set_ylabel("elpd skill  (elpd_model - null)/(elpd_oracle - null)")
        c.set_title("C. Joint-score skill toward oracle  (>0 beats null, 1 = oracle)")
        c.legend(fontsize=7)
    else:
        means = [s.elpd.mean() for s in sc]
        ses = [s.elpd.std() / np.sqrt(s.n) for s in sc]
        c.bar(range(len(sc)), means, yerr=ses, color=[colors[n] for n in names], alpha=0.7)
        c.set_xticks(range(len(sc))); c.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        c.set_ylabel("mean held-out elpd per clone")
        c.set_title("C. Joint log score (raw elpd; always <0 -- pass skills= for skill view)")

    # D) summary: shape accuracy vs magnitude calibration, one point per holdout
    d = ax[1, 1]
    for s in sc:
        ssum = s.summary()
        x = 1 - ssum["jsd_wmean"]                     # shape accuracy
        y = ssum["coverage90"]                        # magnitude calibration
        d.scatter(x, y, s=90, color=colors[s.name], edgecolor="k", zorder=3)
        d.annotate(s.name, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")
    d.axhline(0.90, ls=":", color="grey", lw=1, label="nominal 90% coverage")
    d.set_xlabel("shape accuracy  (1 - weighted JS)")
    d.set_ylabel("magnitude calibration  (90% interval coverage)")
    d.set_title("D. Summary: shape vs magnitude per holdout")
    d.legend(fontsize=7)

    fig.suptitle(title or "Held-out predictive accuracy", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def figure_influence(inf, ss, path, *, title=None):
    """3-panel influence read-out for a boundary holdout vs the full fit."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 1.0])

    # A) predictive delta on shared transitions (full - withheld)
    a = fig.add_subplot(gs[0, 0])
    keys = ["elpd_mean", "shape_ll_mean", "dss_mean", "jsd_wmean", "coverage90"]
    labs = ["elpd", "shape ll", "DSS", "JS(w)", "cov90"]
    vals = [inf.delta[k] for k in keys]
    a.bar(range(len(keys)), vals, color=["#4c72b0" if v >= 0 else "#c44e52" for v in vals], alpha=0.8)
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(range(len(keys))); a.set_xticklabels(labs, rotation=30, fontsize=8)
    a.set_ylabel("full  -  withheld")
    a.set_title(f"A. Predictive delta on shared data\n(rel.||M|| shift = {inf.rel_frobenius:.3f})")

    # B) entrywise z of (M_full - M_withheld)/sd_full = baseline contribution
    b = fig.add_subplot(gs[0, 1])
    vmax = float(np.nanpercentile(np.abs(inf.z), 99)) or 1.0
    im = b.imshow(inf.z, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    for k in range(1, ss.S):
        b.axhline(k * ss.K - 0.5, color="k", lw=0.6); b.axvline(k * ss.K - 0.5, color="k", lw=0.6)
    b.set_title("B. Operator contribution of withheld step\n(M_full - M_held, z vs full posterior sd)")
    b.set_xlabel("destination state"); b.set_ylabel("source state")
    fig.colorbar(im, ax=b, fraction=0.046, label="z  (red = raised by baseline)")

    # C) per-source-state read-out delta: T-row JS + g log-ratio
    c = fig.add_subplot(gs[0, 2])
    L = ss.L
    c.barh(np.arange(L) - 0.2, inf.t_row_jsd, height=0.4, color="#55a868", label="route shift  JS(T)")
    c.barh(np.arange(L) + 0.2, inf.g_log_ratio, height=0.4, color="#8172b3", label="growth  log(g_full/g_held)")
    c.axvline(0, color="k", lw=0.6)
    c.set_yticks([k * ss.K + ss.K / 2 for k in range(ss.S)])
    c.set_yticklabels(list(ss.tissues))
    c.set_title("C. Where baseline bends dynamics\nper source state")
    c.legend(fontsize=7)
    c.invert_yaxis()

    fig.suptitle(title or f"Baseline (withheld-step) contribution to the operator: {inf.name}",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return path


def figure_feasibility(results, path, *, title=None):
    """4-panel feasibility read-out: model vs null baselines, per-clone & aggregate.

    `results`: list of dicts from evaluate_holdout (model/static/pooled/
    skill/aggregate). The headline is whether the model beats both nulls and
    approaches the noise floor at the aggregate-marginal scale.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    res = [r for r in results if r is not None]
    names = [r["skill"]["name"] for r in res]
    x = np.arange(len(res))
    fig, ax = plt.subplots(2, 2, figsize=(14, 11))

    # A) per-clone shape: model vs static vs pooled (weighted JS, lower=better)
    a = ax[0, 0]
    jm = [r["skill"]["jsd_model"] for r in res]
    jp = [r["skill"]["jsd_static"] for r in res]
    jc = [r["skill"]["jsd_pooled"] for r in res]
    a.bar(x - 0.25, jm, 0.25, label="model", color="#4c72b0")
    a.bar(x, jp, 0.25, label="static", color="#c44e52", alpha=0.8)
    a.bar(x + 0.25, jc, 0.25, label="pooled population-average", color="#dd8452", alpha=0.8)
    a.set_xticks(x); a.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    a.set_ylabel("weighted JS (lower=better)")
    a.set_title("A. Per-clone shape: model vs nulls")
    a.legend(fontsize=8)

    # B) per-clone skill (>0 => beats null); shape & elpd vs static
    b = ax[0, 1]
    ss_p = [r["skill"]["skill_shape_vs_static"] for r in res]
    ss_c = [r["skill"]["skill_shape_vs_pooled"] for r in res]
    b.bar(x - 0.2, ss_p, 0.4, label="shape skill vs static", color="#55a868")
    b.bar(x + 0.2, ss_c, 0.4, label="shape skill vs pooled pop-avg", color="#8172b3", alpha=0.85)
    b.axhline(0, color="k", lw=0.8)
    b.set_xticks(x); b.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    b.set_ylabel("skill = 1 - JS_model/JS_null")
    b.set_title("B. Shape skill (>0 = adds value over null)")
    b.legend(fontsize=8)

    # C) AGGREGATE-marginal feasibility: JS(model/null vs observed) vs noise floor
    c = ax[1, 0]
    am = [r["aggregate"]["js_model"] for r in res]
    ap = [r["aggregate"]["js_static"] for r in res]
    acl = [r["aggregate"]["js_pooled"] for r in res]
    af = [r["aggregate"]["js_noise_floor"] for r in res]
    c.bar(x - 0.25, am, 0.25, label="model", color="#4c72b0")
    c.bar(x, ap, 0.25, label="static", color="#c44e52", alpha=0.8)
    c.bar(x + 0.25, acl, 0.25, label="pooled population-average", color="#dd8452", alpha=0.8)
    c.plot(x, af, "k_", ms=14, mew=2.5, label="replicate noise floor")
    c.set_xticks(x); c.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    c.set_ylabel("JS(predicted aggregate, observed)")
    c.set_title("C. Aggregate-marginal feasibility (model should approach floor)")
    c.legend(fontsize=8)

    # D) elpd skill toward the saturated ceiling (per-clone)
    d = ax[1, 1]
    se_p = [r["skill"]["skill_elpd_vs_static"] for r in res]
    se_c = [r["skill"]["skill_elpd_vs_pooled"] for r in res]
    d.bar(x - 0.2, se_p, 0.4, label="elpd skill vs static", color="#4c72b0")
    d.bar(x + 0.2, se_c, 0.4, label="elpd skill vs pooled pop-avg", color="#dd8452", alpha=0.85)
    d.axhline(0, color="k", lw=0.8)
    d.set_xticks(x); d.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    d.set_ylabel("(elpd_model - null) / (oracle - null)")
    d.set_title("D. elpd skill toward saturated ceiling")
    d.legend(fontsize=8)

    fig.suptitle(title or "Feasibility: model skill vs null baselines", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
