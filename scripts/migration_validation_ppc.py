"""Migration validation -- posterior-predictive (PPC) structural menu.

Goes past the pooled tissue marginal to test structure it cannot see. Replicate datasets
are drawn from the posterior predictive (pick draw s -> mu = D (x~ M_s) -> y_rep ~ NB2(mu, phi)):

  ppc_traffic_matrix : source-conditioned destination distribution (the 3x3), observed vs the
                       replicate band, p-value per route -- goodness-of-fit conditioned on source.
  ppc_tissue_spread  : distribution of #tissues a clone occupies at t+1 (0/1/2/3) -- multi-homing
                       structure the marginal is blind to.
  ppc_cooccurrence   : fraction of clones present in BOTH tissue a and b -- the dependence structure.
  ppc_pit            : randomized-PIT / rank histogram of observed counts under the posterior
                       predictive -- U-shape = underdispersed, hump = overdispersed (tests NB phi).
  loo_influence      : pointwise WAIC (elpd + p_waic per clone) -- which clones are hard/influential
                       (PSIS-LOO / Pareto-k would refine this).

Env: TRAFFIC_H5 (required), FIT, OUTDIR, N_REP (default 200), DRAWS (default 200).
"""
import os
import sys

import numpy as np
from scipy.special import logsumexp
from scipy.stats import nbinom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from traffic import data, io, statespace  # noqa: E402

RNG = np.random.default_rng(3)


def ppval(reps, obs):
    return 2 * min((reps >= obs).mean(), (reps <= obs).mean())


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/migration")
    n_rep = int(os.environ.get("N_REP", "200"))
    ndraw = int(os.environ.get("DRAWS", "200"))
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L; tis = list(ss.tissues)
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D; J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    obsv = D > 0; Yt = np.where(obsv, Y, 0.0)
    src_tis = obs.src_tissue   # raw-count attribution (not Xtilde); see docs/DATA.md
    idx = np.sort(RNG.choice(fit.samples.shape[0], min(ndraw, fit.samples.shape[0]), replace=False))
    print(f"J={J} phi={phi:.3f} reps={n_rep} draws={len(idx)}")

    # observed structural stats
    obs_tm = Yt.reshape(J, S, K).sum(2)                       # [J,S] dest tissue counts
    def traffic(tm):
        P = np.zeros((S, S))
        for a in range(S):
            sel = src_tis == a
            v = tm[sel].sum(0); P[a] = v / max(v.sum(), 1e-12)
        return P
    def spread(tm):
        n = (tm > 0).sum(1); return np.array([(n == k).mean() for k in range(S + 1)])
    pairs = [(0, 1), (0, 2), (1, 2)]
    def cooc(tm):
        pres = tm > 0
        return np.array([(pres[:, a] & pres[:, b]).mean() for a, b in pairs])
    P_obs, spr_obs, coo_obs = traffic(obs_tm), spread(obs_tm), cooc(obs_tm)

    # replicate simulation (reuse across figs 1-3)
    Ptr = np.empty((n_rep, S, S)); Spr = np.empty((n_rep, S + 1)); Coo = np.empty((n_rep, len(pairs)))
    for i in range(n_rep):
        s = idx[i % len(idx)]
        mu = np.where(obsv, D * (Xt @ fit.samples[s]), 0.0)
        pp = np.clip(phi / (phi + mu), 1e-9, 1.0)
        yr = np.where(obsv, RNG.negative_binomial(np.maximum(phi, 1e-6), pp), 0)
        tmr = yr.reshape(J, S, K).sum(2)
        Ptr[i] = traffic(tmr); Spr[i] = spread(tmr); Coo[i] = cooc(tmr)

    # ---- FIG 1: source-conditioned traffic-matrix PPC ---- #
    fig, ax = plt.subplots(1, S, figsize=(15, 4.6), sharey=True)
    for a in range(S):
        lo = np.quantile(Ptr[:, a, :], 0.025, 0); hi = np.quantile(Ptr[:, a, :], 0.975, 0); md = Ptr[:, a, :].mean(0)
        xs = np.arange(S)
        ax[a].bar(xs, md, 0.6, color="#4c72b0", alpha=0.5,
                  yerr=[md - lo, hi - md], capsize=4, label="replicate 95%")
        ax[a].plot(xs, P_obs[a], "rD", ms=9, label="observed")
        for b in range(S):
            ax[a].text(b, max(P_obs[a, b], md[b]) + 0.02, f"p={ppval(Ptr[:, a, b], P_obs[a, b]):.2f}",
                       ha="center", fontsize=8)
        ax[a].set_xticks(xs); ax[a].set_xticklabels(tis); ax[a].set_title(f"source = {tis[a]}"); ax[a].set_xlabel("destination")
    ax[0].set_ylabel("fraction of descendant mass"); ax[0].legend(fontsize=8)
    fig.suptitle("PPC: source-conditioned traffic matrix (observed vs posterior-predictive replicates; p per route)")
    fig.tight_layout(); f1 = os.path.join(outdir, "ppc_traffic_matrix.png"); fig.savefig(f1, dpi=140); plt.close(fig)

    # ---- FIG 2: tissue-spread PPC ---- #
    fig, ax = plt.subplots(figsize=(8, 5)); xs = np.arange(S + 1)
    lo = np.quantile(Spr, 0.025, 0); hi = np.quantile(Spr, 0.975, 0); md = Spr.mean(0)
    ax.bar(xs, md, 0.6, color="#55a868", alpha=0.5, yerr=[md - lo, hi - md], capsize=4, label="replicate 95%")
    ax.plot(xs, spr_obs, "rD", ms=10, label="observed")
    for k in range(S + 1):
        ax.text(k, max(spr_obs[k], md[k]) + 0.01, f"p={ppval(Spr[:, k], spr_obs[k]):.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in xs]); ax.set_xlabel("# tissues occupied at t+1")
    ax.set_ylabel("fraction of clones"); ax.legend()
    ax.set_title("PPC: tissue spread (multi-homing) -- does the model get dispersal breadth right?")
    fig.tight_layout(); f2 = os.path.join(outdir, "ppc_tissue_spread.png"); fig.savefig(f2, dpi=140); plt.close(fig)

    # ---- FIG 3: co-occurrence PPC ---- #
    fig, ax = plt.subplots(figsize=(8, 5)); xs = np.arange(len(pairs))
    lo = np.quantile(Coo, 0.025, 0); hi = np.quantile(Coo, 0.975, 0); md = Coo.mean(0)
    ax.bar(xs, md, 0.5, color="#8172b3", alpha=0.5, yerr=[md - lo, hi - md], capsize=4, label="replicate 95%")
    ax.plot(xs, coo_obs, "rD", ms=10, label="observed")
    for k, (a, b) in enumerate(pairs):
        ax.text(k, max(coo_obs[k], md[k]) + 0.002, f"p={ppval(Coo[:, k], coo_obs[k]):.2f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels([f"{tis[a]}&{tis[b]}" for a, b in pairs]); ax.set_xlabel("tissue pair")
    ax.set_ylabel("fraction of clones present in both"); ax.legend()
    ax.set_title("PPC: tissue co-occurrence -- the dependence structure the marginal misses")
    fig.tight_layout(); f3 = os.path.join(outdir, "ppc_cooccurrence.png"); fig.savefig(f3, dpi=140); plt.close(fig)

    # ---- FIG 4: randomized-PIT / rank histogram (dispersion) ---- #
    jj, zz = np.where(obsv)
    sub = RNG.choice(jj.size, min(8000, jj.size), replace=False)
    js, zsub = jj[sub], zz[sub]; ys = Yt[js, zsub]
    Fy = np.zeros(sub.size); Fy1 = np.zeros(sub.size)
    for s in idx:
        mu = (D[js, zsub] * (Xt[js] @ fit.samples[s])[np.arange(sub.size), zsub])
        p = np.clip(phi / (phi + np.maximum(mu, 1e-9)), 1e-9, 1.0)
        Fy += nbinom.cdf(ys, phi, p); Fy1 += nbinom.cdf(ys - 1, phi, p)
    Fy /= len(idx); Fy1 /= len(idx)
    u = Fy1 + RNG.random(sub.size) * (Fy - Fy1)                # randomized PIT
    fig, ax = plt.subplots(figsize=(8, 5)); nb = 20
    cnt, edges, _ = ax.hist(u, bins=nb, color="#4c72b0", alpha=0.7, edgecolor="w")
    exp = sub.size / nb
    ax.axhline(exp, color="k", ls="--", label="uniform (calibrated)")
    ax.fill_between([0, 1], exp - 2 * np.sqrt(exp), exp + 2 * np.sqrt(exp), color="gray", alpha=0.2, label="±2σ band")
    ax.set_xlabel("randomized PIT of observed count"); ax.set_ylabel("count")
    ax.set_title("PPC: rank histogram (U = underdispersed, hump = overdispersed; flat = NB φ calibrated)")
    ax.legend()
    fig.tight_layout(); f4 = os.path.join(outdir, "ppc_pit.png"); fig.savefig(f4, dpi=140); plt.close(fig)

    # ---- FIG 5: pointwise WAIC influence (LOO-style) ---- #
    csub = RNG.choice(J, min(6000, J), replace=False)
    Xc, Dc, oc, Yc = Xt[csub], D[csub], obsv[csub], Yt[csub]
    ll = np.empty((len(idx), csub.size))
    for k, s in enumerate(idx):
        mu = np.where(oc, Dc * (Xc @ fit.samples[s]), 0.0)
        p = np.clip(phi / (phi + mu), 1e-9, 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            lp = nbinom.logpmf(Yc, phi, p)
        ll[k] = np.where(oc, lp, 0.0).sum(1)
    lppd = logsumexp(ll, axis=0) - np.log(len(idx))            # per-clone log predictive density
    p_waic = ll.var(0)                                          # per-clone effective complexity / influence
    elpd_waic = float((lppd - p_waic).sum() / csub.size)       # per-clone mean
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].hist(lppd, bins=60, color="#4c72b0", alpha=0.8)
    ax[0].axvline(np.median(lppd), color="k", ls="--", label=f"median {np.median(lppd):.2f}")
    ax[0].set_xlabel("per-clone log predictive density (lppd)"); ax[0].set_ylabel("clones")
    ax[0].set_title("A. Predictability per clone (left tail = hard clones)"); ax[0].legend()
    ax[1].hist(p_waic, bins=60, color="#c44e52", alpha=0.8)
    thr = np.quantile(p_waic, 0.99)
    ax[1].axvline(thr, color="k", ls="--", label=f"99th pct {thr:.2f}")
    ax[1].set_xlabel("per-clone p_waic (influence / effective complexity)"); ax[1].set_ylabel("clones")
    ax[1].set_title(f"B. Influence (high = fragile/outlier);  elpd_waic/clone={elpd_waic:.2f}"); ax[1].legend()
    fig.suptitle("Pointwise WAIC influence on a clone subsample (PSIS-LOO / Pareto-k would refine)")
    fig.tight_layout(); f5 = os.path.join(outdir, "loo_influence.png"); fig.savefig(f5, dpi=140); plt.close(fig)

    print("wrote:")
    for f in [f1, f2, f3, f4, f5]:
        print("  " + f)
    print(f"\nPPC p-values (traffic diag): " + "  ".join(f"{tis[a]}->{tis[a]}={ppval(Ptr[:,a,a],P_obs[a,a]):.2f}" for a in range(S)))
    print(f"spread p: " + "  ".join(f"{k}tis={ppval(Spr[:,k],spr_obs[k]):.2f}" for k in range(S + 1)))


if __name__ == "__main__":
    main()
