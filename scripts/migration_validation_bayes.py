"""Migration validation -- Bayesian sub-batch (posterior-sampling checks).

Three checks that use the posterior draws (not the M_hat point estimate) to say something
the plug-in cannot:

  entropy_decomposition : predictive destination uncertainty split into ALEATORIC (irreducible,
                          within-draw spread) and EPISTEMIC (reducible, disagreement ACROSS draws
                          = mutual information). Epistemic dominates where data is thin -> a
                          principled "where does the model not know" map. H[E p] = I(y;θ) + E H[p].
  contraction           : how much the posterior narrowed vs the Gamma(1,1) prior, per route
                          (1 - Var_post/Var_prior). ~1 = data-pinned; ~0 = prior-dominated.
  cluster_bootstrap     : CIs resampling PATIENTS (clusters) vs resampling CLONES. Clones are
                          correlated within patient, so clone bootstrap is anti-conservative;
                          the cluster CI is the honest one.

Env: TRAFFIC_H5 (required), FIT (default nb_global_putative.npz), OUTDIR, DRAWS (default 300).
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from predictability_breakdown import decompose_prediction, _js_rows, abundance_bin, ABUND_LABELS  # noqa: E402
from migration_validation import auc, boot_ci  # noqa: E402

from traffic import data, io, statespace  # noqa: E402
from traffic.readouts import decompose, tissue_traffic  # noqa: E402

RNG = np.random.default_rng(1)


def ent(p):
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log2(p), 0.0), axis=-1)


def cluster_ci(fn, patient, base_idx, n_boot=300, alpha=0.05):
    """(lo, hi) resampling PATIENTS with replacement (cluster bootstrap)."""
    pats = np.unique(patient[base_idx])
    groups = {pp: base_idx[patient[base_idx] == pp] for pp in pats}
    vals = []
    for _ in range(n_boot):
        samp = RNG.choice(pats, len(pats), replace=True)
        idx = np.concatenate([groups[pp] for pp in samp])
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def clone_ci(fn, base_idx, n_boot=300, alpha=0.05):
    vals = []
    n = base_idx.size
    for _ in range(n_boot):
        idx = base_idx[RNG.integers(0, n, n)]
        v = fn(idx)
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 10:
        return np.nan, np.nan
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5")
    fit_path = os.environ.get("FIT", "nb_global_putative.npz")
    outdir = os.environ.get("OUTDIR", "figures/validation/migration")
    ndraw = int(os.environ.get("DRAWS", "300"))
    os.makedirs(outdir, exist_ok=True)

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L; tis = list(ss.tissues)
    obs = data.build(data.load_obs_table(h5), ss)
    fit = io.load_fit(fit_path)
    Xt, Y, D = obs.Xtilde, obs.Y, obs.D; J = Xt.shape[0]
    phi = float(np.exp(np.asarray(fit.dispersion["params"]["log_r"]).mean())) if fit.dispersion else np.inf
    obsv = D > 0; Yt = np.where(obsv, Y, 0.0)
    pi = Yt.sum(0); pi = pi / max(pi.sum(), 1e-12)
    patient = np.asarray(obs.patient)
    n_src = obs.n_src.astype(int); abin = np.array([abundance_bin(n) for n in n_src])
    src_tis = Xt.reshape(J, S, K).sum(2).argmax(1)
    idx = np.sort(RNG.choice(fit.samples.shape[0], min(ndraw, fit.samples.shape[0]), replace=False))

    mean_model = np.where(obsv, D * (Xt @ fit.M_hat), 0.0)
    mean_pooled = np.where(obsv, D * (Xt.sum(1, keepdims=True) * pi[None, :]), 0.0)
    m = decompose_prediction(Xt, Y, D, mean_model, ss, phi)
    c = decompose_prediction(Xt, Y, D, mean_pooled, ss, phi)
    surv = m["surv"]
    print(f"J={J} phi={phi:.3f} draws={len(idx)} outdir={outdir}")

    # ===================== FIG A: aleatoric / epistemic entropy ================ #
    svv = np.where(surv)[0]
    Xs, Ds, ov = Xt[svv], D[svv], obsv[svv]
    ale = np.zeros(svv.size); pbar = np.zeros((svv.size, S))
    for si in idx:
        mu = np.where(ov, Ds * (Xs @ fit.samples[si]), 0.0).reshape(svv.size, S, K).sum(2)
        muf = mu / np.maximum(mu.sum(1, keepdims=True), 1e-12)
        ale += ent(muf); pbar += muf
    ale /= len(idx); pbar /= len(idx)
    tot = ent(pbar / np.maximum(pbar.sum(1, keepdims=True), 1e-12)); epi = np.maximum(tot - ale, 0.0)
    st = src_tis[svv]; ab = abin[svv]
    Hale = np.full((S, len(ABUND_LABELS)), np.nan); Hepi = np.full_like(Hale, np.nan); Hfrac = np.full_like(Hale, np.nan)
    for a in range(S):
        for j, lb in enumerate(ABUND_LABELS):
            sel = (st == a) & (ab == lb)
            if sel.sum() >= 5:
                Hale[a, j] = ale[sel].mean(); Hepi[a, j] = epi[sel].mean()
                Hfrac[a, j] = epi[sel].sum() / max(tot[sel].sum(), 1e-9)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for k, (Hm, ttl, cm, vmx) in enumerate([
            (Hale, "A. Aleatoric (irreducible) entropy", "magma", np.log2(S)),
            (Hepi, "B. Epistemic (reducible = data-limited) entropy", "viridis", None),
            (Hfrac, "C. Epistemic fraction  (1 = all uncertainty is data-limited)", "cividis", 1.0)]):
        vmx = vmx if vmx is not None else np.nanmax(Hepi) * 1.05
        im = ax[k].imshow(Hm, cmap=cm, vmin=0, vmax=vmx, aspect="auto")
        ax[k].set_yticks(range(S)); ax[k].set_yticklabels(tis); ax[k].set_xticks(range(len(ABUND_LABELS)))
        ax[k].set_xticklabels(ABUND_LABELS, fontsize=8); ax[k].set_xlabel("source clone size")
        for a in range(S):
            for j in range(len(ABUND_LABELS)):
                if np.isfinite(Hm[a, j]):
                    ax[k].text(j, a, f"{Hm[a, j]:.2f}", ha="center", va="center", fontsize=7,
                               color="w" if Hm[a, j] < vmx * 0.6 else "k")
        ax[k].set_title(ttl, fontsize=10); fig.colorbar(im, ax=ax[k], fraction=0.046)
    fig.suptitle("Destination-uncertainty decomposition (posterior draws): where is uncertainty irreducible vs data-limited?")
    fig.tight_layout(); fA = os.path.join(outdir, "entropy_decomposition.png"); fig.savefig(fA, dpi=140); plt.close(fig)

    # ===================== FIG B: posterior contraction ======================== #
    Xr = obs.X.reshape(J, S, K).sum(0); rho = Xr / np.maximum(Xr.sum(1, keepdims=True), 1e-12)
    Ppost = np.array([tissue_traffic(decompose(fit.samples[si])[0], ss, rho) for si in idx])
    Pprior = np.array([tissue_traffic(decompose(RNG.gamma(1.0, 1.0, (L, L)))[0], ss, rho) for _ in range(len(idx))])
    contr = 1 - Ppost.var(0) / np.maximum(Pprior.var(0), 1e-12)
    ent_post_var = fit.samples[idx].var(0)                       # [L,L], entry-level
    contr_entry = 1 - ent_post_var / 1.0                         # Gamma(1,1) prior var = 1
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    im0 = ax[0].imshow(contr, cmap="RdYlGn", vmin=0, vmax=1)
    ax[0].set_xticks(range(S)); ax[0].set_xticklabels(tis); ax[0].set_yticks(range(S)); ax[0].set_yticklabels(tis)
    ax[0].set_xlabel("destination"); ax[0].set_ylabel("source")
    for a in range(S):
        for b in range(S):
            ax[0].text(b, a, f"{contr[a, b]:.2f}", ha="center", va="center", fontsize=9,
                       color="k")
    ax[0].set_title("A. Route contraction 1 - Var_post/Var_prior\n(green = data-pinned, red = prior-dominated)")
    fig.colorbar(im0, ax=ax[0], fraction=0.046)
    im1 = ax[1].imshow(contr_entry, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for kk in range(1, S):
        ax[1].axhline(kk * K - 0.5, color="k", lw=0.5); ax[1].axvline(kk * K - 0.5, color="k", lw=0.5)
    ax[1].set_xticks([a * K + K / 2 for a in range(S)]); ax[1].set_xticklabels(tis)
    ax[1].set_yticks([a * K + K / 2 for a in range(S)]); ax[1].set_yticklabels(tis)
    ax[1].set_title("B. Entry-level contraction (39x39 M)\nblocks = tissue x tissue")
    fig.colorbar(im1, ax=ax[1], fraction=0.046)
    fig.suptitle("Posterior contraction: which migration structure the data actually informs")
    fig.tight_layout(); fB = os.path.join(outdir, "contraction.png"); fig.savefig(fB, dpi=140); plt.close(fig)

    # ===================== FIG C: cluster vs clone bootstrap =================== #
    obs_tm = Yt.reshape(J, S, K).sum(2); prd_tm = mean_model.reshape(J, S, K).sum(2)
    pf = prd_tm / np.maximum(prd_tm.sum(1, keepdims=True), 1e-12)
    pleave = 1 - pf[np.arange(J), src_tis]
    leave = (obs_tm[np.arange(J), src_tis] < obs_tm.sum(1) - 1e-9).astype(int)
    src_t = obs.X.reshape(J, S, K).sum(2); src_t = src_t / np.maximum(src_t.sum(1, keepdims=True), 1e-12)
    obs_t = obs_tm / np.maximum(obs_tm.sum(1, keepdims=True), 1e-12)
    mig_ch = np.full(J, np.nan); mig_ch[surv] = _js_rows(src_t[surv], obs_t[surv]); mov = surv & (mig_ch > 0.05)
    mig_m, mig_c = m["migration"], c["migration"]

    stats = [
        ("stay/go AUC", np.where(surv)[0], lambda ix: auc(pleave[ix], leave[ix])),
        ("migration skill\nvs pooled", np.where(mov)[0], lambda ix: 1 - np.median(mig_m[ix]) / max(np.median(mig_c[ix]), 1e-9)),
        ("observed\nleave rate", np.where(surv)[0], lambda ix: leave[ix].mean()),
    ]
    fig, ax = plt.subplots(figsize=(9, 5)); xs = np.arange(len(stats)); w = 0.3
    for k, (name, base, fn) in enumerate(stats):
        pt = fn(base)
        cl_lo, cl_hi = clone_ci(fn, base); cu_lo, cu_hi = cluster_ci(fn, patient, base)
        ax.errorbar(k - w / 2, pt, yerr=[[pt - cl_lo], [cl_hi - pt]], fmt="o", color="#4c72b0", capsize=5,
                    label="clone bootstrap" if k == 0 else None)
        ax.errorbar(k + w / 2, pt, yerr=[[pt - cu_lo], [cu_hi - pt]], fmt="s", color="#c44e52", capsize=5,
                    label="patient cluster bootstrap" if k == 0 else None)
        ax.text(k, pt, f"  {pt:.2f}", fontsize=8, va="bottom")
    ax.set_xticks(xs); ax.set_xticklabels([s[0] for s in stats]); ax.legend()
    ax.set_ylabel("statistic (95% CI)")
    ax.set_title("Cluster (patient) vs clone bootstrap: clone CIs are too tight\n(clones correlate within patient -> resample patients)")
    fig.tight_layout(); fC = os.path.join(outdir, "cluster_bootstrap.png"); fig.savefig(fC, dpi=140); plt.close(fig)

    print(f"wrote:\n  {fA}\n  {fB}\n  {fC}")


if __name__ == "__main__":
    main()
