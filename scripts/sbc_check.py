"""Simulation-Based Calibration harness (Goal 2). Runs R rounds of draw-theta*~prior -> simulate ->
fit -> rank, then reports rank uniformity (Sailynoja ECDF-difference simultaneous bands + rank
histograms) and rank-implied central-interval coverage over the science scalars.

Two tiers via env vars (same script):
  CI GATE (default): reduced scale K=4 -> L=12, few rounds, short chains, hard asserts. Fast smoke
    that the harness runs end-to-end and is not grossly miscalibrated -- NOT a powered uniformity test.
  PAPER: K=13 -> L=39, R=500+, long chains (Phase 3: W=NS~=1000 for r_hat<1.01 & ESS to thin),
    GATE=0 (figures only). Reduced scale keeps S=3 fixed (fit_nuts assumes S=3) and shrinks K, so no
    core change is needed.

Run (gate):  python -u scripts/sbc_check.py
Run (paper): K=13 J=2000 R=500 W=1000 NS=1000 GATE=0 OUTDIR=figures/validation/sbc python -u scripts/sbc_check.py
Env: K, J, R, W, NS, L_POST, SEED, TA, GATE, RHAT_GATE, DIV_GATE, OUTDIR.
"""
import json
import os
import sys
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import sbc
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.statespace import StateSpace


def _chi2_uniform_p(ranks01, n_bins=20):
    from scipy.stats import chi2 as chi2dist
    counts, _ = np.histogram(ranks01, bins=n_bins, range=(0, 1))
    exp = ranks01.size / n_bins
    return float(chi2dist.sf(((counts - exp) ** 2 / exp).sum(), n_bins - 1))


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    S = 3                                                   # fit_nuts assumes S=3; reduce K to shrink L
    K = int(os.environ.get("K", "4"))
    ss = StateSpace(tissues=tuple(f"T{i}" for i in range(S)),
                    phenotypes=tuple(f"p{i}" for i in range(K)))
    L = ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0,
                                beta=1.0, sigma_phi=1.0)
    J = int(os.environ.get("J", "300"))
    R = int(os.environ.get("R", "24"))
    W = int(os.environ.get("W", "250")); NS = int(os.environ.get("NS", "250"))
    L_post = int(os.environ.get("L_POST", "99"))
    seed = int(os.environ.get("SEED", "0"))
    ta = float(os.environ.get("TA", "0.9"))
    gate = os.environ.get("GATE", "1") == "1"
    rhat_gate = float(os.environ.get("RHAT_GATE", "1.1"))
    div_gate = float(os.environ.get("DIV_GATE", "0.02"))
    outdir = os.environ.get("OUTDIR", "figures/validation/sbc")
    os.makedirs(outdir, exist_ok=True)
    base = MCMCConfig(num_warmup=W, num_samples=NS, num_chains=2, target_accept=ta)
    ndraws = NS * base.num_chains
    levels = [0.5, 0.8, 0.9, 0.95]

    print(f"SBC: S={S} K={K} L={L} J={J} R={R} W={W} NS={NS} L_post={L_post} ta={ta} "
          f"gate={gate}", flush=True)
    round_keys = jax.random.split(jax.random.PRNGKey(seed), R)
    names = None
    rows, ess_rows, meta_rows, fails = [], [], [], 0
    for r in range(R):
        cfg_r = replace(base, seed=r)
        try:
            out = sbc.sbc_round(round_keys[r], prior, cfg_r, ss, J, L_post)
        except Exception as e:
            fails += 1
            print(f"  round {r}: FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
            continue
        if names is None:
            names = list(out["ranks"].keys())
        rows.append([out["ranks"][n] for n in names])
        ess_rows.append([out["ess"][n] for n in names])
        meta_rows.append((out["L_post"], out["r_hat_max"], out["num_divergences"], out["phi"]))
        print(f"  round {r}: r_hat={out['r_hat_max']:.3f} div={out['num_divergences']} "
              f"phi={out['phi']:.2f} L_post={out['L_post']} min_ess={min(out['ess'].values()):.0f}",
              flush=True)

    if not rows:
        raise SystemExit(f"all {R} SBC rounds failed -- check the model/simulator before proceeding")
    ranks = np.array(rows); ess = np.array(ess_rows)
    L_posts = np.array([m[0] for m in meta_rows]); rhats = np.array([m[1] for m in meta_rows])
    divs = np.array([m[2] for m in meta_rows]); phis = np.array([m[3] for m in meta_rows])
    kept = (rhats < rhat_gate) & (divs < div_gate * ndraws)
    Lp = int(np.median(L_posts))
    print(f"\nrounds: {len(rows)} ok, {fails} failed; kept {int(kept.sum())}/{len(rows)} after "
          f"convergence filter (r_hat<{rhat_gate}, div_rate<{div_gate})", flush=True)
    if kept.sum() < len(rows):
        dropped_phi = phis[~kept]
        print(f"  dropped rounds phi: {np.round(dropped_phi, 2).tolist()} "
              f"(median kept phi={np.median(phis[kept]):.2f}) -- watch for theta*-hardness bias", flush=True)

    rk = ranks[kept]; ess_k = ess[kept]
    ranks01 = rk / Lp
    P = len(names)
    cov = sbc.interval_coverage(ranks01, levels)                 # {q: [P]}
    pooled = ranks01.ravel()
    cov_pooled = sbc.interval_coverage(pooled, levels)
    p_unif = _chi2_uniform_p(pooled)
    print(f"pooled coverage: " + "  ".join(f"{int(q*100)}%={cov_pooled[q]:.2f}" for q in levels), flush=True)
    print(f"pooled rank uniformity chi2 p={p_unif:.3f}   min per-scalar ESS={ess_k.min():.0f}", flush=True)

    # ---- results ----
    np.savez(os.path.join(outdir, "sbc_results.npz"),
             ranks=rk, L_post=Lp, scalar_names=np.array(names), ess=ess_k,
             r_hat_max=rhats[kept], num_divergences=divs[kept], phi=phis[kept],
             levels=np.array(levels), coverage=np.array([cov[q] for q in levels]))
    with open(os.path.join(outdir, "sbc_meta.json"), "w") as fh:
        json.dump({"S": S, "K": K, "L": L, "J": J, "R": R, "W": W, "NS": NS, "L_post": Lp,
                   "target_accept": ta, "n_failed": fails, "n_dropped": int((~kept).sum()),
                   "prior": prior.__dict__, "scalar_names": names}, fh, indent=2, default=str)

    # ---- figures ----
    ncol = 4; nrow = int(np.ceil(P / ncol))
    exp, blo, bhi = sbc.rank_hist_expected_band(kept.sum(), 10)
    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow))
    for i, nm in enumerate(names):
        a = ax.flat[i]
        a.hist(ranks01[:, i], bins=10, range=(0, 1), color="#4c72b0", alpha=.8)
        a.axhspan(blo, bhi, color="grey", alpha=.25); a.axhline(exp, color="k", lw=.6)
        a.set_title(nm, fontsize=8); a.set_yticks([])
    for j in range(P, nrow * ncol):
        ax.flat[j].axis("off")
    fig.suptitle(f"SBC rank histograms (grey = 99% uniform band, {int(kept.sum())} rounds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sbc_rank_hist.png"), dpi=140); plt.close(fig)

    fig, ax = plt.subplots(nrow, ncol, figsize=(3.4 * ncol, 2.6 * nrow))
    for i, nm in enumerate(names):
        a = ax.flat[i]
        grid, obs, lo, hi = sbc.ecdf_diff_band(rk[:, i], Lp, n_sim=1000, alpha=0.05, seed=1)
        a.fill_between(grid, lo, hi, color="grey", alpha=.3)
        a.plot(grid, obs, color="#c44e52", lw=1); a.axhline(0, color="k", lw=.5)
        a.set_title(nm, fontsize=8)
    for j in range(P, nrow * ncol):
        ax.flat[j].axis("off")
    fig.suptitle("SBC ECDF-difference with 95% simultaneous band (Sailynoja); inside = calibrated")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sbc_ecdf_bands.png"), dpi=140); plt.close(fig)

    fig, a = plt.subplots(figsize=(6, 6))
    for i in range(P):
        a.plot(levels, [cov[q][i] for q in levels], color="#bbbbbb", lw=.8, marker=".", alpha=.6)
    a.plot(levels, [cov_pooled[q] for q in levels], color="#c44e52", lw=2, marker="o", label="pooled")
    a.plot([0, 1], [0, 1], "k:", lw=1)
    a.set_xlabel("nominal central interval"); a.set_ylabel("empirical coverage")
    a.set_xlim(0.4, 1.0); a.set_ylim(0.3, 1.0); a.legend()
    a.set_title(f"Interval coverage (grey = per scalar, red = pooled; {int(kept.sum())} rounds)")
    fig.tight_layout(); fig.savefig(os.path.join(outdir, "sbc_coverage.png"), dpi=140); plt.close(fig)
    print("wrote figures + sbc_results.npz to", outdir, flush=True)

    # ---- gate (smoke, not a powered uniformity test at CI scale) ----
    if gate:
        print("\ngates: fails=0, kept>=0.7R, ranks in [0,L_post], pooled cov90 in [0.80,0.98] & "
              "cov50 in [0.35,0.65], pooled uniformity p>1e-4", flush=True)
        assert fails == 0, f"{fails} rounds raised"
        assert kept.sum() >= 0.7 * R, f"only {int(kept.sum())}/{R} rounds converged"
        assert rk.min() >= 0 and rk.max() <= Lp, "ranks out of [0, L_post]"
        assert 0.80 <= cov_pooled[0.9] <= 0.98, f"pooled cov90={cov_pooled[0.9]:.2f}"
        assert 0.35 <= cov_pooled[0.5] <= 0.65, f"pooled cov50={cov_pooled[0.5]:.2f}"
        assert p_unif > 1e-4, f"pooled ranks grossly non-uniform (p={p_unif:.2e})"
        print("SBC GATE PASS", flush=True)


if __name__ == "__main__":
    main()
