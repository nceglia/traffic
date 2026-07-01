"""Forward forecast of the trafficking dynamics past the observed window (T1..T6).

The per-transition fits (temporal_Mr.npz) show the operator still drifting mildly at T6.
This projects the SMOOTHED stabilization the model implies: iterate the row-stochastic
redistribution operator T (share-supported -- redistribution only, no magnitude; see
traffic/readouts.py CONTRACT) forward from the observed T1 composition, driving the
OBSERVED window with the actual per-transition operators and then FORECASTING past T6
under the late (T5->T6) dynamics, until the composition converges to steady state.

Assumption for the forecast leg: the late (T5->T6) operator persists (time-homogeneous
from T6 on). Drift is decreasing but non-zero at T6, so read the forecast as "if current
dynamics hold". Posterior bands propagate the 400 saved operator draws.

Panels:
  A  tissue occupancy: observed (dots) + model trajectory & forecast (line+band) -> steady (dashed)
  B  the most-shifting phenotype fractions -> steady
  C  distance-to-equilibrium (total variation to the forecast steady state) with 10%/1% marks

Env: FIT_TEMPORAL (npz, default temporal_Mr.npz), TRAFFIC_H5 (required, for the T1 start),
     OUTDIR (default .), HORIZON (forecast steps past T6, default 12)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traffic import data, statespace
from traffic.readouts import decompose

_PH_SHORT = {
    "CD8_Activated_TEMRA": "TEMRA", "CD8_Activated_TEXeff": "TEXeff",
    "CD8_Activated_TEXterm": "TEXterm", "CD8_Activated_TRM": "TRM",
    "CD8_Quiescent_Memory": "Mem", "CD8_Quiescent_Naive": "Naive",
    "CD8_Quiescent_TEXprog": "TEXprog", "CD4_Naive": "4Naive",
    "CD4_Activated": "4Act", "CD4_Quiescent_Exhausted": "4QExh",
    "CD4_Exhausted": "4Exh", "CD4_Treg": "Treg", "CD4_Th": "Th",
}


def _T(M):
    return decompose(M)[0]


def _iterate(c0, ops):
    """Row-vector composition through a sequence of row-stochastic operators. [len(ops)+1, L]."""
    out = [c0]
    for Op in ops:
        out.append(out[-1] @ Op)
    return np.array(out)


def _steady(T):
    vals, vecs = np.linalg.eig(np.asarray(T).T)
    i = int(np.argmin(np.abs(vals - 1.0)))
    pi = np.abs(np.real(vecs[:, i]))
    return pi / pi.sum()


def main():
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    tis = list(ss.tissues); ph = [_PH_SHORT.get(str(p), str(p)) for p in ss.phenotypes]
    fit = os.environ.get("FIT_TEMPORAL", "temporal_Mr.npz")
    h5 = os.environ.get("TRAFFIC_H5")
    if not h5:
        raise SystemExit("set TRAFFIC_H5 (needed for the observed composition trajectory)")
    outdir = os.environ.get("OUTDIR", "."); os.makedirs(outdir, exist_ok=True)
    H = int(os.environ.get("HORIZON", "12"))

    d = np.load(fit, allow_pickle=True)
    M_hat, samples, src_tp = d["M_hat"], d["samples"], d["src_tp"].astype(int)
    R = len(src_tp)                                              # transitions (T1->T2 ... T5->T6)

    obs = data.build(data.load_obs_table(h5), ss)
    stp = obs.src_tp.astype(int)
    # observed pooled depth-normalized composition per timepoint (share space)
    def pooled_src(t):
        m = stp == t; v = obs.Xtilde[m].sum(0); return v / max(v.sum(), 1e-12)
    c_obs = np.array([pooled_src(t) for t in src_tp])           # [R, L] at T1..T5 (source)
    # T(R+1) composition = destination of the last transition
    mlast = stp == src_tp[-1]
    Dl = obs.D[mlast]; Yl = np.where(Dl > 0, obs.Y[mlast], 0.0)
    vlast = (Yl / np.maximum(Dl, 1e-12)).sum(0); c_last = vlast / max(vlast.sum(), 1e-12)
    c_obs = np.vstack([c_obs, c_last])                          # [R+1, L] observed T1..T6
    tp_labels = [f"T{t}" for t in src_tp] + [f"T{src_tp[-1]+1}"]

    def trajectory(Ms):
        """observed window driven by per-transition ops, then forecast under the last op."""
        ops = [_T(Ms[r]) for r in range(R)]
        obs_traj = _iterate(c_obs[0], ops)                      # [R+1, L] model over observed window
        Tlate = ops[-1]
        fore = _iterate(obs_traj[-1], [Tlate] * H)[1:]          # [H, L] past T6
        return np.vstack([obs_traj, fore]), _steady(Tlate)      # [R+1+H, L], pi

    traj_pt, pi_pt = trajectory(M_hat)
    dr = min(200, samples.shape[1])
    traj_draws = np.array([trajectory(samples[:, s])[0] for s in range(dr)])   # [dr, R+1+H, L]
    n_t = traj_pt.shape[0]
    x = np.arange(n_t); x_obs = np.arange(R + 1)

    def occ(c):
        return np.array([c[..., a * K:(a + 1) * K].sum(-1) for a in range(S)])  # [S, ...]

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    cols = ["#c44e52", "#4c72b0", "#8172b3"]

    # A: tissue occupancy
    occ_pt = occ(traj_pt)                                       # [S, n_t]
    occ_dr = np.array([occ(traj_draws[s]) for s in range(dr)])  # [dr, S, n_t]
    occ_obs = occ(c_obs)                                        # [S, R+1]
    pi_occ = np.array([pi_pt[a * K:(a + 1) * K].sum() for a in range(S)])
    for a in range(S):
        lo = np.quantile(occ_dr[:, a, :], .025, 0); hi = np.quantile(occ_dr[:, a, :], .975, 0)
        ax[0].fill_between(x, lo, hi, color=cols[a], alpha=.15)
        ax[0].plot(x, occ_pt[a], color=cols[a], lw=2, label=tis[a])
        ax[0].plot(x_obs, occ_obs[a], "o", color=cols[a], ms=6, mec="k", mew=.5)
        ax[0].axhline(pi_occ[a], color=cols[a], ls=":", lw=1)
    ax[0].axvline(R, color="k", ls="--", lw=1); ax[0].text(R + .1, .02, "forecast ->", fontsize=8)
    ax[0].set_xticks(x); ax[0].set_xticklabels(tp_labels + [f"+{i}" for i in range(1, H + 1)], fontsize=7, rotation=90)
    ax[0].set_ylabel("tissue occupancy (share)"); ax[0].set_title("A. Tissue occupancy: observed (dots) + forecast (line) -> steady (dotted)")
    ax[0].legend(fontsize=8)

    # B: most-shifting phenotype fractions (overall, top movers across the trajectory)
    ph_pt = traj_pt                                             # [n_t, L]
    swing = ph_pt.max(0) - ph_pt.min(0)
    top = np.argsort(-swing)[:6]
    for z in top:
        a, u = z // K, z % K
        lo = np.quantile(traj_draws[:, :, z], .025, 0); hi = np.quantile(traj_draws[:, :, z], .975, 0)
        ax[1].fill_between(x, lo, hi, alpha=.12)
        ax[1].plot(x, ph_pt[:, z], lw=1.6, label=f"{tis[a]}:{ph[u]}")
        ax[1].axhline(pi_pt[z], color="gray", ls=":", lw=.6)
    ax[1].axvline(R, color="k", ls="--", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(tp_labels + [f"+{i}" for i in range(1, H + 1)], fontsize=7, rotation=90)
    ax[1].set_ylabel("state share"); ax[1].set_title("B. Most-shifting states -> steady"); ax[1].legend(fontsize=7)

    # C: distance-to-equilibrium (total variation to forecast steady state)
    tv_dr = 0.5 * np.abs(traj_draws - pi_pt[None, None, :]).sum(-1)   # [dr, n_t]
    tv_pt = 0.5 * np.abs(traj_pt - pi_pt[None, :]).sum(-1)
    lo = np.quantile(tv_dr, .025, 0); hi = np.quantile(tv_dr, .975, 0)
    ax[2].fill_between(x, lo, hi, color="#55a868", alpha=.2)
    ax[2].plot(x, tv_pt, "-o", color="#2e6b3e", lw=2)
    tv0 = tv_pt[0]
    for frac, lab in [(.10, "10%"), (.01, "1%")]:
        ax[2].axhline(frac * tv0, color="gray", ls=":", lw=.8); ax[2].text(0, frac * tv0, f" {lab} of start", fontsize=7, va="bottom")
    ax[2].axvline(R, color="k", ls="--", lw=1)
    ax[2].set_xticks(x); ax[2].set_xticklabels(tp_labels + [f"+{i}" for i in range(1, H + 1)], fontsize=7, rotation=90)
    ax[2].set_ylabel("total variation to steady state"); ax[2].set_title("C. Distance-to-equilibrium (smoothed stabilization)")

    fig.suptitle("Model-inferred forward stabilization: redistribution iterated past T6 (forecast under late dynamics; 95% posterior bands)", y=1.02)
    fig.tight_layout()
    p = os.path.join(outdir, "temporal_forecast.png"); fig.savefig(p, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"wrote {p}")
    # console: steady-state occupancy + steps-to-stabilize
    settle = next((i - R for i in range(R, n_t) if tv_pt[i] <= .10 * tv0), None)
    print("forecast steady tissue occupancy:", {tis[a]: round(float(pi_occ[a]), 3) for a in range(S)})
    print(f"steps past T6 to reach 10% of initial distance: {settle}")


if __name__ == "__main__":
    main()
