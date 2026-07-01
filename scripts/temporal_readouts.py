"""Temporal-stabilization readouts from per-transition operators (temporal_Mr.npz).

For each transition M_r we read the SHARE-SUPPORTED behaviors off the row-stochastic
T_r (redistribution) and ask whether each STABILIZES across transitions (curve flattens)
-- i.e. whether the trafficking RULES become time-homogeneous after the OVI transient.
This is a T (redistribution) question, which shares fully support; NO magnitude /
expansion axis is shown -- absolute expansion is not identifiable from shares
(compositional closure; see traffic/readouts.py CONTRACT).

  A  transport composition of T_r: persistence / switching / migration (mask mass) vs
     transition, with posterior bands.
  B  operator drift: mean row-JS(T_r, T_{r+1}) and JS(T_r, T_pooled) -> decays to a floor
     when the operator stops changing.
  C  T-spectrum: subdominant modulus |lambda2(T)| (redistribution mixing; 0=fast, 1=slow)
     and the subdominant oscillation angle (recirculation mode) vs transition. Computed on
     the row-stochastic T (lambda1(T)=1), so it is scale-free and never touches magnitude.

Point estimates from M_hat; bands from the saved posterior-draw subset.
Env: FIT_TEMPORAL (npz, default temporal_Mr.npz)  OUTDIR (default .)
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from traffic import statespace
from traffic.readouts import _js


def _mask_indices(S, K):
    """Boolean [L,L] masks: persistence (diag), switching (in-block off-diag), migration (off-block)."""
    L = S * K
    a = np.repeat(np.arange(S), K)                       # source tissue of each state
    same_tis = a[:, None] == a[None, :]
    diag = np.eye(L, dtype=bool)
    return diag, same_tis & ~diag, ~same_tis


def metrics(M, ss):
    """Per-transition SHARE-SUPPORTED scalars: T mask masses + T-spectrum (no magnitude)."""
    S, K, L = ss.S, ss.K, ss.L
    g = M.sum(1)
    T = M / np.maximum(g[:, None], 1e-12)
    per, swi, mig = _mask_indices(S, K)
    frac = lambda msk: float(T[msk].sum() / L)           # T rows sum to 1 -> total mass = L
    # spectrum on the row-stochastic redistribution operator T (lambda1=1; scale-free).
    ev = np.linalg.eigvals(T)
    ev = ev[np.argsort(-np.abs(ev))]
    lam2 = ev[1] if len(ev) > 1 else 0.0
    subdom = float(abs(lam2))                            # mixing: 0 fast -> 1 slow to converge
    osc = float(abs(np.angle(lam2))) if abs(np.imag(lam2)) > 1e-9 else 0.0
    return {"persistence": frac(per), "switching": frac(swi), "migration": frac(mig),
            "subdom": subdom, "osc": osc, "T": T}


def main():
    ss = statespace.default()
    fit = os.environ.get("FIT_TEMPORAL", "temporal_Mr.npz")
    outdir = os.environ.get("OUTDIR", ".")
    os.makedirs(outdir, exist_ok=True)
    d = np.load(fit, allow_pickle=True)
    M_hat, samples, src_tp = d["M_hat"], d["samples"], d["src_tp"].astype(int)
    R = len(src_tp)
    labels = [f"T{r}->T{r+1}" for r in src_tp]

    pt = [metrics(M_hat[i], ss) for i in range(R)]
    draw_m = [[metrics(samples[i][s], ss) for s in range(samples.shape[1])] for i in range(R)]

    def band(key):
        lo, md, hi = [], [], []
        for i in range(R):
            vals = np.array([m[key] for m in draw_m[i]])
            lo.append(np.percentile(vals, 2.5)); md.append(np.median(vals)); hi.append(np.percentile(vals, 97.5))
        return np.array(lo), np.array(md), np.array(hi)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(R)

    # A: transport composition of T_r (share-supported)
    for key, col in [("persistence", "#6b6b6b"), ("switching", "#e6852f"), ("migration", "#2c6fbb")]:
        lo, md, hi = band(key)
        ax[0].plot(x, md, "-o", color=col, label=key)
        ax[0].fill_between(x, lo, hi, color=col, alpha=.15)
    ax[0].set_title("A. Transport composition of T_r (mask mass)")
    ax[0].set_ylabel("fraction of transport"); ax[0].legend(fontsize=8)

    # B: operator drift
    Tpt = [pt[i]["T"] for i in range(R)]
    Tpool = np.mean(M_hat, 0); Tpool = Tpool / np.maximum(Tpool.sum(1, keepdims=True), 1e-12)
    consec = [np.mean([_js(Tpt[i][z], Tpt[i + 1][z]) for z in range(ss.L)]) for i in range(R - 1)]
    vspool = [np.mean([_js(Tpt[i][z], Tpool[z]) for z in range(ss.L)]) for i in range(R)]
    ax[1].plot(np.arange(R - 1) + .5, consec, "-o", color="#c44", label="JS(T_r, T_{r+1})")
    ax[1].plot(x, vspool, "-s", color="#446", label="JS(T_r, T_pooled)")
    ax[1].set_title("B. Operator drift (flattens = time-homogeneous)")
    ax[1].set_ylabel("mean row Jensen-Shannon"); ax[1].legend(fontsize=8)

    # C: T-spectrum (scale-free redistribution convergence; no magnitude)
    lo, md, hi = band("subdom")
    ax[2].plot(x, md, "-o", color="#8e44ad", label="|λ₂(T)| (mixing)")
    ax[2].fill_between(x, lo, hi, color="#8e44ad", alpha=.15)
    axo = ax[2].twinx()
    lo, md, hi = band("osc")
    axo.plot(x, md, "--s", color="#1b9e77", alpha=.7, label="arg(λ₂) (oscillation)")
    axo.set_ylabel("oscillation angle (rad)", color="#1b9e77")
    ax[2].set_title("C. T-spectrum: redistribution mixing + oscillation")
    ax[2].set_ylabel("|λ₂(T)|"); ax[2].legend(fontsize=8, loc="upper left")

    for a in ax:
        a.set_xticks(x); a.set_xticklabels(labels); a.set_xlabel("transition")
    fig.suptitle("Temporal stabilization of the trafficking OPERATOR T (per-transition M_r) "
                 "-- redistribution only, no magnitude", y=1.02)
    fig.tight_layout()
    p = os.path.join(outdir, "temporal_stabilization.png")
    fig.savefig(p, dpi=140, bbox_inches="tight")
    print(f"wrote {p}")

    print("\ntransition   persist  switch  migrate |  |λ2(T)|  osc   div  rhat")
    for i in range(R):
        print(f"  {labels[i]:11s}  {pt[i]['persistence']:.3f}   {pt[i]['switching']:.3f}   "
              f"{pt[i]['migration']:.3f} |  {pt[i]['subdom']:.3f}  {pt[i]['osc']:.2f}  "
              f"{int(d['divergences'][i]):3d}  {d['r_hat'][i]:.3f}")


if __name__ == "__main__":
    main()
