"""analysis_recirculation_paths_setup.py

Compute the tissue-level recirculation structure on the GROWTH operator M (non-conserving),
for each seed ("home") tissue, and save it for the visualizer. See docs/analysis_recirculation_paths.md.

Method (see docs): T = decompose(M) is the conserving redistribution; here we keep the growth,
G = rho-weighted tissue collapse of M (row sums g != 1). Mass leaving a home tissue meets one of
several per-step fates -- RETAINED (self-loop g_tt), DIED (decay 1-g_t), RECIRCULATE (return to
home), or CONTINUE (migrate to the other tissue). The recirculate-vs-decay split is an absorbing
Markov chain (transient = the two non-home tissues); the ranked paths are TPT-style reactive
routes home->...->home, ranked by returning mass, with consecutive dwell-repeats collapsed.

Outputs (figures/analysis/recirculation_paths/):
  recirc_operators.npz   G (posterior mean) + g + per-tissue rates, per config
  recirc_summary.json    fate table (recirc/decay by seed x config) + ranked paths (off)
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic import io, statespace  # noqa: E402

ss = statespace.default(); S, K = ss.S, ss.K
TISS = list(ss.tissues)
iP, iC, iT = TISS.index("PBMC"), TISS.index("CSF"), TISS.index("TP")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "figures", "analysis", "recirculation_paths")
os.makedirs(OUT, exist_ok=True)

CONFIGS = {
    "off":       "/Users/ceglian/Data/misc/nb_global_putative.npz",
    "stay4":     "/Users/ceglian/Data/gbm/gpu_node_fits/nb_global_putative_stay4.npz",
    "stay6":     "/Users/ceglian/Data/gbm/gpu_node_fits/nb_global_putative_stay6.npz",
    "stay6_2_6": "/Users/ceglian/Data/gbm/gpu_node_fits/nb_global_putative_stay6_2_6.npz",
}


def tissue_growth_operator(M, rho):
    """3x3 rho-weighted collapse of the GROWTH operator M (NOT row-normalized)."""
    G = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            G[a, b] = sum(rho[a, u] * M[a * K + u, b * K:(b + 1) * K].sum() for u in range(K))
    return G


def fate_split(G, home):
    """Absorbing-chain recirculate-vs-decay fate for mass leaving `home` (None if compartment self-sustains)."""
    others = [t for t in range(S) if t != home]
    g = G.sum(1)
    Q = G[np.ix_(others, others)]
    if np.max(np.abs(np.linalg.eigvals(Q))) >= 1 or g[others[0]] >= 1 or g[others[1]] >= 1:
        return None
    N = np.linalg.inv(np.eye(2) - Q)
    Bret = N @ G[others, home]
    start = G[home, others].copy(); start = start / max(start.sum(), 1e-12)
    return dict(recirc=float(start @ Bret), decay=float(1 - start @ Bret),
                cond_return={TISS[others[i]]: float(Bret[i]) for i in range(2)},
                first_hop={TISS[others[i]]: float(start[i]) for i in range(2)})


def ranked_paths(G, home, eps=2e-4, maxd=12, topn=14):
    """TPT-style reactive routes home->...->home, ranked by returning mass; dwell-repeats collapsed."""
    agg = {}
    stack = [([home], 1.0, False)]
    while stack:
        path, m, left = stack.pop()
        cur = path[-1]
        for nxt in range(S):
            if nxt == cur:  # dwell/self-loop is folded out of the itinerary (kept as a rate)
                continue
            mn = m * G[cur, nxt]
            if mn < eps or len(path) > maxd:
                continue
            if nxt == home and left:
                key = tuple(path + [home])
                agg[key] = agg.get(key, 0.0) + mn
            elif nxt != home:
                stack.append((path + [nxt], mn, True))
    routes = sorted(agg.items(), key=lambda kv: -kv[1])[:topn]
    return [{"path": [TISS[i] for i in p], "mass": float(m)} for p, m in routes]


# ---- compute + save ----
per_tissue_rates = {}  # config -> {tissue: {retain, die}}
operators = {}
fate_tbl = {name: {} for name in CONFIGS}
for name, path in CONFIGS.items():
    fit = io.load_fit(path); rho = fit.rho; draws = fit.draws(2000)
    Gmean = tissue_growth_operator(fit.M_hat, rho)
    operators[name] = Gmean
    g = Gmean.sum(1)
    per_tissue_rates[name] = {TISS[t]: {"retain": float(Gmean[t, t]), "die": float(max(1 - g[t], 0))}
                              for t in range(S)}
    for home in range(S):
        rec = [fate_split(tissue_growth_operator(draws[d], rho), home) for d in range(len(draws))]
        rec = [x["recirc"] for x in rec if x is not None]
        if len(rec) > 10:
            fate_tbl[name][TISS[home]] = {"recirc_mean": float(np.mean(rec)),
                                          "recirc_lo": float(np.percentile(rec, 2.5)),
                                          "recirc_hi": float(np.percentile(rec, 97.5)),
                                          "valid_frac": 1.0}
        else:
            fate_tbl[name][TISS[home]] = {"self_sustains": True, "valid_frac": 0.0}

# ranked paths + conditional detail on the baseline
Goff = operators["off"]
paths_off = {TISS[home]: ranked_paths(Goff, home) for home in range(S)}
fate_detail_off = {TISS[home]: fate_split(Goff, home) for home in range(S)}

rho_off = io.load_fit(CONFIGS["off"]).rho           # [S,K]; tissue occupancy weights for the viz
np.savez(os.path.join(OUT, "recirc_operators.npz"),
         tissues=np.array(TISS), rho_tissue=rho_off.sum(1),
         **{f"G_{name}": G for name, G in operators.items()})
summary = {"configs": list(CONFIGS), "tissues": TISS, "fate_table": fate_tbl,
           "per_tissue_rates": per_tissue_rates, "ranked_paths_off": paths_off,
           "fate_detail_off": fate_detail_off}
with open(os.path.join(OUT, "recirc_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("wrote", os.path.join(OUT, "recirc_operators.npz"))
print("wrote", os.path.join(OUT, "recirc_summary.json"))
print("\nRECIRCULATE fraction of mass leaving each home tissue:")
for home in TISS:
    row = "  " + f"{home:5s}"
    for name in CONFIGS:
        e = fate_tbl[name][home]
        row += f"  {name}=" + (f"{e['recirc_mean']:.2f}" if "recirc_mean" in e else "self-sust")
    print(row)
