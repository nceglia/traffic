"""analysis_recirculation_paths_visualize.py

Render the recirculation figures from analysis_recirculation_paths_setup.py output. Three views of
the same tissue-level growth operator G (rho-weighted collapse of M), per seed ("home") tissue:

  recirc_paths_trees.png   UNROLLED excursion -- home-rooted chains that FAN outward (each step a
                           little higher/lower), nodes colored by tissue. Each node draws a green
                           RETURN-HOME arc back to home (out-and-around, sized by the mass that
                           returns, m*recirc) and a compact per-step fate glyph (recirc / retain /
                           continue / decay, sums to 1).
  recirc_loops.png         COLLAPSED stationary loop (the roll-up) -- home + the two non-home
                           tissues as a mini-recirculation loop A<->B, with masses from the
                           absorbing-chain fundamental matrix N=(I-Q)^-1 so the two return arcs sum
                           EXACTLY to the headline recirculate %. Decay stubs stay dangling.
  recirc_paths_ranked.png  ranked reactive routes home->...->home, tagged direct vs via-loop.

"decay" is share-normalization loss (the global contraction lambda~0.68), NOT a literal death rate;
the glyph splits it into the shared-global part vs the tissue-specific excess. See
docs/analysis_recirculation_paths.md.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures", "analysis", "recirculation_paths")
TCOL = {"PBMC": "#b2182b", "CSF": "#2166ac", "TP": "#929292"}      # project style.py
TLAB = {"PBMC": "Blood", "CSF": "CSF", "TP": "Tumor"}

npz = np.load(os.path.join(OUT, "recirc_operators.npz"), allow_pickle=True)
TISS = [str(t) for t in npz["tissues"]]
G = npz["G_off"]; g = G.sum(1)
summary = json.load(open(os.path.join(OUT, "recirc_summary.json")))
idx = {t: i for i, t in enumerate(TISS)}
EPS, MAXD, DX = 0.0025, 7, 1.75                                    # lowered EPS surfaces the loop tail

# global contraction reference (occupancy-weighted mean g) -> the "universal" decay level (#2)
try:
    W = np.asarray(npz["rho_tissue"], float); W = W / W.sum()
    LAM = float((W * g).sum())
except KeyError:
    LAM = float(g.mean())
DECAY_GLOBAL = max(1.0 - LAM, 0.0)

# per-step fate split of a node's mass (recirculate + retain + continue + decay = 1)
FATE3 = [("recirculate", "#2E7D32"), ("retain", "#5b8db8"), ("continue", "#e0a458")]
C_DGLOB, C_DEXC = "#cdb0b0", "#c0392b"                             # global vs excess decay


def chain(home):
    """Two alternating excursions from home; list of node dicts with mass + per-step fates."""
    hi = idx[home]; others = [t for t in range(len(TISS)) if t != hi]
    chains = []
    for start in others:
        alt = [o for o in others if o != start][0]
        seq, cur, m, d = [], start, G[hi, start], 1
        while m >= EPS and d <= MAXD:
            seq.append(dict(t=cur, m=float(m), depth=d,
                            recirc=float(G[cur, hi]), retain=float(G[cur, cur]),
                            cont=float(G[cur, alt]), die=float(max(1 - g[cur], 0))))
            m *= G[cur, alt]; cur, alt = alt, cur; d += 1
        chains.append(seq)
    return hi, chains


def fund_matrix(hi):
    """Absorbing-chain fundamental matrix for mass leaving home hi (None if it self-sustains)."""
    others = [t for t in range(len(TISS)) if t != hi]
    Q = G[np.ix_(others, others)]
    if np.max(np.abs(np.linalg.eigvals(Q))) >= 1 or g[others].max() >= 1:
        return None
    N = np.linalg.inv(np.eye(len(others)) - Q)
    start = G[hi, others].astype(float); start = start / max(start.sum(), 1e-12)
    return dict(others=others, N=N, start=start, v=start @ N)          # v = expected visits / leaving mass


def fate_glyph(ax, x, ytop, fracs, w=0.92, h=0.15):
    """Stacked bar of the per-step fate; decay split into shared-global vs tissue-excess (#2)."""
    left = x - w / 2
    for frac, (_, col) in zip(fracs[:3], FATE3):
        seg = max(frac, 0.0) * w
        if seg > 1e-4:
            ax.add_patch(Rectangle((left, ytop - h), seg, h, fc=col, ec="none", zorder=3))
        left += seg
    decay = max(fracs[3], 0.0)
    g_part, ex_part = min(decay, DECAY_GLOBAL), max(decay - DECAY_GLOBAL, 0.0)
    for seg, col in [(g_part, C_DGLOB), (ex_part, C_DEXC)]:
        if seg * w > 1e-4:
            ax.add_patch(Rectangle((left, ytop - h), seg * w, h, fc=col, ec="none", zorder=3))
            left += seg * w
    ax.add_patch(Rectangle((x - w / 2, ytop - h), w, h, fill=False, ec="0.5", lw=0.5, zorder=3))


def edgept(c, tgt, r):
    d = np.asarray(tgt, float) - np.asarray(c, float); n = np.hypot(*d)
    u = d / n if n else d * 0.0
    return (c[0] + r * u[0], c[1] + r * u[1])


def arc(ax, p0, p1, rad, lw, color, alpha=1.0, ms=9, z=2, style="-|>"):
    ax.add_patch(FancyArrowPatch(p0, p1, connectionstyle=f"arc3,rad={rad}", arrowstyle=style,
                 mutation_scale=ms, lw=lw, color=color, alpha=alpha, zorder=z))


# =========================== FIG 1: unrolled fanning chains ============================= #
Y0, STEP = 1.25, 0.62
fig, axes = plt.subplots(1, 3, figsize=(21, 8.2))
for ax, home in zip(axes, TISS):
    hi, chains = chain(home)
    others = [t for t in range(len(TISS)) if t != hi]
    depmax = max((len(seq) for seq in chains), default=1)
    ret_max = max((n["m"] * n["recirc"] for seq in chains for n in seq), default=1e-9)
    ax.add_patch(Circle((0, 0), 0.36, fc=TCOL[home], ec="k", lw=1.6, zorder=5))
    ax.text(0, -0.62, TLAB[home], ha="center", fontsize=12, fontweight="bold")
    ax.text(0, 0, "home", ha="center", va="center", fontsize=7.5, color="w", fontweight="bold", zorder=6)
    for ci, seq in enumerate(chains):
        sgn = 1 if ci == 0 else -1
        prev = (0.0, 0.0)
        for n in seq:
            x = n["depth"] * DX; y = sgn * (Y0 + STEP * (n["depth"] - 1)); t = TISS[n["t"]]
            r = 0.14 + 0.30 * np.sqrt(n["m"])
            ax.plot([prev[0], x], [prev[1], y], "-", color="0.65", lw=0.7 + 6 * n["m"], alpha=0.55, zorder=1)
            # (A) return-home arc, bowed OUTWARD (up for top chain, down for bottom) so it never crosses
            rm = n["m"] * n["recirc"]
            if rm > 0.003:
                arc(ax, edgept((x, y), (0, 0), r), edgept((0, 0), (x, y), 0.36),
                    rad=-sgn * (0.32 + 0.12 * n["depth"]), lw=0.8 + 4.5 * rm / ret_max,
                    color="#2E7D32", alpha=0.32 + 0.55 * rm / ret_max)
            ax.add_patch(Circle((x, y), r, fc=TCOL[t], ec="k", lw=1, zorder=4))
            ax.text(x, y + r + 0.14, TLAB[t], ha="center", va="bottom", fontsize=9, fontweight="bold")
            fate_glyph(ax, x, y - r - 0.18, [n["recirc"], n["retain"], n["cont"], n["die"]])
            prev = (x, y)
    # (#1) drawn arcs = the strict-alternation returns; dwell-then-return paths are folded out here
    # (the unrolled chain skips self-loops) and only re-enter in the exact loop view -> annotate the gap
    rc = summary['fate_table']['off'][home]['recirc_mean']
    leaving = float(sum(G[hi, o] for o in others))
    captured = float(sum(n["m"] * n["recirc"] for seq in chains for n in seq)) / max(leaving, 1e-9)
    ax.set_title(f"home = {TLAB[home]}  —  {rc:.0%} returns home\n"
                 f"(no-dwell arcs = {captured:.0%}; dwell paths → loop view)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(-1.1, depmax * DX + 0.6)
    lim = Y0 + STEP * (depmax - 1) + 1.1
    ax.set_ylim(-lim, lim); ax.axis("off"); ax.set_aspect("equal")

fig.legend(handles=[FancyArrowPatch((0, 0), (1, 0), color="#2E7D32", arrowstyle="-|>", mutation_scale=10)]
                   + [Patch(fc=c, ec="0.5") for _, c in FATE3]
                   + [Patch(fc=C_DGLOB, ec="0.5"), Patch(fc=C_DEXC, ec="0.5")],
           labels=["return-home arc (recirculating mass)"] + [l for l, _ in FATE3]
                  + [f"decay: global ({DECAY_GLOBAL:.0%})", "decay: tissue excess"],
           loc="lower center", ncol=6, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig.suptitle("Recirculation reactive paths by seed tissue — unrolled excursion (node size ∝ still-circulating "
             "mass; green arc = mass returning home; bar = per-step fate split)", fontsize=13.5, fontweight="bold")
fig.tight_layout(rect=[0, 0.05, 1, 0.94])
fig.savefig(os.path.join(OUT, "recirc_paths_trees.png"), dpi=150, bbox_inches="tight")
print("wrote recirc_paths_trees.png")

# ========================= FIG 2: collapsed stationary loops (B) ======================== #
RX, RY = 2.7, 1.25
fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6.6))
for ax, home in zip(axes2, TISS):
    hi = idx[home]
    fm = fund_matrix(hi)
    ax.add_patch(Circle((0, 0), 0.34, fc=TCOL[home], ec="k", lw=1.6, zorder=5))
    ax.text(0, -0.58, TLAB[home], ha="center", fontsize=12, fontweight="bold")
    ax.text(0, 0, "home", ha="center", va="center", fontsize=7, color="w", fontweight="bold", zorder=6)
    if fm is None:
        ax.text(RX * 0.5, 0, "compartment\nself-sustains\n(ρ(Q) ≥ 1)", ha="center", va="center",
                fontsize=11, color="0.3", style="italic")
        ax.set_title(f"home = {TLAB[home]}", fontsize=12.5, fontweight="bold")
        ax.set_xlim(-1.1, RX + 1.6); ax.set_ylim(-2.4, 2.4); ax.axis("off"); ax.set_aspect("equal")
        continue
    A, B = fm["others"]; v = fm["v"]; start = fm["start"]
    pos = {A: np.array([RX, RY]), B: np.array([RX, -RY])}
    ret = {A: v[0] * G[A, hi], B: v[1] * G[B, hi]}                 # return mass; sums to recirc
    dec = {A: v[0] * max(1 - g[A], 0), B: v[1] * max(1 - g[B], 0)}
    flux = {(A, B): v[0] * G[A, B], (B, A): v[1] * G[B, A]}         # mini-loop fluxes
    dwell = {A: v[0] * G[A, A], B: v[1] * G[B, B]}                 # self-retention
    vmax = max(v.max(), 1e-9); fmax = max(max(ret.values()), max(dec.values()), max(flux.values()), 1e-9)
    for i, ci in enumerate((A, B)):
        c = pos[ci]; sgn = 1 if i == 0 else -1
        rr = 0.20 + 0.34 * np.sqrt(v[i] / vmax)
        # seed hop home -> node (grey)
        arc(ax, edgept((0, 0), c, 0.34), edgept(c, (0, 0), rr), rad=-sgn * 0.16,
            lw=0.8 + 3.5 * start[i], color="0.6", alpha=0.6, ms=8, z=2)
        # return arc node -> home (green, bowed outward), sized by returning mass
        arc(ax, edgept(c, (0, 0), rr), edgept((0, 0), c, 0.34), rad=-sgn * 0.42,
            lw=1.0 + 5 * ret[ci] / fmax, color="#2E7D32", alpha=0.85, ms=11, z=3)
        # decay stub -> dangling (outward, red)
        tip = c + np.array([0.9, sgn * 0.5])
        arc(ax, edgept(c, tip, rr), tip, rad=0.0, lw=1.0 + 5 * dec[ci] / fmax, color=C_DEXC, alpha=0.8, ms=10, z=2)
        ax.text(tip[0] + 0.08, tip[1] + sgn * 0.05, f"decay {dec[ci] / (ret[A] + ret[B] + dec[A] + dec[B]):.0%}",
                ha="left", va="center", fontsize=7.5, color=C_DEXC)
        # node
        ax.add_patch(Circle(tuple(c), rr, fc=TCOL[TISS[ci]], ec="k", lw=1.2, zorder=4))
        ax.text(c[0], c[1] + sgn * (rr + 0.14), TLAB[TISS[ci]], ha="center",
                va="bottom" if sgn > 0 else "top", fontsize=10, fontweight="bold")
        ax.text(c[0], c[1], f"×{v[i]:.2f}", ha="center", va="center", fontsize=7.5,
                color="w", fontweight="bold", zorder=5)
    # mini-loop back-edges A<->B (the roll-up), two lanes both bowed right (away from home)
    rA, rB = 0.20 + 0.34 * np.sqrt(v[0] / vmax), 0.20 + 0.34 * np.sqrt(v[1] / vmax)
    arc(ax, edgept(pos[A], pos[B], rA), edgept(pos[B], pos[A], rB),
        rad=0.34, lw=0.8 + 5 * flux[(A, B)] / fmax, color="#e0a458", alpha=0.9, ms=10, z=3)
    arc(ax, edgept(pos[B], pos[A], rB), edgept(pos[A], pos[B], rA),
        rad=-0.34, lw=0.8 + 5 * flux[(B, A)] / fmax, color="#e0a458", alpha=0.9, ms=10, z=3)
    surv = G[A, B] * G[B, A]
    ax.text(RX + 0.62, 0, f"loop\n{TLAB[TISS[A]][0]}⇄{TLAB[TISS[B]][0]}\nsurvival\n{surv:.1%}/cycle",
            ha="center", va="center", fontsize=8, color="#b07a2a", fontweight="bold")
    rc = ret[A] + ret[B]
    ax.set_title(f"home = {TLAB[home]}   —   recirculate {rc:.0%}  (green arcs sum exactly)",
                 fontsize=12.5, fontweight="bold")
    ax.set_xlim(-1.1, RX + 1.7); ax.set_ylim(-2.4, 2.4); ax.axis("off"); ax.set_aspect("equal")

fig2.legend(handles=[FancyArrowPatch((0, 0), (1, 0), color="0.6", arrowstyle="-|>", mutation_scale=10),
                     FancyArrowPatch((0, 0), (1, 0), color="#e0a458", arrowstyle="-|>", mutation_scale=10),
                     FancyArrowPatch((0, 0), (1, 0), color="#2E7D32", arrowstyle="-|>", mutation_scale=10),
                     FancyArrowPatch((0, 0), (1, 0), color=C_DEXC, arrowstyle="-|>", mutation_scale=10)],
            labels=["seed: home → tissue", "mini-loop A⇄B (continue)", "return to home (recirculate)",
                    "decay (dangling)"], loc="lower center", ncol=4, fontsize=10, frameon=False,
            bbox_to_anchor=(0.5, 0.0))
fig2.suptitle("Collapsed stationary recirculation loop — node ×N = expected visits (fundamental matrix); "
              "green return arcs sum exactly to the recirculate %", fontsize=13.5, fontweight="bold")
fig2.tight_layout(rect=[0, 0.06, 1, 0.93])
fig2.savefig(os.path.join(OUT, "recirc_loops.png"), dpi=150, bbox_inches="tight")
print("wrote recirc_loops.png")

# ============================= FIG 3: ranked reactive routes ============================ #
fig3, axes3 = plt.subplots(1, 3, figsize=(22, 6.4))
fig3.subplots_adjust(left=0.045, right=0.985, wspace=0.5, top=0.86, bottom=0.14)
for ax, home in zip(axes3, TISS):
    routes = summary["ranked_paths_off"][home][:12]
    ms = [r["mass"] for r in routes]
    via_loop = [len(r["path"]) > 3 for r in routes]                 # (#3) direct bounce vs via the A⇄B loop
    y = np.arange(len(routes))[::-1]
    cols = [(TCOL[home] if not vl else "#e0a458") for vl in via_loop]
    ax.barh(y, ms, color=cols, alpha=0.85)
    for yi, m in zip(y, ms):
        ax.text(m + max(ms) * 0.02, yi, f"{m:.03f}", va="center", fontsize=8)
    # route labels drawn as little tissue-colored node-link paths (matching the other figures)
    tr = ax.get_yaxis_transform()                                   # x = axes fraction, y = data row
    for r, yi in zip(routes, y):
        p = r["path"]; n = len(p)
        xs = [-0.022 - 0.028 * (n - 1 - k) for k in range(n)]       # last node nearest the axis
        ax.plot(xs, [yi] * n, "-", color="0.55", lw=1.0, transform=tr, clip_on=False, zorder=2)
        for xk, code in zip(xs, p):
            ax.scatter([xk], [yi], s=120, c=TCOL[code], edgecolors="k", linewidths=0.6,
                       transform=tr, clip_on=False, zorder=3)
            ax.text(xk, yi, TLAB[code][0], ha="center", va="center", fontsize=6, color="w",
                    fontweight="bold", transform=tr, clip_on=False, zorder=4)
    ax.set_yticks([]); ax.tick_params(left=False)
    ax.set_title(f"home = {TLAB[home]}: ranked reactive routes", fontsize=12, fontweight="bold")
    ax.set_xlabel("returning mass"); ax.set_xlim(0, max(ms) * 1.35); ax.grid(axis="x", alpha=0.25)
fig3.legend(handles=[Patch(fc="#e0a458", ec="none")],
            labels=["via the A⇄B loop (longer routes) — direct bounces drawn in the home-tissue colour"],
            loc="lower center", ncol=1, fontsize=10, frameon=False, bbox_to_anchor=(0.5, 0.0))
fig3.suptitle("Ranked recirculation routes (home → … → home), by returning mass — node-path = itinerary "
              "(B=Blood C=CSF T=Tumor)", fontsize=13, fontweight="bold")
fig3.savefig(os.path.join(OUT, "recirc_paths_ranked.png"), dpi=150, bbox_inches="tight")
print("wrote recirc_paths_ranked.png")
