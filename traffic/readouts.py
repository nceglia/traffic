"""Per-step read-outs of the fitted growth matrix (model_methods.tex Sec. 7).

Everything is a deterministic function of M-hat and its factorization
M-hat = diag(g) T (T row-stochastic).  All quantities are per one forward step.
"""
from __future__ import annotations

import numpy as np

from .statespace import StateSpace


def decompose(M):
    """M = diag(g) T.  Returns (T [L,L] row-stochastic where g>0, g [L])."""
    g = M.sum(1)
    T = M / np.maximum(g[:, None], 1e-12)
    return T, g


def growth(M):
    """Per-state net growth factor g_z (row sum)."""
    return np.asarray(M).sum(1)


def tissue_traffic(T, ss: StateSpace, rho=None):
    """Collapsed tissue->tissue trafficking, [S, S].

    entry[a, b] = sum_u rho^a(u) sum_v T[(a,u),(b,v)].  Rows sum to 1.
    rho: [S, K] source phenotype weights per tissue (uniform if None).
    """
    S, K = ss.S, ss.K
    if rho is None:
        rho = np.full((S, K), 1.0 / K)
    P = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            block = T[a * K:(a + 1) * K, b * K:(b + 1) * K]   # [K, K]
            P[a, b] = rho[a] @ block.sum(1)                    # sum_u rho(u) sum_v T
    return P


def growth_by_tissue(M, ss: StateSpace, rho=None):
    """Phenotype-averaged net growth per source tissue, [S]."""
    S, K = ss.S, ss.K
    g = growth(M)
    if rho is None:
        rho = np.full((S, K), 1.0 / K)
    return np.array([rho[a] @ g[a * K:(a + 1) * K] for a in range(S)])


def within_tissue_switching(T, ss: StateSpace):
    """Phenotype switching matrix within each tissue (stay-in-tissue routes), [S, K, K].

    For source tissue a: h[a, u, v] = T[(a,u),(a,v)] / sum_v' T[(a,u),(a,v')].
    """
    S, K = ss.S, ss.K
    H = np.zeros((S, K, K))
    for a in range(S):
        block = T[a * K:(a + 1) * K, a * K:(a + 1) * K]       # [K, K]
        H[a] = block / np.maximum(block.sum(1, keepdims=True), 1e-12)
    return H


def _js(p, q, eps=1e-12):
    """Jensen-Shannon divergence between two distributions (base 2, in [0,1])."""
    p = np.asarray(p) + eps; p /= p.sum()
    q = np.asarray(q) + eps; q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log2(a / b))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def phenotype_resolved_route(T, ss, a, b, rho):
    """Share of route a->b carried by each source phenotype u, [K] (sums to 1)."""
    K = ss.K
    contrib = np.array([rho[a, u] * T[a * K + u, b * K:(b + 1) * K].sum() for u in range(K)])
    return contrib / max(contrib.sum(), 1e-12)


def outflow_distribution(T, ss, a, rho):
    """Tissue a's full destination distribution over L states (sums to 1)."""
    K = ss.K
    out = sum(rho[a, u] * T[a * K + u] for u in range(K))
    return out / max(out.sum(), 1e-12)


def backward_attribution(M, ss, dest_tissue, source_occ):
    """Which source (tissue,phenotype) states feed `dest_tissue`, [L] (sums to 1).

    w(z) propto source_occ(z) * sum_{z' in dest_tissue} M[z, z'].
    source_occ: [L] current source mass (e.g. rho broadcast x tissue occupancy).
    """
    K = ss.K
    reach = M[:, dest_tissue * K:(dest_tissue + 1) * K].sum(1)   # [L]
    w = np.asarray(source_occ) * reach
    return w / max(w.sum(), 1e-12)


def dest_pheno_by_source_tissue(T, ss, dest_tissue, rho):
    """For destination tissue b, the arriving phenotype distribution from each
    source tissue a, [S, K]. Convergence => rows similar across a."""
    S, K = ss.S, ss.K
    H = np.zeros((S, K))
    for a in range(S):
        v = sum(rho[a, u] * T[a * K + u, dest_tissue * K:(dest_tissue + 1) * K] for u in range(K))
        H[a] = v / max(v.sum(), 1e-12)
    return H


def reshaping(T, ss, rho):
    """Per route a->b, JS divergence between source phenotype mix (rho^a) and
    the arriving phenotype mix, [S, S]. Larger => more reshaping on that route."""
    S, K = ss.S, ss.K
    R = np.zeros((S, S))
    for a in range(S):
        for b in range(S):
            arr = sum(rho[a, u] * T[a * K + u, b * K:(b + 1) * K] for u in range(K))
            R[a, b] = _js(rho[a], arr) if arr.sum() > 0 else np.nan
    return R


def clone_expansion(theta, g):
    """Per-clone net expansion = sum_z theta(z) g_z. theta: [J,L] source comp."""
    return np.asarray(theta) @ np.asarray(g)


def summary(M, ss: StateSpace, rho=None) -> dict:
    """Bundle the main read-outs for reporting."""
    T, g = decompose(M)
    return {
        "T": T, "g": g,
        "tissue_traffic": tissue_traffic(T, ss, rho),
        "growth_by_tissue": growth_by_tissue(M, ss, rho),
        "within_tissue_switching": within_tissue_switching(T, ss),
    }
