"""Per-step read-outs of the fitted trafficking matrix M-hat.

===========================================================================
CONTRACT -- read before adding anything here.

M is fit in DEPTH-NORMALIZED SHARE space (x~ = x / d_src), so every quantity is
a share of a sampled pool, never an absolute cell count.

  SOUND read-outs (fully supported by shares):
    - redistribution     : the row-stochastic operator T = M / g -- WHERE a
                           state's share goes (persistence / switching / migration,
                           routes). Comparable, always valid.
    - composition change : phenotype mixes, JS divergences, convergence.
    - relative, WITHIN one pool : who gains / loses share of a single tissue's
                           repertoire (a within-pool selection coefficient).

  g = M.sum(1) is the NET PER-STEP SHARE MULTIPLIER (the diag(g) factor of
  M = diag(g) . T). It is a RELATIVE, within-pool diagnostic under compositional
  closure:
    - NOT absolute expansion / proliferation / fecundity -- share change is not
      cell-count change; a clone's share rises if it grows OR if others shrink,
      and shares cannot tell these apart;
    - NOT comparable across states or tissues -- different pools, and the
      pool-weighted mean is pinned to ~1 by closure.
  Absolute cell-count fold-change is recoverable ONLY with an ABUNDANCE ANCHOR
  (per-sample total T-cell counts / cellularity / spike-in) -- see
  `absolute_fold_change`, which REQUIRES one and cannot be called without it.

  Rule for new code: no symbol here may be named growth / expansion / fecundity,
  and no function may return a per-tissue or cross-tissue magnitude aggregate.
  Within-pool magnitude is allowed only when named and labelled RELATIVE.
===========================================================================
"""
from __future__ import annotations

import numpy as np

from .statespace import StateSpace


def decompose(M):
    """M = diag(g) T.  Returns (T [L,L] row-stochastic where g>0, g [L]).

    g is the net per-step share multiplier (see module CONTRACT): a relative,
    within-pool diagnostic -- not absolute expansion, not cross-tissue comparable.
    """
    g = M.sum(1)
    T = M / np.maximum(g[:, None], 1e-12)
    return T, g


def net_share_multiplier(M):
    """Per-state net per-step share multiplier g_z = row sum of M.

    CONTRACT: RELATIVE, within-pool only. g_z > 1 = state z gains share of its pool,
    < 1 = loses share; the pool-weighted mean is ~1 by closure. NOT absolute expansion
    and NOT comparable across states/tissues. For an interpretable within-pool signal
    use `relative_within_tissue_share_change`; for absolute counts use
    `absolute_fold_change` (needs an abundance anchor).
    """
    return np.asarray(M).sum(1)


def relative_within_tissue_share_change(M, ss: StateSpace, rho=None):
    """Within-tissue RELATIVE net share change -- the sanctioned 'who gains/loses share'.

    For each source tissue a, every phenotype's net share multiplier g_z expressed
    RELATIVE to that tissue's own pool average (rho-weighted): s_z = g_z / gbar_a.
    A within-pool selection coefficient: s_z > 1 out-competes the local repertoire,
    < 1 under-competes; by construction the rho-weighted mean of s within each tissue
    is 1.

    CONTRACT: relative and WITHIN a single pool only -- never compare s across tissues
    as a magnitude (the pools differ). Returns s [L] (per state).
    """
    S, K = ss.S, ss.K
    g = net_share_multiplier(M)
    if rho is None:
        rho = np.full((S, K), 1.0 / K)
    s = np.zeros(S * K)
    for a in range(S):
        blk = slice(a * K, (a + 1) * K)
        pool_avg = rho[a] @ g[blk]
        s[blk] = g[blk] / max(float(pool_avg), 1e-12)
    return s


def absolute_fold_change(M, ss: StateSpace, abundance_anchor):
    """Absolute per-state fold-change in CELL NUMBER (true expansion/contraction).

    NOT identifiable from depth-normalized shares (compositional closure): g measures
    share change, not cell-count change. Recovering absolute fold-change REQUIRES an
    abundance anchor -- per-sample total T-cell counts / cellularity / spike-in --
    mapping shares to absolute counts. `abundance_anchor` is a REQUIRED argument for
    exactly this reason: the absolute claim is unrepresentable without the data that
    licenses it. Not implemented until an anchor exists and the model consumes it.
    """
    if abundance_anchor is None:
        raise NotImplementedError(
            "Absolute fold-change is not identifiable from shares; it needs an abundance "
            "anchor (per-sample total T-cell counts / cellularity / spike-in), which is "
            "not available. See the module CONTRACT."
        )
    raise NotImplementedError(
        "absolute_fold_change: anchor provided but the share->count model is not yet "
        "implemented -- wire this up when the abundance anchor lands."
    )


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

    w(z) propto source_occ(z) * sum_{z' in dest_tissue} M[z, z'].  A NORMALIZED
    (composition) attribution -- relative shares of the inflow, not absolute counts.
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


def summary(M, ss: StateSpace, rho=None) -> dict:
    """Bundle the SOUND read-outs for reporting: redistribution (T), collapsed
    trafficking, within-tissue switching, and the within-pool RELATIVE share-change
    diagnostic. Emits no cross-tissue magnitude (see module CONTRACT)."""
    T, _g = decompose(M)
    return {
        "T": T,
        "tissue_traffic": tissue_traffic(T, ss, rho),
        "within_tissue_switching": within_tissue_switching(T, ss),
        "relative_within_tissue_share_change": relative_within_tissue_share_change(M, ss, rho),
    }
