"""Contract tests for traffic.readouts -- the compositional-limit guardrails.

Run: PYTHONPATH=. python -m pytest -q tests/test_readouts_contract.py

These lock the module CONTRACT: shares support redistribution / composition and
within-pool RELATIVE magnitude only. A future re-introduction of a cross-pool or
absolute 'expansion' quantity must fail here.
"""
import re

import numpy as np
import pytest

from traffic import readouts, statespace

_BANNED = re.compile(r"growth|expansion|fecundity", re.I)


def _M(seed=0):
    ss = statespace.default()
    rng = np.random.default_rng(seed)
    M = np.abs(rng.normal(size=(ss.L, ss.L))) + 0.05
    return M, ss


def test_public_api_has_no_magnitude_named_symbol():
    public = [n for n in dir(readouts) if not n.startswith("_")]
    offenders = [n for n in public if _BANNED.search(n)]
    assert not offenders, f"banned magnitude-named symbols in readouts API: {offenders}"


def test_traps_are_removed():
    for gone in ("growth", "growth_by_tissue", "within_tissue_growth", "clone_expansion"):
        assert not hasattr(readouts, gone), f"{gone} must not exist (compositional trap)"


def test_summary_emits_no_cross_tissue_magnitude():
    M, ss = _M()
    s = readouts.summary(M, ss)
    assert not any(_BANNED.search(k) for k in s), f"summary keys leak magnitude: {list(s)}"
    assert set(s) <= {"T", "tissue_traffic", "within_tissue_switching",
                      "relative_within_tissue_share_change"}


def test_absolute_fold_change_requires_anchor():
    M, ss = _M()
    with pytest.raises(TypeError):                      # missing the required anchor arg
        readouts.absolute_fold_change(M, ss)
    with pytest.raises(NotImplementedError):            # None anchor refused, not faked
        readouts.absolute_fold_change(M, ss, None)


def test_relative_within_tissue_share_change_is_pool_relative():
    M, ss = _M()
    s = readouts.relative_within_tissue_share_change(M, ss)
    S, K = ss.S, ss.K
    for a in range(S):                                  # rho-weighted mean is 1 per tissue
        blk = s[a * K:(a + 1) * K]
        assert abs(blk.mean() - 1.0) < 1e-9
