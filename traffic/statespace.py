"""Joint tissue-phenotype state space  Z = S x K,  L = S*K = 33.

State-index convention:
    z = tissue_index * K + phenotype_index

The default orderings below are 3 tissues x 11 T-cell phenotypes ("TP" is the
tumor tissue). Pass your own ``tissues``/``phenotypes`` to ``StateSpace`` to use
a different state space.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TISSUES = ("PBMC", "CSF", "TP")  # TP = Tumor
PHENOTYPES = (
    "CD8_Activated_TEMRA",
    "CD8_Activated_TEXeff",
    "CD8_Activated_TEXterm",
    "CD8_Activated_TRM",
    "CD8_Quiescent_Memory",
    "CD8_Quiescent_Naive",
    "CD8_Quiescent_TEXprog",
    "CD4_Naive_Memory",
    "CD4_Exhausted",
    "CD4_Treg",
    "CD4_Th",
)


@dataclass(frozen=True)
class StateSpace:
    tissues: tuple = TISSUES
    phenotypes: tuple = PHENOTYPES

    @property
    def S(self) -> int:
        return len(self.tissues)

    @property
    def K(self) -> int:
        return len(self.phenotypes)

    @property
    def L(self) -> int:
        return self.S * self.K

    def index(self, tissue: str, phenotype: str) -> int:
        return self.tissues.index(tissue) * self.K + self.phenotypes.index(phenotype)

    @property
    def tissue_of(self) -> np.ndarray:
        """[L] integer tissue index of each state."""
        return np.repeat(np.arange(self.S), self.K)

    def labels(self) -> list[tuple[str, str]]:
        return [(s, p) for s in self.tissues for p in self.phenotypes]

    def state_labels(self) -> list[str]:
        return [f"{s}:{p}" for s, p in self.labels()]


def default() -> StateSpace:
    return StateSpace()
