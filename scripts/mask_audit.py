"""Transparency dump of the sampling grid, the forward-step mask, and the valid-observation
set per patient -- computed exactly as traffic.data.build does. Backs docs/DATA.md.

    TRAFFIC_H5=/path/to.h5ad python scripts/mask_audit.py

build's rules: a sample (p,tp,t) exists iff >=1 TCR+ cell there; a forward step is built for
every (patient, clone, source_tp) with dtp=source_tp+1 present for that patient; the
destination-tissue MASK for the step = { t : depth[(p, dtp, t)] > 0 } (independent of clone/source).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from traffic import data, statespace  # noqa: E402

ss = statespace.default()
TIS = list(ss.tissues)
H5 = os.environ.get(
    "TRAFFIC_H5",
    "/Users/ceglian/Codebase/GitHub/gbm_trafficking/data/objects/GBM_TCR_POS_TCELLS_singlets.h5ad",
)

df = data.load_obs_table(H5)
df = df[df["tissue"].isin(ss.tissues) & df["phenotype"].isin(ss.phenotypes)].copy()

dep = df.groupby(["patient", "timepoint", "tissue"], observed=True).size()
depth = {(p, int(tp), t): int(n) for (p, tp, t), n in dep.items()}
patients = sorted(df["patient"].unique())
tps_present = {p: sorted(int(x) for x in df[df.patient == p]["timepoint"].unique()) for p in patients}

print("=" * 78 + "\n(1) TCR+ CELLS per patient / timepoint / tissue   ('.' = not sampled)\n" + "=" * 78)
for p in patients:
    tps = tps_present[p]
    print(f"\n{p}   timepoints: {tps}")
    print(f"    {'tissue':6s} " + " ".join(f"t{t:<6d}" for t in tps))
    for t in TIS:
        print(f"    {t:6s} " + " ".join(f"{depth.get((p,tp,t),0) or '.':<7}" for tp in tps))

print("\n" + "=" * 78 + "\n(2) FORWARD-STEP DESTINATION MASK  (source tp -> tp+1)\n" + "=" * 78)
for p in patients:
    print(f"\n{p}")
    for tp in tps_present[p]:
        if (tp + 1) not in tps_present[p]:
            print(f"    t{tp}->t{tp+1}:  NO STEP (t{tp+1} not sampled)")
            continue
        dm = [t for t in TIS if depth.get((p, tp + 1, t), 0) > 0]
        sm = [t for t in TIS if depth.get((p, tp, t), 0) > 0]
        print(f"    t{tp}->t{tp+1}:  source={sm}  -> DEST MASK (observable)={dm}")

print("\n" + "=" * 78 + "\n(3) VALID OBSERVATIONS per patient x source tissue (raw-count attribution)\n" + "=" * 78)
obs = data.build(df, ss)
st = obs.src_tissue  # canonical raw-count source tissue
Dt = obs.D.reshape(len(st), ss.S, ss.K).sum(2)
R = pd.DataFrame({"patient": obs.patient, "src": [TIS[i] for i in st],
                  "TP_obs": Dt[:, TIS.index("TP")] > 0})
print(f"\nTotal valid clone-steps J = {len(R)}\n")
print(pd.crosstab(R["patient"], R["src"]).to_string())
print("\nCSF-source steps with tumor destination observable (drainage-informative):")
csf = R[R.src == "CSF"]
print(csf.groupby("patient").agg(n_CSF=("src", "size"), TP_dest_obs=("TP_obs", "sum")).to_string())
