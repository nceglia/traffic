#!/usr/bin/env python3
"""
Plate diagram for the FACTORED joint tissue-phenotype population-dynamics model
(model_methods.tex).

Factored generative model:
    g_{a,u}        ~ LogNormal(mu_g, sigma_g^2)        (per-source-state growth)
    pi_{(a,u)}     ~ Dirichlet(alpha_a)                (destination-tissue / trafficking)
    Phi_{(a,u),b}  ~ Dirichlet(beta)                   (destination-phenotype / switching)
    M_{(a,u),(b,v)} = g_{a,u} * pi_{(a,u)}(b) * Phi_{(a,u),b}(v)   (assembled mean matrix)
    log phi        ~ Normal(0, sigma_phi^2)            (NB concentration; shared)
    xtilde_irc     = x_irc / d^src_irs                 (depth-rescaled source; deterministic)
    mu_irc         = xtilde_irc M                      (destination intensity; deterministic)
    y_irc(z')      ~ NB2(d_irs * mu_irc(z'), phi)      over non-missing tissue sub-blocks

The three operator factors (g, pi, Phi) are drawn once per source state z=(a,u)
from their priors and assembled into the mean matrix M; this block, together with
the shared NB concentration phi, sits OUTSIDE the patient plate (one M and one phi
are shared across all patients and steps). M is a non-negative mean matrix with free
row sums (row sum g_z is the net expansion factor of state z).

The observation side is drawn with explicit nesting so the data hierarchy is
visible:
    patient i
      └── forward step r in R_i          (r = source timepoint; dest = r+1)
            └── clonotype c in C_ir
                  └── tissue s in S          (per-tissue destination draw)

x_irc lives in the clone plate (shared across tissues); the per-tissue destination
draw y_irc and its sequencing-depth exposure d_irs live in the innermost tissue
sub-plate. mu_irc = xtilde_irc M is deterministic and has NO node -- the multiply
and the depth-rescaling xtilde = x / d^src ride the x -> y and M -> y edges, so
x_irc (the observed input) and M point straight at y_irc (the modeled outcome).
The missingness flag m_irs also has no node: masking is a support restriction on
the tissue sub-blocks, not a random variable.

Outputs: docs/figures/plate_joint_transition.{pdf,png}

Dependency note: needs the plate-notation library `daft` (daft-pgm.org), whose
import name collides with an unrelated `daft` dataframe engine. Install the PGM one
in an isolated env, e.g.:

    python -m venv /tmp/daftpgm
    /tmp/daftpgm/bin/pip install daft-pgm matplotlib
    /tmp/daftpgm/bin/python docs/_model_methods_plate_diagram.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import rc

try:
    import daft

    if not hasattr(daft, "PGM"):
        raise ImportError
except ImportError as exc:  # pragma: no cover - environment guard
    raise SystemExit(
        "This script needs the plate-diagram library 'daft-pgm' "
        "(imports as 'daft'). Note that the PyPI name 'daft' now refers to an "
        "unrelated dataframe library. Install with:\n\n"
        "    pip install daft-pgm\n"
    ) from exc

rc("font", family="serif", size=11)
rc("text", usetex=False)  # mathtext -> no LaTeX install required

# Figures live alongside the rest of the docs figures, regardless of cwd.
OUT = Path(__file__).resolve().parent / "figures"


def build_plate(out_dir: Path = OUT) -> None:
    # Factored population model.  Nesting: patient > step > clone > tissue.
    #   * the three operator factors g, pi, Phi -> M sit in the top z-plate,
    #     outside the patient plate, each fed by its fixed hyperparameters;
    #   * the shared NB concentration phi (with hyperparameter sigma_phi) also sits
    #     outside the patient plate and feeds every destination draw y;
    #   * NO mu node: mu = xtilde M is deterministic, so x -> y and M -> y carry the
    #     multiply and the depth-rescaling xtilde=x/d^src straight into the draw;
    #   * x is the observed INPUT (double-ring, observed_style="inner"); y is the
    #     modeled outcome (true gray shade, plot_params fc=0.7);
    #   * within the tissue sub-plate the destination draw y is fed by its
    #     sequencing-depth exposure d_irs and the shared dispersion phi (no mask node).
    pgm = daft.PGM(
        shape=(8.8, 8.6),
        origin=(0.0, 0.0),
        observed_style="inner",
        grid_unit=1.5,
        node_unit=1.05,
        directed=True,
    )

    # ---------------------------------------------------------------------
    # Operator factors (top z-plate).  For each source state z=(a,u): growth g,
    # trafficking pi, switching Phi, each from its fixed hyperparameters, assembled
    # into the mean matrix M = diag(g) (pi (x) Phi).  Shared across all patients
    # and steps -> drawn OUTSIDE the patient plate.
    # ---------------------------------------------------------------------
    pgm.add_node("hg", r"$(\mu_g,\sigma_g)$", 2.45, 7.35, fixed=True, offset=(0, 6))
    pgm.add_node("ha", r"$\alpha_a$",          3.95, 7.35, fixed=True, offset=(0, 6))
    pgm.add_node("hb", r"$\beta$",             5.45, 7.35, fixed=True, offset=(0, 6))

    pgm.add_node("g",   r"$g_{z}$",              2.45, 6.55)
    pgm.add_node("pi",  r"$\pi_{(a,u)}$",        3.95, 6.55)
    pgm.add_node("Phi", r"$\Phi_{(a,u),b}$",     5.45, 6.55)
    pgm.add_node("M",   r"$M$",                  3.95, 5.45)

    pgm.add_edge("hg", "g")
    pgm.add_edge("ha", "pi")
    pgm.add_edge("hb", "Phi")
    pgm.add_edge("g", "M")
    pgm.add_edge("pi", "M")
    pgm.add_edge("Phi", "M")

    pgm.add_plate(
        [1.75, 4.95, 4.45, 3.25],
        label=r"$z \in \mathcal{Z}$",
        shift=-0.12,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Shared NB concentration phi (with hyperparameter sigma_phi), outside the plates.
    pgm.add_node("hr", r"$\sigma_\phi$", 7.25, 7.35, fixed=True, offset=(0, 6))
    pgm.add_node("r",  r"$\phi$",         7.25, 6.55)
    pgm.add_edge("hr", "r")

    # ---------------------------------------------------------------------
    # Observation model.  NO mu node: mu_irc = xtilde_irc M is deterministic, so the
    # assembled operator M and the source x point STRAIGHT at y (the multiply and the
    # depth-rescaling xtilde = x / d^src ride the edges).  Two observed glyphs so the
    # nodes read as different kinds of thing: x (the observed INPUT we condition on) is
    # the double-ring (observed_style="inner"); y (the modeled outcome) is a TRUE GRAY
    # SHADE (plot_params fc=0.7).  NO mask node: missingness is a support restriction,
    # not a random node.
    # ---------------------------------------------------------------------
    pgm.add_node("x", r"$x_{irc}$", 2.90, 2.25, observed=True)                # double-ring (input)
    pgm.add_node("y", r"$y_{irc}$", 4.88, 2.25, plot_params={"fc": "0.7"})    # gray shade (outcome)
    pgm.add_node("d", r"$d_{irs}$", 4.88, 2.97, fixed=True, offset=(0, 6))    # per-tissue depth exposure

    pgm.add_edge("x", "y")           # source composition (depth-rescaled to xtilde) enters the mean
    pgm.add_edge("M", "y")           # assembled operator; mu = xtilde M folded into this edge
    pgm.add_edge("d", "y")           # per-tissue sequencing-depth exposure
    pgm.add_edge("r", "y")           # shared dispersion into every destination draw

    # Tissue sub-plate (innermost): the per-tissue destination draw {y, d}.
    # Nudged down-and-left of y_irc so the "s in S" label clears the shaded node;
    # right edge held so y_irc (which sits mid-plate) stays contained.
    pgm.add_plate(
        [3.95, 1.40, 1.55, 2.22],
        label=r"$s \in \mathcal{S}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Clone plate: wraps x and the tissue sub-plate.
    pgm.add_plate(
        [2.25, 1.30, 3.45, 2.65],
        label=r"$c \in \mathcal{C}_{ir}$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Step plate, nested inside patient.
    pgm.add_plate(
        [2.00, 0.70, 3.95, 3.45],
        label=r"$r \in \mathcal{R}_i$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # Patient plate (outermost on the observation side; M and phi live outside it),
    # matched to the z-plate width (left 1.75, right 6.20).
    pgm.add_plate(
        [1.75, 0.10, 4.45, 4.45],
        label=r"$i = 1,\ldots,I$",
        shift=-0.10,
        rect_params={"ec": "k", "fc": "none"},
    )

    # ---------------------------------------------------------------------
    # Render and save
    # ---------------------------------------------------------------------
    pgm.render()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir / "plate_joint_transition.pdf"
    out_png = out_dir / "plate_joint_transition.png"

    pgm.savefig(str(out_pdf))
    pgm.figure.savefig(str(out_png), dpi=300, bbox_inches="tight")
    plt.close(pgm.figure)
    print(f"wrote {out_pdf} and {out_png}")


if __name__ == "__main__":
    build_plate()
