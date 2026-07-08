"""pi->0 boundary / degeneracy experiment (Goal 1): plant EXACT zeros on chosen trafficking routes,
fit, and verify the degeneracy behaves as the model claims:

  (a) starved-route Phi reverts to the Dir(beta) prior mean 1/K (d_prior -> 0, entropy -> log K);
  (b) g, on-route pi, on-route Phi, M, and global phi all still recover;
  (c) M route-mass on the starved routes is ~0 AND is recovered -- the OPERATOR is fine there; only
      the internal Phi factorization is unidentified.

This is the rigorous version of the identifiability caveat: it plants the zeros deliberately (via
traffic.recovery.plant_pi_zeros + the simulate injection hook) rather than fishing for small random
draws. Hard asserts (like recovery_check_factored.py) + one figure to figures/validation/recovery/.

Run:  J=800 W=400 N=400 python scripts/boundary_pi_zero.py
Env:  J, W, N, SEED, PHI, NZERO (planted routes), OUTDIR.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.recovery import plant_pi_zeros, recover_once, route_mass
from traffic.simulate import prior_sample


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0,
                                beta=1.0, sigma_phi=1.0)
    J = int(os.environ.get("J", "800"))
    phi = float(os.environ.get("PHI", "10.0"))
    seed = int(os.environ.get("SEED", "0"))
    n_zero = int(os.environ.get("NZERO", "6"))
    cfg = MCMCConfig(num_warmup=int(os.environ.get("W", "400")),
                     num_samples=int(os.environ.get("N", "400")), num_chains=2, seed=seed)
    outdir = os.environ.get("OUTDIR", "figures/validation/recovery")
    os.makedirs(outdir, exist_ok=True)

    # --- base truth from the prior, then plant EXACT zeros on off-stay routes ------------------
    base = prior_sample(jax.random.PRNGKey(seed), prior, L, S)
    g0 = np.asarray(base["g"]); pi_base = np.asarray(base["pi"]); Phi0 = np.asarray(base["Phi"])
    tissue_of = ss.tissue_of
    zs = np.unique(np.linspace(0, L - 1, n_zero).astype(int))
    zero_routes = [(int(z), int((tissue_of[z] + 1) % S)) for z in zs]     # each: one off-tissue dest
    pi0, routes = plant_pi_zeros(pi_base, zero_routes, eps=0.0)
    planted = np.zeros((L, S), bool)
    for z, b in routes:
        planted[z, b] = True
    print(f"planted {len(routes)} exact-zero routes (off-stay): {routes}")

    # --- simulate at the planted truth, fit, score --------------------------------------------
    res, truth, m, _ = recover_once(jax.random.PRNGKey(seed), prior, cfg, ss, J=J, phi=phi,
                                    factors={"g": g0, "pi": pi0, "Phi": Phi0})
    print(f"fit: div={res.num_divergences} r_hat={res.r_hat_max:.3f} "
          f"ess_min={res.ess_min:.0f} n_draws={res.n_draws}")

    # --- planted-route-specific behavior (the boundary claims) --------------------------------
    Phi_h = res.factors["Phi"].mean(0)
    unif = np.full(K, 1.0 / K)
    d_prior_planted = float(np.mean([0.5 * np.abs(Phi_h[z, b] - unif).sum() for z, b in routes]))
    ent = np.array([-np.sum(np.where(Phi_h[z, b] > 0, Phi_h[z, b] * np.log(Phi_h[z, b]), 0.0))
                    for z, b in routes])
    ent_ratio_planted = float(ent.mean() / np.log(K))

    mass_true = truth["M"].reshape(L, S, K).sum(-1)      # == route_mass(g, pi0); planted rows are 0
    mass_hat = res.M_hat.reshape(L, S, K).sum(-1)
    planted_mass_hat = mass_hat[planted]
    onroute_mass = mass_hat[route_mass(truth["g"], truth["pi"]) >= m["route"]["tau_on"]]
    print(f"(a) planted-Phi d_prior={d_prior_planted:.3f}  entropy/logK={ent_ratio_planted:.3f}")
    print(f"(b) g corr={m['g']['corr']:.3f}  pi corr={m['pi']['corr']:.3f}  "
          f"Phi_on corr={m['Phi_on']['corr']:.3f}  M corr={m['M']['corr']:.3f}  "
          f"phi covered={m['phi']['covered']}")
    print(f"(c) planted M route-mass_hat: mean={planted_mass_hat.mean():.4f} "
          f"max={planted_mass_hat.max():.4f}  (true=0; median on-route mass={np.median(onroute_mass):.3f})")

    # --- figure -------------------------------------------------------------------------------
    Phid = res.factors["Phi"]
    w = route_mass(truth["g"], truth["pi"])
    Phi_t = truth["Phi"]
    d_prior_all = 0.5 * np.abs(Phi_h.reshape(L * S, K) - unif).sum(1)
    d_truth_all = 0.5 * np.abs(Phi_h.reshape(L * S, K) - Phi_t.reshape(L * S, K)).sum(1)
    w_flat = w.ravel(); planted_flat = planted.ravel()

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # (3a) d_prior vs d_truth, colored by route mass; planted ringed
    a = ax[0, 0]
    sc = a.scatter(d_prior_all[~planted_flat], d_truth_all[~planted_flat],
                   c=np.log10(w_flat[~planted_flat] + 1e-4), cmap="viridis", s=18, alpha=.8)
    a.scatter(d_prior_all[planted_flat], d_truth_all[planted_flat], facecolors="none",
              edgecolors="red", s=90, linewidths=1.6, label="planted zero")
    fig.colorbar(sc, ax=a, label="log10 route mass w=g*pi")
    a.set_xlabel("|posterior mean Phi - prior 1/K|  (d_prior)")
    a.set_ylabel("|posterior mean Phi - true Phi|  (d_truth)")
    a.set_title("A. Starved routes revert to prior (d_prior~0); on-route land on truth (d_truth~0)")
    a.legend(fontsize=8, loc="upper right")

    # (3c) recovered vs true M route-mass; planted at ~0
    a = ax[0, 1]
    mt, mh = mass_true.ravel(), mass_hat.ravel()
    a.scatter(mt[~planted_flat], mh[~planted_flat], s=18, color="#4c72b0", alpha=.6, label="open routes")
    a.scatter(mt[planted_flat], mh[planted_flat], s=60, color="#c44e52", alpha=.9, label="planted (true=0)")
    hi = 1.1 * max(mt.max(), mh.max())
    a.plot([0, hi], [0, hi], "k:", lw=1)
    a.set_xlabel("true M route-mass"); a.set_ylabel("posterior mean M route-mass")
    a.set_title("C. Operator route-mass IS recovered -- even where Phi is unidentified")
    a.legend(fontsize=8, loc="upper left")

    # (3b-i) an example planted route: Phi posterior vs prior 1/K (should sit on the prior)
    zp, bp = routes[0]
    a = ax[1, 0]
    dr = np.asarray(Phid[:, zp, bp, :])
    mean_, lo_, hi_ = dr.mean(0), np.quantile(dr, .025, 0), np.quantile(dr, .975, 0)
    a.bar(range(K), mean_, yerr=[mean_ - lo_, hi_ - mean_], capsize=2, color="#8172b3", alpha=.8)
    a.axhline(1.0 / K, color="r", ls="--", lw=1.5, label="prior mean 1/K")
    a.set_xticks(range(K)); a.set_xlabel("destination phenotype"); a.set_ylabel("Phi (95% CrI)")
    a.set_title(f"B. Planted route z={zp}->tissue {bp}: Phi posterior == prior (unidentified)")
    a.legend(fontsize=8)

    # (3b-ii) the strongest on-route route: Phi posterior spikes on the true phenotype
    z_on, b_on = np.unravel_index(np.argmax(w), w.shape)
    a = ax[1, 1]
    dr = np.asarray(Phid[:, z_on, b_on, :])
    mean_, lo_, hi_ = dr.mean(0), np.quantile(dr, .025, 0), np.quantile(dr, .975, 0)
    a.bar(range(K), mean_, yerr=[mean_ - lo_, hi_ - mean_], capsize=2, color="#55a868", alpha=.8,
          label="posterior")
    a.plot(range(K), Phi_t[z_on, b_on], "kD", ms=6, label="true Phi")
    a.axhline(1.0 / K, color="r", ls="--", lw=1, alpha=.6, label="prior mean 1/K")
    a.set_xticks(range(K)); a.set_xlabel("destination phenotype"); a.set_ylabel("Phi (95% CrI)")
    a.set_title(f"B. On-route z={int(z_on)}->tissue {int(b_on)} (w={w[z_on, b_on]:.2f}): "
                f"Phi recovers the truth")
    a.legend(fontsize=8)

    fig.suptitle(f"pi->0 boundary: starved-route switching is correctly unidentified while the "
                 f"operator recovers (J={J}, {len(routes)} planted zeros)", y=1.0)
    fig.tight_layout()
    f = os.path.join(outdir, "boundary_pi_zero.png")
    fig.savefig(f, dpi=140, bbox_inches="tight"); plt.close(fig)
    print("wrote:", f)

    # --- hard gates ---------------------------------------------------------------------------
    print("\ngates: (a) planted d_prior<0.10 & entropy/logK>0.90; (b) g/pi/M corr>0.90, "
          "Phi_on corr>0.80, phi covered; (c) planted mass_hat<0.15")
    assert d_prior_planted < 0.10, f"(a) planted Phi d_prior {d_prior_planted:.3f} >= 0.10 (should revert to prior)"
    assert ent_ratio_planted > 0.90, f"(a) planted Phi entropy/logK {ent_ratio_planted:.3f} <= 0.90"
    assert m["g"]["corr"] > 0.90, f"(b) g corr {m['g']['corr']:.3f} <= 0.90"
    assert m["pi"]["corr"] > 0.90, f"(b) pi corr {m['pi']['corr']:.3f} <= 0.90"
    assert m["M"]["corr"] > 0.90, f"(b) M corr {m['M']['corr']:.3f} <= 0.90"
    assert m["Phi_on"]["corr"] > 0.80, f"(b) on-route Phi corr {m['Phi_on']['corr']:.3f} <= 0.80"
    assert m["phi"]["covered"], "(b) true phi not inside 90% posterior interval"
    assert float(planted_mass_hat.max()) < 0.15, \
        f"(c) planted M route-mass_hat max {planted_mass_hat.max():.4f} >= 0.15 (should be ~0)"
    print("BOUNDARY PASS")


if __name__ == "__main__":
    main()
