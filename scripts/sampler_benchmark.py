"""Sampler hardening benchmark (Phase 3 of the simulation validation). Before committing SBC compute:

  1. measure the DIVERGENCE RATE on truths drawn from the FULL prior -- the hard, funnel-tail draws
     (heavy-g, extreme-phi) that SBC will face, unlike the fixed phi=10 recovery draw -- and pick
     target_accept; and
  2. time per-round WALL-CLOCK to size the SBC run.

The N truths theta* ~ prior (INCLUDING log_r) are drawn ONCE and reused across every target_accept, so
the only thing that differs between configs is the sampler setting -- a controlled comparison. The
simulated data and the NUTS seed for round r are identical across configs.

Note on the reparameterization question: the model's g ~ LogNormal has a FIXED scale sigma_g (not a
sampled/hierarchical scale), so there is no Neal's funnel in g and a non-centered reparam is not
expected to help. This benchmark tests that empirically -- if raising target_accept already zeroes the
divergences, no model change is warranted.

Run:  N=5 J=800 W=400 NS=400 python -u scripts/sampler_benchmark.py
Env:  N (rounds), J, W (warmup), NS (samples), TA (comma target_accepts), SEED.
"""
import os
import sys
import time
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

from traffic import statespace
from traffic.config import FactoredPriorConfig, MCMCConfig
from traffic.mcmc import fit_nuts
from traffic.simulate import make_synthetic_factored, prior_sample


def one_fit(prior, cfg, ss, J, key_truth, key_sim):
    """Draw theta* ~ prior (incl. log_r), simulate at phi=exp(log_r), fit; return (res, seconds, phi)."""
    L, S = ss.L, ss.S
    th = prior_sample(key_truth, prior, L, S)
    phi = float(np.exp(np.asarray(th["log_r"])))
    Xt, Y, D, _, _, _ = make_synthetic_factored(
        key_sim, prior, L=L, J=J, S=S, phi=phi,
        factors={"g": np.asarray(th["g"]), "pi": np.asarray(th["pi"]), "Phi": np.asarray(th["Phi"])})
    t0 = time.perf_counter()
    res = fit_nuts(np.asarray(Xt), np.asarray(Y), np.asarray(D), prior, cfg)
    return res, time.perf_counter() - t0, phi


def main():
    ss = statespace.default(); S, K, L = ss.S, ss.K, ss.L
    prior = FactoredPriorConfig(mu_g=0.0, sigma_g=0.4, alpha_off=1.0, alpha_stay=4.0,
                                beta=1.0, sigma_phi=1.0)
    N = int(os.environ.get("N", "5")); J = int(os.environ.get("J", "800"))
    W = int(os.environ.get("W", "400")); NS = int(os.environ.get("NS", "400"))
    seed = int(os.environ.get("SEED", "0"))
    tas = [float(x) for x in os.environ.get("TA", "0.9,0.95,0.99").split(",")]
    base = MCMCConfig(num_warmup=W, num_samples=NS, num_chains=2)
    ndraws = NS * base.num_chains

    # pre-draw N (truth_key, sim_key) pairs -- SAME across every config
    round_keys = jax.random.split(jax.random.PRNGKey(seed), N)
    pairs = [tuple(jax.random.split(k)) for k in round_keys]

    print(f"benchmark: N={N} rounds, J={J}, W={W}, NS={NS}, chains={base.num_chains}, "
          f"target_accepts={tas}  (first fit includes JAX compile)", flush=True)
    summary = {}
    for ta in tas:
        divs, rhats, esss, times, fails = [], [], [], [], 0
        for r in range(N):
            cfg_r = replace(base, target_accept=ta, seed=r)
            try:
                res, dt, phi = one_fit(prior, cfg_r, ss, J, *pairs[r])
            except Exception as e:                                    # a pathological tiny-phi draw
                fails += 1
                print(f"  ta={ta} round {r}: FAILED ({type(e).__name__}: {str(e)[:60]})", flush=True)
                continue
            divs.append(res.num_divergences); rhats.append(res.r_hat_max)
            esss.append(res.ess_min); times.append(dt)
            print(f"  ta={ta} round {r}: div={res.num_divergences} r_hat={res.r_hat_max:.3f} "
                  f"ess={res.ess_min:.0f} phi={phi:.2f} {dt:.1f}s", flush=True)
        n_ok = len(divs)
        div_rate = (sum(divs) / (n_ok * ndraws)) if n_ok else float("nan")
        summary[ta] = dict(div_rate=div_rate, frac_rounds_div=float(np.mean([d > 0 for d in divs])) if n_ok else float("nan"),
                           max_div=max(divs) if n_ok else -1, rhat_max=max(rhats) if n_ok else float("nan"),
                           ess_min=min(esss) if n_ok else float("nan"),
                           t_med=float(np.median(times)) if n_ok else float("nan"),
                           t_max=max(times) if n_ok else float("nan"), fails=fails)
        s = summary[ta]
        print(f"== ta={ta}: div_rate={s['div_rate']:.4f} ({sum(divs)} divs / {n_ok * ndraws} draws), "
              f"rounds_with_div={s['frac_rounds_div']:.0%}, max_r_hat={s['rhat_max']:.3f}, "
              f"min_ess={s['ess_min']:.0f}, wallclock med={s['t_med']:.1f}s max={s['t_max']:.1f}s, "
              f"fails={fails}", flush=True)

    ok = [t for t in tas if summary[t]["div_rate"] == summary[t]["div_rate"]]           # drop NaN
    if ok:
        best = min(ok, key=lambda t: (summary[t]["div_rate"], summary[t]["t_med"]))
        tmed = summary[best]["t_med"]
        print(f"\n--- SBC cost extrapolation @ this tier (best target_accept={best}, "
              f"{tmed:.1f}s/round median): ---", flush=True)
        for Rn in (200, 500, 1000):
            print(f"  R={Rn}: ~{Rn * tmed / 3600:.1f} h sequential  "
                  f"(~{Rn * tmed / 3600 / 8:.1f} h across 8 workers)", flush=True)
        print("Paper-tier fits (larger J/W/NS) scale ~linearly; re-benchmark at the chosen SBC config.",
              flush=True)
        print(f"\nRECOMMENDATION: target_accept={best} "
              f"(div_rate={summary[best]['div_rate']:.4f}). "
              f"{'Non-centered log g NOT needed.' if summary[best]['div_rate'] < 0.002 else 'Consider non-centered log g if div_rate stays high.'}",
              flush=True)


if __name__ == "__main__":
    main()
