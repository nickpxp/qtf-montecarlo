"""
QTF speed-over-timing Monte Carlo.

vw = migration finish year minus CRQC emergence year; positive means too slow.
Base year 2026, 10,000 draws per scenario, fixed seed for reproducibility.
Prints the scenario grid and sensitivity sweeps, writes Figures 1-3.
Run:  python qtf_montecarlo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

SEED = 42
BASE = 2026
N = 10_000

# CRQC arrival: lognormal offset from BASE. Later median = later arrival.
CRQC = {
    "aggressive":   (np.log(5),  0.35),   # Google/IonQ ~2031
    "moderate":     (np.log(8),  0.45),   # NIST/GRI ~2034
    "conservative": (np.log(13), 0.50),   # Nat'l Academies ~2039
}

# Migration: finish = start + length, both truncated normal.
# (start_mid, start_sd, start_lo, start_hi, len_mid, len_sd, len_lo, len_hi)
MIG = {
    "optimistic":  (2026, 0.75, 2026, 2028,  4, 1.0, 2.5,  6),
    "baseline":    (2027, 1.00, 2026, 2030,  7, 1.5, 4.0, 10),
    "pessimistic": (2028, 1.50, 2026, 2032, 10, 2.0, 6.0, 15),
}


def run_one(cm, cs, m, rng, force_len=None):
    """One scenario. Returns p_vuln, median exposure among vulnerable, and vw."""
    t_crqc = BASE + rng.lognormal(cm, cs, N)
    start = np.clip(rng.normal(m[0], m[1], N), m[2], m[3])
    lmid = m[4] if force_len is None else force_len
    length = np.clip(rng.normal(lmid, m[5], N), m[6], m[7])
    vw = (start + length) - t_crqc
    bad = vw > 0
    return {
        "p_vuln": float(bad.mean()),
        "exposure_yrs": float(np.median(vw[bad])) if bad.any() else np.nan,
        "vw": vw,
        "t_crqc": t_crqc,
        "start": start,
        "completion": start + length,
    }


def scenario_grid(rng):
    rows = []
    for cn, (cm, cs) in CRQC.items():
        for mn in MIG:
            r = run_one(cm, cs, MIG[mn], rng)
            rows.append((cn, mn, round(r["p_vuln"], 4), round(r["exposure_yrs"], 2)))
    return rows


def emergence_quantiles(rng):
    out = {}
    for cn, (cm, cs) in CRQC.items():
        x = BASE + rng.lognormal(cm, cs, 500_000)
        out[cn] = (np.median(x), np.quantile(x, 0.90))
    return out


# ---------------------------------------------------------------- run + report
rng = np.random.default_rng(SEED)
grid = scenario_grid(rng)

print(f"{'crqc':13} {'migration':12} {'p_vuln':>8} {'exp_yrs':>8}")
for cn, mn, p, e in grid:
    print(f"{cn:13} {mn:12} {p:8.4f} {e:8.2f}")

gmap = {(cn, mn): (p, e) for cn, mn, p, e in grid}
pb, eb = gmap[("moderate", "baseline")]
print(f"\nBASELINE (moderate x baseline): p_vuln={pb:.4f}  median_exposure={eb:.2f}")

# baseline distribution stats
rng = np.random.default_rng(SEED + 1)
base_run = run_one(*CRQC["moderate"], MIG["baseline"], rng)
print(f"overall median window (all iters): {np.median(base_run['vw']):.2f}")
print(f"median emergence year (moderate): {np.median(base_run['t_crqc']):.2f}")
print(f"median completion year (baseline): {np.median(base_run['completion']):.2f}")

# worst case window
rng = np.random.default_rng(SEED + 2)
wc = run_one(*CRQC["aggressive"], MIG["pessimistic"], rng)
print(f"worst-case (aggr x pess) median window among vulnerable: "
      f"{np.median(wc['vw'][wc['vw'] > 0]):.2f}")

# emergence quantiles
rng = np.random.default_rng(SEED + 3)
eq = emergence_quantiles(rng)
print("\nemergence median / P90:")
for cn, (med, p90) in eq.items():
    print(f"  {cn:13} median={med:.1f}  P90={p90:.1f}")

# ---- sensitivity sweeps -----------------------------------------------------
# (a) emergence median sweep, deployment held at baseline
print("\nemergence sweep (median year -> p_vuln), deployment=baseline:")
emg_years, emg_p = [], []
rng = np.random.default_rng(SEED + 4)
for target in range(2029, 2041):
    mu = np.log(target - BASE)
    r = run_one(mu, CRQC["moderate"][1], MIG["baseline"], rng)
    emg_years.append(target)
    emg_p.append(r["p_vuln"])
    print(f"  median~{target}  p_vuln={r['p_vuln']:.3f}")

# (b) deployment duration sweep, emergence held at moderate
print("\ndeployment duration sweep (years -> p_vuln), emergence=moderate:")
dur_len, dur_p = [], []
rng = np.random.default_rng(SEED + 5)
for L in range(3, 13):
    r = run_one(*CRQC["moderate"], MIG["baseline"], rng, force_len=L)
    dur_len.append(L)
    dur_p.append(r["p_vuln"])
    print(f"  len={L:2}  p_vuln={r['p_vuln']:.3f}")

# per-year marginals near baseline
def marginal(xs, ys, x0):
    i = xs.index(x0)
    return abs(ys[i + 1] - ys[i - 1]) / 2 * 100
print(f"\nemergence marginal near 2034: {marginal(emg_years, emg_p, 2034):.1f} pts/yr")
print(f"deployment marginal near 7yr: {marginal(dur_len, dur_p, 7):.1f} pts/yr")
print(f"emergence range {emg_years[0]}->{emg_years[-1]}: "
      f"{emg_p[0]*100:.0f}% -> {emg_p[-1]*100:.0f}%")
print(f"deployment range {dur_len[-1]}yr->{dur_len[0]}yr: "
      f"{dur_p[-1]*100:.0f}% -> {dur_p[0]*100:.0f}%")

# ============================================================== FIGURES =======
plt.rcParams.update({"font.size": 10, "axes.grid": False})

EMG = ["aggressive", "moderate", "conservative"]
DEP = ["optimistic", "baseline", "pessimistic"]

# ---- Figure 1: 9-scenario vulnerability probability heatmap -----------------
M = np.array([[gmap[(e, d)][0] for d in DEP] for e in EMG])
fig, ax = plt.subplots(figsize=(6.4, 4.2))
im = ax.imshow(M, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(3)); ax.set_xticklabels([d.capitalize() for d in DEP])
ax.set_yticks(range(3)); ax.set_yticklabels([e.capitalize() for e in EMG])
ax.set_xlabel("Deployment scenario"); ax.set_ylabel("CRQC emergence scenario")
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{M[i, j]*100:.1f}%", ha="center", va="center",
                color="black", fontweight="bold")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Vulnerability probability")
ax.set_title("Vulnerability probability across the nine scenarios")
fig.tight_layout()
fig.savefig("figure1_scenario_matrix.png", dpi=300)
plt.close(fig)

# ---- Figure 2: baseline four-panel ------------------------------------------
b = base_run
fig = plt.figure(figsize=(9, 6.4))
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.24)

ax_a = fig.add_subplot(gs[0, 0])
ax_a.hist(b["t_crqc"], bins=60, color="#4C72B0", alpha=0.85)
ax_a.axvline(np.median(b["t_crqc"]), color="k", ls="--", lw=1)
ax_a.set_title("(a) CRQC emergence"); ax_a.set_xlabel("Year"); ax_a.set_ylabel("Count")

ax_b = fig.add_subplot(gs[0, 1])
ax_b.hist(b["start"], bins=40, color="#55A868", alpha=0.8, label="Start")
ax_b.hist(b["completion"], bins=60, color="#C44E52", alpha=0.6, label="Completion")
ax_b.set_title("(b) Deployment start and completion")
ax_b.set_xlabel("Year"); ax_b.set_ylabel("Count"); ax_b.legend(fontsize=8)

ax_c = fig.add_subplot(gs[1, 0])
ax_c.hist(b["vw"], bins=60, color="#8172B3", alpha=0.85)
ax_c.axvline(0, color="k", lw=1.5)
ax_c.set_title("(c) Vulnerability-window distribution")
ax_c.set_xlabel("Window (years); >0 = vulnerable"); ax_c.set_ylabel("Count")

ax_d = fig.add_subplot(gs[1, 1])
sv = np.sort(b["vw"])
ax_d.plot(sv, np.linspace(0, 1, sv.size), color="#937860", lw=1.6)
ax_d.axvline(0, color="k", ls="--", lw=1)
frac = float((b["vw"] > 0).mean())
ax_d.set_title(f"(d) CDF; {frac*100:.1f}% vulnerable")
ax_d.set_xlabel("Window (years)"); ax_d.set_ylabel("Cumulative probability")

fig.suptitle("Baseline scenario: moderate emergence x seven-year deployment",
             fontsize=11)
fig.savefig("figure2_baseline_distributions.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ---- Figure 3: sensitivity two-panel ----------------------------------------
fig, (axL, axR) = plt.subplots(1, 2, figsize=(9, 3.8))
axL.plot(emg_years, np.array(emg_p) * 100, "o-", color="#4C72B0")
axL.axvline(2034, color="grey", ls=":", lw=1)
axL.set_title("(a) Sensitivity to CRQC emergence")
axL.set_xlabel("Median CRQC emergence year"); axL.set_ylabel("Vulnerability probability (%)")
axL.set_ylim(0, 100)

axR.plot(dur_len, np.array(dur_p) * 100, "o-", color="#C44E52")
axR.axvline(7, color="grey", ls=":", lw=1)
axR.set_title("(b) Sensitivity to deployment duration")
axR.set_xlabel("Deployment duration (years)"); axR.set_ylabel("Vulnerability probability (%)")
axR.set_ylim(0, 100)
fig.tight_layout()
fig.savefig("figure3_sensitivity.png", dpi=300)
plt.close(fig)

print("\nWrote figure1_scenario_matrix.png, figure2_baseline_distributions.png, "
      "figure3_sensitivity.png")
