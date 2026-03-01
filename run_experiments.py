"""
run_experiments.py
------------------
Sweeps circuit depth and noise level to compare Fibonacci vs linear vs odd
ZNE scheduling. Saves results to experiments/results/ and generates figures.

Run from project root:
    python run_experiments.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import json
from itertools import product

from src.zne_pipeline import run_pipeline, make_test_circuit, noisy_executor
from src.stability_filter import StabilityFilter
from src.fibonacci_scaling import compare_schedules, golden_ratio_density

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("experiments/results", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

# ── Experiment Parameters ─────────────────────────────────────────────────────
DEPTHS       = [2, 4, 6, 8]
NOISE_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05]
N_SCALES     = 5
GAMMA        = 0.05
NU           = 0.5
N_TRIALS     = 10          # repeat each config for variance estimation
SCHEDULES    = ["fibonacci", "linear", "odd"]

print("=" * 65)
print("Quantum-Vortex QEM: Experiment Runner")
print(f"Depths: {DEPTHS}")
print(f"Noise levels: {NOISE_LEVELS}")
print(f"Trials per config: {N_TRIALS}")
print("=" * 65)

# ── Data Collection ───────────────────────────────────────────────────────────
# Structure: results[depth][noise][schedule] = list of PipelineResult
all_results = {}

total = len(DEPTHS) * len(NOISE_LEVELS) * N_TRIALS
done = 0

for depth, noise in product(DEPTHS, NOISE_LEVELS):
    key = f"d{depth}_n{noise}"
    all_results[key] = {s: [] for s in SCHEDULES}

    for trial in range(N_TRIALS):
        trial_results = run_pipeline(
            circuit_depth=depth,
            base_noise=noise,
            stability_threshold=GAMMA,
            nu=NU,
            n_scales=N_SCALES,
            schedules=SCHEDULES
        )
        for s in SCHEDULES:
            all_results[key][s].append(trial_results[s])

        done += 1
        if done % 10 == 0:
            print(f"  Progress: {done}/{total} ({100*done//total}%)")

print("Data collection complete.\n")

# ── Statistical Summary ───────────────────────────────────────────────────────
summary = {}

for depth, noise in product(DEPTHS, NOISE_LEVELS):
    key = f"d{depth}_n{noise}"
    summary[key] = {}

    for s in SCHEDULES:
        trials = all_results[key][s]
        zne_vals    = np.array([t.zne_value for t in trials])
        epsilon_vals = np.array([t.stability.epsilon for t in trials])
        flag_rate   = np.mean([t.stability.flagged for t in trials])

        summary[key][s] = {
            "depth": depth,
            "noise": noise,
            "zne_mean":     float(np.mean(zne_vals)),
            "zne_std":      float(np.std(zne_vals)),
            "epsilon_mean": float(np.mean(epsilon_vals)),
            "epsilon_std":  float(np.std(epsilon_vals)),
            "flag_rate":    float(flag_rate),
        }

# Save JSON
with open("experiments/results/summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Saved: experiments/results/summary.json")

# ── Figure 1: ε vs Noise Level (per schedule, depth=4) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

depth = 4
colors = {"fibonacci": "#E63946", "linear": "#457B9D", "odd": "#2A9D8F"}
markers = {"fibonacci": "o", "linear": "s", "odd": "^"}

for s in SCHEDULES:
    eps_means = []
    eps_stds  = []
    for noise in NOISE_LEVELS:
        key = f"d{depth}_n{noise}"
        eps_means.append(summary[key][s]["epsilon_mean"])
        eps_stds.append(summary[key][s]["epsilon_std"])

    axes[0].errorbar(
        NOISE_LEVELS, eps_means, yerr=eps_stds,
        label=s.capitalize(), color=colors[s],
        marker=markers[s], linewidth=2, capsize=4
    )

axes[0].axhline(y=GAMMA, color="black", linestyle="--",
                linewidth=1.5, label=f"Γ = {GAMMA}")
axes[0].set_xlabel("Base Noise Level", fontsize=12)
axes[0].set_ylabel("ε (Energy Dissipation Norm)", fontsize=12)
axes[0].set_title(f"Stability Criterion vs Noise (depth={depth})", fontsize=13)
axes[0].legend()
axes[0].set_xscale("log")
axes[0].grid(True, alpha=0.3)

# ── Figure 2: ZNE Value vs Circuit Depth (noise=0.01) ────────────────────────
noise = 0.01
ideal = [np.cos(d * np.pi / 4) for d in DEPTHS]

for s in SCHEDULES:
    zne_means = []
    zne_stds  = []
    for depth in DEPTHS:
        key = f"d{depth}_n{noise}"
        zne_means.append(summary[key][s]["zne_mean"])
        zne_stds.append(summary[key][s]["zne_std"])

    axes[1].errorbar(
        DEPTHS, zne_means, yerr=zne_stds,
        label=s.capitalize(), color=colors[s],
        marker=markers[s], linewidth=2, capsize=4
    )

axes[1].plot(DEPTHS, ideal, "k--", linewidth=1.5, label="Ideal (noiseless)")
axes[1].set_xlabel("Circuit Depth", fontsize=12)
axes[1].set_ylabel("ZNE Extrapolated Value", fontsize=12)
axes[1].set_title(f"ZNE Accuracy vs Circuit Depth (noise={noise})", fontsize=13)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("paper/figures/fig1_stability_and_accuracy.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig1_stability_and_accuracy.png")

# ── Figure 3: Flag Rate Heatmap ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, s in enumerate(SCHEDULES):
    heatmap = np.zeros((len(DEPTHS), len(NOISE_LEVELS)))
    for i, depth in enumerate(DEPTHS):
        for j, noise in enumerate(NOISE_LEVELS):
            key = f"d{depth}_n{noise}"
            heatmap[i, j] = summary[key][s]["flag_rate"]

    im = axes[idx].imshow(heatmap, aspect="auto", cmap="RdYlGn_r",
                           vmin=0, vmax=1, origin="lower")
    axes[idx].set_xticks(range(len(NOISE_LEVELS)))
    axes[idx].set_xticklabels([str(n) for n in NOISE_LEVELS], rotation=45, fontsize=9)
    axes[idx].set_yticks(range(len(DEPTHS)))
    axes[idx].set_yticklabels([str(d) for d in DEPTHS])
    axes[idx].set_xlabel("Noise Level")
    axes[idx].set_ylabel("Circuit Depth")
    axes[idx].set_title(f"{s.capitalize()} — Flag Rate")
    plt.colorbar(im, ax=axes[idx], label="Fraction flagged")

plt.suptitle("Stability Filter Flag Rate: Fibonacci vs Linear vs Odd", fontsize=13, y=1.02)
plt.tight_layout()
fig.savefig("paper/figures/fig2_flag_rate_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig2_flag_rate_heatmap.png")

# ── Console Summary Table ─────────────────────────────────────────────────────
print("\n" + "=" * 75)
print(f"SUMMARY: depth=4, noise=0.01")
print(f"{'Schedule':<14} {'ZNE Mean':>10} {'ZNE Std':>10} "
      f"{'ε Mean':>12} {'ε Std':>10} {'Flag%':>8}")
print("-" * 75)

key = "d4_n0.01"
for s in SCHEDULES:
    d = summary[key][s]
    print(f"{s:<14} {d['zne_mean']:>10.6f} {d['zne_std']:>10.6f} "
          f"{d['epsilon_mean']:>12.6f} {d['epsilon_std']:>10.6f} "
          f"{d['flag_rate']*100:>7.1f}%")

print("=" * 75)
print("\nDone. Check paper/figures/ for plots.")
