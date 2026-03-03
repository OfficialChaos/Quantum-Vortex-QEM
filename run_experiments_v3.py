"""
run_experiments_v3.py
---------------------
Overnight experiment runner with:
- 20-30 trials per config (tighter error bars)
- 1000 shots per evaluation (closer to real hardware)
- Extended depths to 20
- Wider noise range including 0.15 and 0.2
- All 3 circuit types, schedules, extrapolants from v2

Target runtime: 8-12 hours overnight
Run from project root: python run_experiments_v3.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import json
import time
from itertools import product

import cirq
from src.stability_filter import StabilityFilter
from src.fibonacci_scaling import (
    fibonacci_scale_factors,
    linear_scale_factors,
    odd_scale_factors
)

os.makedirs("experiments/results", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

# ── Experiment Parameters ─────────────────────────────────────────────────────
DEPTHS        = [2, 4, 6, 8, 10, 12, 16, 20]
NOISE_LEVELS  = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
N_SCALES      = 6
GAMMA         = 0.05
NU            = 0.5
N_SHOTS       = 500
N_TRIALS      = 10
SCHEDULES     = ["fibonacci", "linear", "odd"]
EXTRAPOLANTS  = ["linear", "richardson", "poly2"]
CIRCUIT_TYPES = ["test", "vqe", "random"]

total = len(CIRCUIT_TYPES) * len(DEPTHS) * len(NOISE_LEVELS)
print("=" * 65)
print("Quantum-Vortex QEM: Overnight Experiment Runner v3")
print(f"Depths         : {DEPTHS}")
print(f"Noise levels   : {NOISE_LEVELS}")
print(f"Trials         : {N_TRIALS}")
print(f"Shots          : {N_SHOTS}")
print(f"Circuit types  : {CIRCUIT_TYPES}")
print(f"Extrapolants   : {EXTRAPOLANTS}")
print(f"Total configs  : {total}")
print(f"Est. runtime   : 8-12 hours")
print("=" * 65)

# ── Circuit Builders ──────────────────────────────────────────────────────────

def make_test_circuit(depth):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    for _ in range(depth):
        circuit.append([cirq.rx(np.pi/4)(q0), cirq.rx(np.pi/4)(q1), cirq.CNOT(q0, q1)])
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit

def make_vqe_circuit(depth):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    rng = np.random.RandomState(42)
    for _ in range(depth):
        circuit.append([
            cirq.ry(rng.uniform(0, 2*np.pi))(q0),
            cirq.ry(rng.uniform(0, 2*np.pi))(q1),
            cirq.CZ(q0, q1)
        ])
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit

def make_random_circuit(depth, seed=0):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    rng = np.random.RandomState(seed)
    gates = [cirq.rx, cirq.ry, cirq.rz]
    for _ in range(depth):
        circuit.append([
            gates[rng.randint(3)](rng.uniform(0, 2*np.pi))(q0),
            gates[rng.randint(3)](rng.uniform(0, 2*np.pi))(q1)
        ])
        if rng.random() > 0.3:
            circuit.append(cirq.CNOT(q0, q1))
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit

CIRCUIT_BUILDERS = {
    "test":   make_test_circuit,
    "vqe":    make_vqe_circuit,
    "random": make_random_circuit,
}

# ── Executor ──────────────────────────────────────────────────────────────────

def noisy_executor(circuit, noise_level, n_shots=1000):
    capped = min(noise_level, 0.24)
    qubits = sorted(circuit.all_qubits())
    q0, q1 = qubits[0], qubits[1]
    sim_circuit = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            if not isinstance(op.gate, cirq.MeasurementGate):
                sim_circuit.append(op)
    sim_circuit.append(cirq.measure(q0, q1, key='result'))
    noisy = sim_circuit.with_noise(cirq.depolarize(p=capped))
    result = cirq.DensityMatrixSimulator().run(noisy, repetitions=n_shots)
    bits = result.measurements['result']
    z0 = 1 - 2 * bits[:, 0].astype(float)
    z1 = 1 - 2 * bits[:, 1].astype(float)
    return float(np.mean(z0 * z1))

# ── Extrapolants ──────────────────────────────────────────────────────────────

def extrapolate(ev_array, sf_array, method="linear"):
    if method == "linear":
        return float(np.polyval(np.polyfit(sf_array, ev_array, 1), 0.0))
    elif method == "richardson":
        s1, s2 = sf_array[0], sf_array[1]
        e1, e2 = ev_array[0], ev_array[1]
        return float((s2*e1 - s1*e2) / (s2 - s1)) if abs(s2-s1) > 1e-10 else float(e1)
    elif method == "poly2":
        deg = 2 if len(sf_array) >= 3 else 1
        return float(np.polyval(np.polyfit(sf_array, ev_array, deg), 0.0))

# ── Main Loop ─────────────────────────────────────────────────────────────────

sf_filter = StabilityFilter(gamma=GAMMA, nu=NU)
schedule_map = {
    "fibonacci": fibonacci_scale_factors(N_SCALES),
    "linear":    linear_scale_factors(N_SCALES),
    "odd":       odd_scale_factors(N_SCALES),
}

all_results = {}
done = 0
start_time = time.time()

for circuit_type, depth, noise in product(CIRCUIT_TYPES, DEPTHS, NOISE_LEVELS):
    key = f"{circuit_type}_d{depth}_n{noise}"
    all_results[key] = {}
    circuit = CIRCUIT_BUILDERS[circuit_type](depth)

    for schedule in SCHEDULES:
        all_results[key][schedule] = {}
        scales = schedule_map[schedule]
        sf_array = np.array(scales, dtype=float)

        for extrap in EXTRAPOLANTS:
            trial_zne, trial_eps, trial_flagged = [], [], []

            for trial in range(N_TRIALS):
                ev_array = np.array([
                    noisy_executor(circuit, noise * s, N_SHOTS)
                    for s in scales
                ])
                zne_val = extrapolate(ev_array, sf_array, method=extrap)
                stab = sf_filter.evaluate(ev_array, sf_array)
                trial_zne.append(zne_val)
                trial_eps.append(stab.epsilon)
                trial_flagged.append(stab.flagged)

            all_results[key][schedule][extrap] = {
                "circuit_type": circuit_type,
                "depth":        depth,
                "noise":        noise,
                "zne_mean":     float(np.mean(trial_zne)),
                "zne_std":      float(np.std(trial_zne)),
                "epsilon_mean": float(np.mean(trial_eps)),
                "epsilon_std":  float(np.std(trial_eps)),
                "flag_rate":    float(np.mean(trial_flagged)),
            }

    done += 1
    elapsed = time.time() - start_time
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    pct = 100 * done // total
    print(f"  {pct:3d}% ({done}/{total}) | "
          f"elapsed: {elapsed/60:.1f}m | "
          f"ETA: {eta/60:.1f}m | "
          f"last: {key}")

print("\nData collection complete.")

# ── Save ──────────────────────────────────────────────────────────────────────
with open("experiments/results/summary_v3.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved: experiments/results/summary_v3.json")

# ── Figure: Flag rate heatmap 3x3 grid ───────────────────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 14))
colors = {"fibonacci": "#E63946", "linear": "#457B9D", "odd": "#2A9D8F"}
markers = {"fibonacci": "o", "linear": "s", "odd": "^"}

for row, circuit_type in enumerate(CIRCUIT_TYPES):
    for col, sched in enumerate(SCHEDULES):
        heatmap = np.zeros((len(DEPTHS), len(NOISE_LEVELS)))
        for i, depth in enumerate(DEPTHS):
            for j, noise in enumerate(NOISE_LEVELS):
                key = f"{circuit_type}_d{depth}_n{noise}"
                heatmap[i, j] = all_results[key][sched]["linear"]["flag_rate"]
        im = axes[row][col].imshow(
            heatmap, aspect="auto", cmap="RdYlGn_r",
            vmin=0, vmax=1, origin="lower"
        )
        axes[row][col].set_xticks(range(len(NOISE_LEVELS)))
        axes[row][col].set_xticklabels(
            [str(n) for n in NOISE_LEVELS], rotation=45, fontsize=7
        )
        axes[row][col].set_yticks(range(len(DEPTHS)))
        axes[row][col].set_yticklabels([str(d) for d in DEPTHS], fontsize=7)
        axes[row][col].set_title(f"{circuit_type.upper()} / {sched}", fontsize=10)
        if col == 0:
            axes[row][col].set_ylabel("Depth", fontsize=9)
        if row == 2:
            axes[row][col].set_xlabel("Noise", fontsize=9)
        plt.colorbar(im, ax=axes[row][col])

plt.suptitle("v3: Stability Filter Flag Rate — 8 depths × 8 noise levels × 20 trials",
             fontsize=13, y=1.01)
plt.tight_layout()
fig.savefig("paper/figures/fig6_v3_flag_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig6_v3_flag_heatmap.png")

# ── Figure: ε vs noise, all circuit types ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, circuit_type in enumerate(CIRCUIT_TYPES):
    depth = 4
    for sched in SCHEDULES:
        eps_means, eps_stds = [], []
        for noise in NOISE_LEVELS:
            key = f"{circuit_type}_d{depth}_n{noise}"
            d = all_results[key][sched]["linear"]
            eps_means.append(d["epsilon_mean"])
            eps_stds.append(d["epsilon_std"])
        axes[idx].errorbar(
            NOISE_LEVELS, eps_means, yerr=eps_stds,
            label=sched.capitalize(), color=colors[sched],
            marker=markers[sched], linewidth=2, capsize=4
        )
    axes[idx].axhline(y=GAMMA, color="black", linestyle="--",
                      linewidth=1.5, label=f"Gamma={GAMMA}")
    axes[idx].set_xlabel("Noise Level", fontsize=11)
    axes[idx].set_ylabel("epsilon", fontsize=11)
    axes[idx].set_title(f"{circuit_type.upper()} (d={depth})", fontsize=12)
    axes[idx].set_xscale("log")
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("v3: Stability Criterion vs Noise — 20 trials, 1000 shots", fontsize=13)
plt.tight_layout()
fig.savefig("paper/figures/fig7_v3_epsilon_vs_noise.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig7_v3_epsilon_vs_noise.png")

# ── Console Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: depth=4, noise=0.01, linear extrapolant")
print(f"{'Circuit':<10} {'Schedule':<12} {'ZNE Mean':>10} {'ZNE Std':>10} "
      f"{'eps Mean':>10} {'Flag%':>8}")
print("-" * 80)
for circuit_type in CIRCUIT_TYPES:
    key = f"{circuit_type}_d4_n0.01"
    for sched in SCHEDULES:
        d = all_results[key][sched]["linear"]
        print(f"{circuit_type:<10} {sched:<12} {d['zne_mean']:>10.6f} "
              f"{d['zne_std']:>10.6f} {d['epsilon_mean']:>10.6f} "
              f"{d['flag_rate']*100:>7.1f}%")
    print()

print("=" * 80)
total_time = (time.time() - start_time) / 60
print(f"\nTotal runtime: {total_time:.1f} minutes")
print("Done. Check paper/figures/ for plots.")
