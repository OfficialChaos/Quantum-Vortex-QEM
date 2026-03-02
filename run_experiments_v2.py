"""
run_experiments_v2.py
---------------------
Enhanced experiment runner with:
1. Larger parameter sweep
2. Multiple extrapolants (linear, Richardson, polynomial)
3. Multiple circuit types (test, VQE ansatz, random)

Target runtime: 2-3 hours on modern CPU
Run from project root: python run_experiments_v2.py
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
DEPTHS       = [2, 4, 6, 8, 10, 12]
NOISE_LEVELS = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1]
N_SCALES     = 6
GAMMA        = 0.05
NU           = 0.5
N_SHOTS      = 300
N_TRIALS     = 5
SCHEDULES    = ["fibonacci", "linear", "odd"]
EXTRAPOLANTS = ["linear", "richardson", "poly2"]
CIRCUIT_TYPES = ["test", "vqe", "random"]

total_configs = len(DEPTHS) * len(NOISE_LEVELS) * N_TRIALS * len(CIRCUIT_TYPES)
print("=" * 65)
print("Quantum-Vortex QEM: Enhanced Experiment Runner v2")
print(f"Depths         : {DEPTHS}")
print(f"Noise levels   : {NOISE_LEVELS}")
print(f"Trials         : {N_TRIALS}")
print(f"Circuit types  : {CIRCUIT_TYPES}")
print(f"Extrapolants   : {EXTRAPOLANTS}")
print(f"Total configs  : {total_configs}")
print("=" * 65)

# ── Circuit Builders ──────────────────────────────────────────────────────────

def make_test_circuit(depth: int) -> cirq.Circuit:
    """Alternating Rx + CNOT. Ideal <ZZ> = cos(depth*pi/4)."""
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    for _ in range(depth):
        circuit.append([cirq.rx(np.pi / 4)(q0), cirq.rx(np.pi / 4)(q1), cirq.CNOT(q0, q1)])
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit


def make_vqe_circuit(depth: int) -> cirq.Circuit:
    """
    Hardware-efficient VQE ansatz: layers of Ry rotations and CZ gates.
    Mimics real variational circuits used in quantum chemistry.
    """
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    # Random but fixed angles (seed for reproducibility)
    rng = np.random.RandomState(42)
    for _ in range(depth):
        theta0 = rng.uniform(0, 2 * np.pi)
        theta1 = rng.uniform(0, 2 * np.pi)
        circuit.append([cirq.ry(theta0)(q0), cirq.ry(theta1)(q1), cirq.CZ(q0, q1)])
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit


def make_random_circuit(depth: int, seed: int = 0) -> cirq.Circuit:
    """
    Random circuit with varied single-qubit gates and entanglers.
    Tests filter generality beyond structured circuits.
    """
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit()
    rng = np.random.RandomState(seed)
    gates = [cirq.rx, cirq.ry, cirq.rz]
    for _ in range(depth):
        g0 = gates[rng.randint(3)]
        g1 = gates[rng.randint(3)]
        t0 = rng.uniform(0, 2 * np.pi)
        t1 = rng.uniform(0, 2 * np.pi)
        circuit.append([g0(t0)(q0), g1(t1)(q1)])
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

def noisy_executor(circuit: cirq.Circuit,
                   noise_level: float,
                   n_shots: int = 500) -> float:
    """Shot-based depolarizing simulation returning <ZZ>."""
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
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(noisy, repetitions=n_shots)

    bits = result.measurements['result']
    z0 = 1 - 2 * bits[:, 0].astype(float)
    z1 = 1 - 2 * bits[:, 1].astype(float)
    return float(np.mean(z0 * z1))

# ── Extrapolants ──────────────────────────────────────────────────────────────

def extrapolate(ev_array: np.ndarray,
                sf_array: np.ndarray,
                method: str = "linear") -> float:
    """
    Extrapolate to zero noise using specified method.

    linear     : degree-1 polynomial fit
    richardson : Richardson extrapolation (first order)
    poly2      : degree-2 polynomial fit
    """
    if method == "linear":
        coeffs = np.polyfit(sf_array, ev_array, deg=1)
        return float(np.polyval(coeffs, 0.0))

    elif method == "richardson":
        # Richardson: weighted combination of first two points
        # E(0) ≈ (s2*E1 - s1*E2) / (s2 - s1)
        s1, s2 = sf_array[0], sf_array[1]
        e1, e2 = ev_array[0], ev_array[1]
        if abs(s2 - s1) < 1e-10:
            return float(e1)
        return float((s2 * e1 - s1 * e2) / (s2 - s1))

    elif method == "poly2":
        if len(sf_array) < 3:
            coeffs = np.polyfit(sf_array, ev_array, deg=1)
        else:
            coeffs = np.polyfit(sf_array, ev_array, deg=2)
        return float(np.polyval(coeffs, 0.0))

    else:
        raise ValueError(f"Unknown extrapolant: {method}")

# ── Main Experiment Loop ──────────────────────────────────────────────────────

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
            trial_zne    = []
            trial_eps    = []
            trial_flagged = []

            for trial in range(N_TRIALS):
                ev_array = np.array([
                    noisy_executor(circuit, noise * s, N_SHOTS)
                    for s in scales
                ])

                zne_val = extrapolate(ev_array, sf_array, method=extrap)
                stab    = sf_filter.evaluate(ev_array, sf_array)

                trial_zne.append(zne_val)
                trial_eps.append(stab.epsilon)
                trial_flagged.append(stab.flagged)

            all_results[key][schedule][extrap] = {
                "circuit_type": circuit_type,
                "depth":        depth,
                "noise":        noise,
                "schedule":     schedule,
                "extrapolant":  extrap,
                "zne_mean":     float(np.mean(trial_zne)),
                "zne_std":      float(np.std(trial_zne)),
                "epsilon_mean": float(np.mean(trial_eps)),
                "epsilon_std":  float(np.std(trial_eps)),
                "flag_rate":    float(np.mean(trial_flagged)),
            }

    done += 1
    elapsed = time.time() - start_time
    total = len(CIRCUIT_TYPES) * len(DEPTHS) * len(NOISE_LEVELS)
    pct = 100 * done // total
    eta = (elapsed / done) * (total - done) if done > 0 else 0
    print(f"  {pct:3d}% ({done}/{total}) | "
          f"elapsed: {elapsed/60:.1f}m | "
          f"ETA: {eta/60:.1f}m | "
          f"last: {key}")

print("\nData collection complete.")

# ── Save Results ──────────────────────────────────────────────────────────────
with open("experiments/results/summary_v2.json", "w") as f:
    json.dump(all_results, f, indent=2)
print("Saved: experiments/results/summary_v2.json")

# ── Figure 1: ε vs Noise, all circuit types, linear extrapolant ───────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
colors  = {"fibonacci": "#E63946", "linear": "#457B9D", "odd": "#2A9D8F"}
markers = {"fibonacci": "o",       "linear": "s",       "odd": "^"}

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
                      linewidth=1.5, label=f"Γ={GAMMA}")
    axes[idx].set_xlabel("Noise Level", fontsize=11)
    axes[idx].set_ylabel("ε", fontsize=11)
    axes[idx].set_title(f"{circuit_type.upper()} circuit (d={depth})", fontsize=12)
    axes[idx].set_xscale("log")
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3)

plt.suptitle("Stability Criterion ε vs Noise Level — Three Circuit Types", fontsize=13)
plt.tight_layout()
fig.savefig("paper/figures/fig3_circuit_type_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig3_circuit_type_comparison.png")

# ── Figure 2: ZNE accuracy by extrapolant, depth=4, noise=0.01 ───────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, circuit_type in enumerate(CIRCUIT_TYPES):
    key = f"{circuit_type}_d4_n0.01"
    x = np.arange(len(SCHEDULES))
    width = 0.25
    extrap_colors = {"linear": "#457B9D", "richardson": "#E63946", "poly2": "#2A9D8F"}

    for i, extrap in enumerate(EXTRAPOLANTS):
        means = [all_results[key][s][extrap]["zne_mean"] for s in SCHEDULES]
        stds  = [all_results[key][s][extrap]["zne_std"]  for s in SCHEDULES]
        axes[idx].bar(x + i * width, means, width,
                      yerr=stds, label=extrap,
                      color=extrap_colors[extrap], alpha=0.8, capsize=4)

    axes[idx].set_xticks(x + width)
    axes[idx].set_xticklabels([s.capitalize() for s in SCHEDULES])
    axes[idx].set_ylabel("ZNE Extrapolated Value", fontsize=11)
    axes[idx].set_title(f"{circuit_type.upper()} — d=4, p=0.01", fontsize=12)
    axes[idx].legend(fontsize=9)
    axes[idx].grid(True, alpha=0.3, axis='y')
    axes[idx].axhline(y=0, color='black', linewidth=0.5)

plt.suptitle("ZNE Accuracy: Schedule × Extrapolant × Circuit Type", fontsize=13)
plt.tight_layout()
fig.savefig("paper/figures/fig4_extrapolant_comparison.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig4_extrapolant_comparison.png")

# ── Figure 3: Flag rate heatmap across all circuit types ─────────────────────
fig, axes = plt.subplots(3, 3, figsize=(15, 12))

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
        axes[row][col].set_xticklabels([str(n) for n in NOISE_LEVELS],
                                        rotation=45, fontsize=7)
        axes[row][col].set_yticks(range(len(DEPTHS)))
        axes[row][col].set_yticklabels([str(d) for d in DEPTHS], fontsize=7)
        axes[row][col].set_title(f"{circuit_type.upper()} / {sched}", fontsize=10)
        if col == 0:
            axes[row][col].set_ylabel("Depth", fontsize=9)
        if row == 2:
            axes[row][col].set_xlabel("Noise", fontsize=9)
        plt.colorbar(im, ax=axes[row][col])

plt.suptitle("Stability Filter Flag Rate: Circuit × Schedule", fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig("paper/figures/fig5_full_flag_heatmap.png", dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig5_full_flag_heatmap.png")

# ── Console Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("SUMMARY: depth=4, noise=0.01, linear extrapolant")
print(f"{'Circuit':<10} {'Schedule':<12} {'ZNE Mean':>10} {'ZNE Std':>10} "
      f"{'ε Mean':>10} {'Flag%':>8}")
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
