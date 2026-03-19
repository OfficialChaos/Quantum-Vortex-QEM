"""
run_experiments_v4.py
─────────────────────────────────────────────────────────────────────────────
v4 experiment runner — extended qubit counts, additional circuit types,
additional noise schedules (Lucas, Prime), and adaptive threshold analysis.

Improvements over v3:
  - Qubit counts: 2, 4, 6 qubits (vs 2 only)
  - Circuit types: test, VQE, random, QAOA (vs 3)
  - Schedules: fibonacci, linear, odd, lucas, prime (vs 3)
  - Extrapolants: linear, richardson, poly2, poly3 (vs 3)
  - Adaptive Gamma analysis across threshold values
  - Per-qubit epsilon normalization
  - Checkpoint/resume: saves progress every 20 configs,
    resumes automatically if interrupted

Total runs:
  4 circuit types × 5 schedules × 4 extrapolants
  × 8 depths × 8 noise levels × 10 trials × 3 qubit counts
  = 153,600 runs

Runtime estimate: ~80 hours on personal workstation
Run overnight (or over multiple nights).

Usage:
    python run_experiments_v4.py

    # Run only 2-qubit configs (fastest, ~20 hours):
    python run_experiments_v4.py --qubits 2

    # Run only new schedules:
    python run_experiments_v4.py --schedules lucas prime

    # Run only new circuit types:
    python run_experiments_v4.py --circuits qaoa

    # Force fresh start (ignore checkpoint):
    python run_experiments_v4.py --fresh
"""

import sys
import os
import argparse
import time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import cirq
from itertools import product

from src.zne_pipeline import run_pipeline, noisy_executor
from src.stability_filter import StabilityFilter
from src.fibonacci_scaling import compare_schedules

# ── Argument Parser ───────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="v4 Experiment Runner")
parser.add_argument("--qubits",    nargs="+", type=int,
                    default=[2, 4, 6], help="Qubit counts to test")
parser.add_argument("--circuits",  nargs="+",
                    default=["test", "vqe", "random", "qaoa"],
                    help="Circuit types")
parser.add_argument("--schedules", nargs="+",
                    default=["fibonacci", "linear", "odd", "lucas", "prime"],
                    help="Noise schedules")
parser.add_argument("--fresh", action="store_true",
                    help="Ignore checkpoint and start fresh")
args = parser.parse_args()

# ── Output directories ────────────────────────────────────────────────────────
os.makedirs("experiments/results", exist_ok=True)
os.makedirs("paper/figures", exist_ok=True)

# ── Checkpoint paths ──────────────────────────────────────────────────────────
CHECKPOINT_PATH = "experiments/results/checkpoint_v4.json"
SUMMARY_PATH    = "experiments/results/summary_v4.json"

# ── Experiment Parameters ─────────────────────────────────────────────────────
DEPTHS         = [2, 4, 6, 8, 10, 12, 16, 20]
NOISE_LEVELS   = [0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
QUBIT_COUNTS   = args.qubits
CIRCUIT_TYPES  = args.circuits
SCHEDULES      = args.schedules
EXTRAPOLANTS   = ["linear", "richardson", "poly2", "poly3"]
N_SCALES       = 6
GAMMA          = 0.05
GAMMA_SWEEP    = [0.01, 0.02, 0.05, 0.1, 0.2]
NU             = 0.5
N_TRIALS       = 10
N_SHOTS        = 500
CHECKPOINT_EVERY = 20   # save progress every N configs

total_configs = (len(CIRCUIT_TYPES) * len(SCHEDULES) * len(EXTRAPOLANTS)
                 * len(DEPTHS) * len(NOISE_LEVELS) * len(QUBIT_COUNTS))
total_runs    = total_configs * N_TRIALS

print("=" * 70)
print("Quantum-Vortex QEM — v4 Experiment Runner")
print(f"  Qubit counts   : {QUBIT_COUNTS}")
print(f"  Circuit types  : {CIRCUIT_TYPES}")
print(f"  Schedules      : {SCHEDULES}")
print(f"  Extrapolants   : {EXTRAPOLANTS}")
print(f"  Depths         : {DEPTHS}")
print(f"  Noise levels   : {NOISE_LEVELS}")
print(f"  Trials         : {N_TRIALS}")
print(f"  Shots          : {N_SHOTS}")
print(f"  Total configs  : {total_configs:,}")
print(f"  Total runs     : {total_runs:,}")
print("=" * 70)

# ── Checkpoint: Load or Fresh Start ──────────────────────────────────────────
def load_checkpoint():
    """
    Load checkpoint and/or existing summary_v4.json.

    Priority:
      1. --fresh flag: ignore everything, start clean
      2. Checkpoint file: resume interrupted run (has completed_keys)
      3. Existing summary_v4.json: merge with previous completed runs
      4. Nothing found: start fresh

    This ensures runs ALWAYS merge, never overwrite previous results.
    """
    if args.fresh:
        print("  [checkpoint] --fresh flag set, starting from scratch.")
        return {}, set()

    # ── Try checkpoint first (mid-run resume) ────────────────────────────────
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, "r") as f:
                ckpt = json.load(f)
            summary   = ckpt.get("summary", {})
            completed = set(ckpt.get("completed_keys", []))
            print(f"  [checkpoint] Resuming from checkpoint — {len(completed)} configs done.")
            return summary, completed
        except Exception as e:
            print(f"  [checkpoint] Failed to load checkpoint ({e}), trying summary file.")

    # ── No checkpoint — load existing summary_v4.json if present ─────────────
    if os.path.exists(SUMMARY_PATH):
        try:
            with open(SUMMARY_PATH, "r") as f:
                summary = json.load(f)
            # Reconstruct completed keys from summary contents
            completed = set(summary.keys())
            print(f"  [checkpoint] Loaded existing summary_v4.json — "
                  f"{len(completed)} configs already complete. Merging.")
            return summary, completed
        except Exception as e:
            print(f"  [checkpoint] Failed to load summary_v4.json ({e}), starting fresh.")

    print("  [checkpoint] No checkpoint or summary found, starting fresh.")
    return {}, set()

def save_checkpoint(summary, completed_keys):
    """Atomic save to checkpoint file."""
    ckpt = {
        "summary":        summary,
        "completed_keys": list(completed_keys),
        "timestamp":      time.strftime("%Y-%m-%d %H:%M:%S"),
        "configs_done":   len(completed_keys),
    }
    tmp_path = CHECKPOINT_PATH + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(ckpt, f, indent=2)
    os.replace(tmp_path, CHECKPOINT_PATH)

summary, completed_keys = load_checkpoint()

# ── Schedule Generators ───────────────────────────────────────────────────────
def get_scale_factors(schedule, n):
    """Return n scale factors for the given schedule name."""
    if schedule == "fibonacci":
        seq = [1, 2]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        return seq[:n]
    elif schedule == "linear":
        return list(range(1, n + 1))
    elif schedule == "odd":
        return [2 * i + 1 for i in range(n)]
    elif schedule == "lucas":
        seq = [2, 1]
        while len(seq) < n:
            seq.append(seq[-1] + seq[-2])
        seq_sorted = sorted(seq[:n])
        if seq_sorted[0] != 1:
            seq_sorted = [1] + seq_sorted[:n-1]
        return seq_sorted
    elif schedule == "prime":
        primes = [1, 2, 3, 5, 7, 11, 13, 17, 19, 23]
        return primes[:n]
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

# ── Circuit Builders ──────────────────────────────────────────────────────────
def make_test_circuit(n_qubits, depth):
    """Alternating Rx + CNOT — same as v1/v2/v3."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for d in range(depth):
        circuit.append([cirq.rx(np.pi / 4)(q) for q in qubits])
        for i in range(0, n_qubits - 1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append([cirq.measure(q) for q in qubits])
    return circuit, qubits

def make_vqe_circuit(n_qubits, depth):
    """Hardware-efficient VQE ansatz — Ry + CZ."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    for d in range(depth):
        angles = np.random.uniform(0, 2 * np.pi, n_qubits)
        circuit.append([cirq.ry(angles[i])(qubits[i]) for i in range(n_qubits)])
        for i in range(0, n_qubits - 1, 2):
            circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    circuit.append([cirq.measure(q) for q in qubits])
    return circuit, qubits

def make_random_circuit(n_qubits, depth):
    """Random single-qubit gates + CNOT layers."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    gates = [cirq.rx, cirq.ry, cirq.rz]
    for d in range(depth):
        for q in qubits:
            gate = np.random.choice(gates)
            circuit.append(gate(np.random.uniform(0, 2 * np.pi))(q))
        for i in range(0, n_qubits - 1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    circuit.append([cirq.measure(q) for q in qubits])
    return circuit, qubits

def make_qaoa_circuit(n_qubits, depth):
    """QAOA-inspired: alternating ZZ problem + Rx mixer layers."""
    qubits = cirq.LineQubit.range(n_qubits)
    circuit = cirq.Circuit()
    circuit.append([cirq.H(q) for q in qubits])
    for d in range(depth):
        gamma_p = np.random.uniform(0, np.pi)
        for i in range(n_qubits - 1):
            circuit.append(cirq.ZZPowGate(exponent=gamma_p / np.pi)(
                qubits[i], qubits[i + 1]))
        beta = np.random.uniform(0, np.pi)
        circuit.append([cirq.rx(2 * beta)(q) for q in qubits])
    circuit.append([cirq.measure(q) for q in qubits])
    return circuit, qubits

CIRCUIT_BUILDERS = {
    "test":   make_test_circuit,
    "vqe":    make_vqe_circuit,
    "random": make_random_circuit,
    "qaoa":   make_qaoa_circuit,
}

# ── Executor ──────────────────────────────────────────────────────────────────
def noisy_executor_v4(circuit, noise_level, n_shots, n_qubits):
    """
    Shot-based density matrix simulation with depolarizing noise.

    Observable: average nearest-neighbour ZZ over all adjacent pairs.
      For n qubits: mean( <Z0Z1>, <Z1Z2>, ..., <Z(n-2)Z(n-1)> )

    Backward compatible:
      q=2: 1 pair only, reduces exactly to original <Z0Z1>.
      q>2: captures how noise accumulates across circuit width.

    Range: always in [-1, +1] regardless of qubit count.
    """
    noise_level = min(noise_level, 0.24)
    noise_model = cirq.ConstantQubitNoiseModel(cirq.depolarize(noise_level))
    simulator = cirq.DensityMatrixSimulator(noise=noise_model)
    result = simulator.run(circuit, repetitions=n_shots)

    # Read all qubit measurements -> shape (n_qubits, n_shots)
    z_vals = np.array([
        1 - 2 * result.measurements[str(cirq.LineQubit(i))].flatten().astype(float)
        for i in range(n_qubits)
    ])

    # ZZ for each adjacent pair -> shape (n_qubits-1, n_shots)
    zz_pairs = np.array([
        z_vals[i] * z_vals[i + 1]
        for i in range(n_qubits - 1)
    ])

    # Average over pairs per shot, then average over shots
    return float(np.mean(np.mean(zz_pairs, axis=0)))

# ── ZNE Extrapolants ──────────────────────────────────────────────────────────
def extrapolate(scale_factors, values, method):
    """Extrapolate to zero noise using the specified method."""
    x = np.array(scale_factors, dtype=float)
    y = np.array(values, dtype=float)
    if method == "linear":
        coeffs = np.polyfit(x, y, 1)
        return float(np.polyval(coeffs, 0))
    elif method == "richardson":
        if len(x) < 2:
            return float(y[0])
        return float((x[1] * y[0] - x[0] * y[1]) / (x[1] - x[0]))
    elif method == "poly2":
        coeffs = np.polyfit(x, y, 1 if len(x) < 3 else 2)
        return float(np.polyval(coeffs, 0))
    elif method == "poly3":
        deg = min(3, len(x) - 1)
        coeffs = np.polyfit(x, y, deg)
        return float(np.polyval(coeffs, 0))
    else:
        raise ValueError(f"Unknown extrapolant: {method}")

# ── Stability Filter ──────────────────────────────────────────────────────────
def compute_epsilon(scale_factors, values, nu=0.5):
    """Compute Navier-Stokes energy dissipation norm."""
    x = np.array(scale_factors, dtype=float)
    y = np.array(values, dtype=float)
    epsilon = 0.0
    for k in range(len(x) - 1):
        grad = (y[k + 1] - y[k]) / (x[k + 1] - x[k])
        delta = x[k + 1] - x[k]
        epsilon += grad ** 2 * delta
    return float(nu * epsilon)

# ── Main Experiment Loop ──────────────────────────────────────────────────────
start_time = time.time()
configs_this_session = 0
preloaded = len(completed_keys)   # configs already done from previous runs

# Count how many configs THIS run will actually skip (keys that overlap)
all_this_run = set(
    f"q{nq}_{ct}_d{d}_n{n}"
    for nq, ct, d, n in product(QUBIT_COUNTS, CIRCUIT_TYPES, DEPTHS, NOISE_LEVELS)
)
will_skip = len(all_this_run & completed_keys)
total_this_session = len(all_this_run) - will_skip

print(f"  [progress] {preloaded} configs pre-loaded total, "
      f"{will_skip} overlap with this run (skipping), "
      f"{total_this_session} new configs to run.")

for n_qubits, circuit_type, depth, noise in product(
        QUBIT_COUNTS, CIRCUIT_TYPES, DEPTHS, NOISE_LEVELS):

    key = f"q{n_qubits}_{circuit_type}_d{depth}_n{noise}"

    # ── Skip already-completed configs ───────────────────────────────────────
    if key in completed_keys:
        continue

    if key not in summary:
        summary[key] = {}

    builder = CIRCUIT_BUILDERS[circuit_type]

    for schedule in SCHEDULES:
        scale_factors = get_scale_factors(schedule, N_SCALES)
        summary[key][schedule] = {}

        for extrapolant in EXTRAPOLANTS:
            zne_vals     = []
            epsilon_vals = []
            flag_counts  = {g: 0 for g in GAMMA_SWEEP}

            for trial in range(N_TRIALS):
                circuit, qubits = builder(n_qubits, depth)
                measured = []
                for sf in scale_factors:
                    scaled_noise = min(noise * sf, 0.24)
                    val = noisy_executor_v4(circuit, scaled_noise,
                                           N_SHOTS, n_qubits)
                    measured.append(val)

                zne_val = extrapolate(scale_factors, measured, extrapolant)
                epsilon = compute_epsilon(scale_factors, measured, NU)

                zne_vals.append(zne_val)
                epsilon_vals.append(epsilon)
                for g in GAMMA_SWEEP:
                    if epsilon >= g:
                        flag_counts[g] += 1

            summary[key][schedule][extrapolant] = {
                "n_qubits":      n_qubits,
                "circuit_type":  circuit_type,
                "depth":         depth,
                "noise":         noise,
                "schedule":      schedule,
                "extrapolant":   extrapolant,
                "zne_mean":      float(np.mean(zne_vals)),
                "zne_std":       float(np.std(zne_vals)),
                "epsilon_mean":  float(np.mean(epsilon_vals)),
                "epsilon_std":   float(np.std(epsilon_vals)),
                "flag_rates":    {str(g): flag_counts[g] / N_TRIALS
                                  for g in GAMMA_SWEEP},
            }

    # ── Mark complete & checkpoint ────────────────────────────────────────────
    completed_keys.add(key)
    configs_this_session += 1

    elapsed = time.time() - start_time
    rate = configs_this_session / elapsed if elapsed > 0 else 0
    remaining = total_this_session - configs_this_session
    eta_mins = (remaining / rate / 60) if rate > 0 else 0

    if configs_this_session % CHECKPOINT_EVERY == 0:
        print(f"  [{configs_this_session}/{total_this_session}]"
              f"  q={n_qubits} {circuit_type} d={depth} n={noise}"
              f"  ETA: {eta_mins:.0f} min  [checkpoint saved]")
        save_checkpoint(summary, completed_keys)
    elif configs_this_session <= 5 or configs_this_session % 5 == 0:
        print(f"  [{configs_this_session}/{total_this_session}]"
              f"  q={n_qubits} {circuit_type} d={depth} n={noise}"
              f"  ETA: {eta_mins:.0f} min")

# ── Final Save ────────────────────────────────────────────────────────────────
save_checkpoint(summary, completed_keys)
with open(SUMMARY_PATH, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSaved: {SUMMARY_PATH}")
print(f"Total runtime this session: {(time.time() - start_time) / 60:.1f} minutes")

# ── Figures ───────────────────────────────────────────────────────────────────
print("\nGenerating figures...")

if 2 in QUBIT_COUNTS:
    fig, axes = plt.subplots(1, len(CIRCUIT_TYPES),
                             figsize=(5 * len(CIRCUIT_TYPES), 5), sharey=True)
    if len(CIRCUIT_TYPES) == 1:
        axes = [axes]

    for idx, ct in enumerate(CIRCUIT_TYPES):
        for sched in SCHEDULES:
            eps_means, eps_stds = [], []
            for noise in NOISE_LEVELS:
                key = f"q2_{ct}_d4_n{noise}"
                if key in summary and sched in summary[key]:
                    vals = [summary[key][sched][ext]["epsilon_mean"]
                            for ext in EXTRAPOLANTS if ext in summary[key][sched]]
                    eps_means.append(np.mean(vals) if vals else 0)
                    eps_stds.append(np.std(vals) if vals else 0)
                else:
                    eps_means.append(0); eps_stds.append(0)
            axes[idx].errorbar(NOISE_LEVELS, eps_means, yerr=eps_stds,
                               label=sched.capitalize(), marker="o",
                               linewidth=2, capsize=4)
        axes[idx].axhline(y=GAMMA, color="black", linestyle="--",
                          linewidth=1.5, label=f"Γ={GAMMA}")
        axes[idx].set_xscale("log")
        axes[idx].set_xlabel("Noise Level", fontsize=11)
        axes[idx].set_title(f"{ct.upper()} (2 qubits)", fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
    axes[0].set_ylabel("ε (Energy Dissipation Norm)", fontsize=11)
    plt.suptitle("v4: ε vs Noise — All Circuit Types, d=4, 2 qubits", fontsize=13)
    plt.tight_layout()
    fig.savefig("paper/figures/fig8_v4_epsilon_vs_noise_all_circuits.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: paper/figures/fig8_v4_epsilon_vs_noise_all_circuits.png")

sched_colors = {
    "fibonacci": "#E63946", "linear": "#457B9D", "odd": "#2A9D8F",
    "lucas": "#F4A261", "prime": "#9B5DE5",
}
fig, axes = plt.subplots(1, len(CIRCUIT_TYPES),
                         figsize=(5 * len(CIRCUIT_TYPES), 5), sharey=True)
if len(CIRCUIT_TYPES) == 1:
    axes = [axes]
for idx, ct in enumerate(CIRCUIT_TYPES):
    for sched in SCHEDULES:
        if sched not in sched_colors:
            continue
        eps_means = []
        for noise in NOISE_LEVELS:
            key = f"q2_{ct}_d4_n{noise}"
            if key in summary and sched in summary[key]:
                vals = [summary[key][sched][ext]["epsilon_mean"]
                        for ext in EXTRAPOLANTS if ext in summary[key][sched]]
                eps_means.append(np.mean(vals) if vals else 0)
            else:
                eps_means.append(0)
        axes[idx].plot(NOISE_LEVELS, eps_means, label=sched.capitalize(),
                       color=sched_colors[sched], marker="o", linewidth=2)
    axes[idx].axhline(y=GAMMA, color="black", linestyle="--",
                      linewidth=1.5, label=f"Γ={GAMMA}")
    axes[idx].set_xscale("log")
    axes[idx].set_xlabel("Noise Level", fontsize=11)
    axes[idx].set_title(f"{ct.upper()}", fontsize=12)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].legend(fontsize=8)
axes[0].set_ylabel("ε (Energy Dissipation Norm)", fontsize=11)
plt.suptitle("v4: All 5 Schedules — ε vs Noise, d=4, 2 qubits", fontsize=13)
plt.tight_layout()
fig.savefig("paper/figures/fig9_v4_schedule_comparison.png",
            dpi=300, bbox_inches="tight")
plt.close()
print("Saved: paper/figures/fig9_v4_schedule_comparison.png")

if len(QUBIT_COUNTS) > 1:
    qubit_colors = {2: "#E63946", 4: "#457B9D", 6: "#2A9D8F"}
    fig, axes = plt.subplots(1, len(CIRCUIT_TYPES),
                             figsize=(5 * len(CIRCUIT_TYPES), 5), sharey=True)
    if len(CIRCUIT_TYPES) == 1:
        axes = [axes]
    for idx, ct in enumerate(CIRCUIT_TYPES):
        for nq in QUBIT_COUNTS:
            eps_means = []
            for noise in NOISE_LEVELS:
                key = f"q{nq}_{ct}_d4_n{noise}"
                if key in summary and "fibonacci" in summary[key]:
                    vals = [summary[key]["fibonacci"][ext]["epsilon_mean"]
                            for ext in EXTRAPOLANTS
                            if ext in summary[key]["fibonacci"]]
                    eps_means.append(np.mean(vals) if vals else 0)
                else:
                    eps_means.append(0)
            axes[idx].plot(NOISE_LEVELS, eps_means, label=f"{nq} qubits",
                           color=qubit_colors.get(nq, "gray"),
                           marker="o", linewidth=2)
        axes[idx].axhline(y=GAMMA, color="black", linestyle="--",
                          linewidth=1.5, label=f"Γ={GAMMA}")
        axes[idx].set_xscale("log")
        axes[idx].set_xlabel("Noise Level", fontsize=11)
        axes[idx].set_title(f"{ct.upper()}", fontsize=12)
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(fontsize=8)
    axes[0].set_ylabel("ε (Energy Dissipation Norm)", fontsize=11)
    plt.suptitle("v4: Qubit Count Scaling — Fibonacci, d=4", fontsize=13)
    plt.tight_layout()
    fig.savefig("paper/figures/fig10_v4_qubit_scaling.png",
                dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved: paper/figures/fig10_v4_qubit_scaling.png")

# ── Console Summary ───────────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("v4 SUMMARY — q=2, d=4, n=0.01, linear extrapolant")
print(f"{'Circuit':<10} {'Schedule':<12} {'ZNE Mean':>10} "
      f"{'ZNE Std':>10} {'ε Mean':>10} {'Flag%':>8}")
print("-" * 75)
for ct in CIRCUIT_TYPES:
    key = f"q2_{ct}_d4_n0.01"
    if key not in summary:
        continue
    for sched in SCHEDULES:
        if sched not in summary[key]:
            continue
        if "linear" not in summary[key][sched]:
            continue
        d = summary[key][sched]["linear"]
        flag_pct = d["flag_rates"].get(str(GAMMA), 0) * 100
        print(f"{ct:<10} {sched:<12} {d['zne_mean']:>10.5f} "
              f"{d['zne_std']:>10.5f} {d['epsilon_mean']:>10.5f} "
              f"{flag_pct:>7.1f}%")
print("=" * 75)
print(f"\nTotal runtime this session: {(time.time() - start_time) / 60:.1f} minutes")
print("Done. Results saved to experiments/results/summary_v4.json")
print("Figures saved to paper/figures/")

# ── Remove checkpoint on clean completion ─────────────────────────────────────
if os.path.exists(CHECKPOINT_PATH):
    os.remove(CHECKPOINT_PATH)
    print("Checkpoint file removed (run complete).")
