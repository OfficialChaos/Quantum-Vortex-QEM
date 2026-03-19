"""
noise_robustness_v4.py
======================
Supplemental robustness experiment for Quantum-Vortex QEM v0.4.0

Tests whether the stability filter's qualitative behaviour --
(1) ε hierarchy VQE > QAOA > random > test
(2) zero flag rates at low noise/depth
(3) monotone qubit scaling
-- holds under two alternative noise models:
   (A) Amplitude damping  (T1-type decay, common on superconducting hardware)
   (B) Correlated depolarising  (two-qubit correlated channel on each CNOT/CZ pair)

Scope: test + VQE circuits, Fibonacci + linear schedules, n in {2,4},
       d=4, p in {0.001, 0.005, 0.01, 0.02, 0.05}, 10 trials, N=500 shots.
       Intentionally smaller than main experiment -- this is a qualitative check.

Output: results_robustness.json  (same schema as summary_v4.json)
        Prints a summary table to stdout on completion.

Runtime estimate: ~20-40 minutes on a Ryzen workstation.

Usage:
    python noise_robustness_v4.py
    python noise_robustness_v4.py --trials 5 --shots 200   # faster test run

Author: Shawn G. Kleipe  |  v0.4.0  |  MIT License
"""

import argparse
import json
import time
import warnings
from itertools import product
from typing import Dict, List, Tuple

import numpy as np
import cirq

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Noise robustness check for QEM stability filter")
    p.add_argument("--trials",  type=int, default=10,  help="Trials per configuration (default 10)")
    p.add_argument("--shots",   type=int, default=500, help="Shots per trial (default 500)")
    p.add_argument("--outfile", type=str, default="results_robustness.json")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Circuit builders  (identical to main experiment v4)
# ─────────────────────────────────────────────────────────────────────────────

def build_test_circuit(n: int, depth: int) -> cirq.Circuit:
    """Alternating Rx(pi/4) + CNOT layers."""
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    for _ in range(depth):
        circuit.append([cirq.rx(np.pi / 4)(q) for q in qubits])
        for i in range(0, n - 1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


def build_vqe_circuit(n: int, depth: int, rng: np.random.Generator) -> cirq.Circuit:
    """Hardware-efficient Ry + CZ ansatz with random angles."""
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    for _ in range(depth):
        angles = rng.uniform(0, 2 * np.pi, n)
        circuit.append([cirq.ry(angles[i])(qubits[i]) for i in range(n)])
        for i in range(0, n - 1, 2):
            circuit.append(cirq.CZ(qubits[i], qubits[i + 1]))
    return circuit


def build_random_circuit(n: int, depth: int, rng: np.random.Generator) -> cirq.Circuit:
    """Randomised single-qubit gates + CNOT."""
    qubits = cirq.LineQubit.range(n)
    gate_set = [cirq.X, cirq.Y, cirq.Z, cirq.H, cirq.S, cirq.T]
    circuit = cirq.Circuit()
    for _ in range(depth):
        for q in qubits:
            gate = gate_set[rng.integers(len(gate_set))]
            circuit.append(gate(q))
        for i in range(0, n - 1, 2):
            circuit.append(cirq.CNOT(qubits[i], qubits[i + 1]))
    return circuit


def build_qaoa_circuit(n: int, depth: int, rng: np.random.Generator) -> cirq.Circuit:
    """QAOA: alternating ZZ problem-unitary + Rx mixer."""
    qubits = cirq.LineQubit.range(n)
    circuit = cirq.Circuit()
    for _ in range(depth):
        gamma = rng.uniform(0, np.pi)
        beta  = rng.uniform(0, np.pi)
        for i in range(n - 1):
            circuit.append(cirq.ZZPowGate(exponent=gamma / np.pi)(qubits[i], qubits[i + 1]))
        circuit.append([cirq.rx(2 * beta)(q) for q in qubits])
    return circuit


CIRCUIT_BUILDERS = {
    "test":   build_test_circuit,
    "vqe":    build_vqe_circuit,
    "random": build_random_circuit,
    "qaoa":   build_qaoa_circuit,
}

# ─────────────────────────────────────────────────────────────────────────────
# Noise models
# ─────────────────────────────────────────────────────────────────────────────

def depolarising_model(p: float) -> cirq.ConstantQubitNoiseModel:
    """Standard single-qubit depolarising (baseline, matches main experiment)."""
    p_capped = min(p, 0.24)
    return cirq.ConstantQubitNoiseModel(cirq.depolarize(p_capped))


def amplitude_damping_model(p: float) -> cirq.ConstantQubitNoiseModel:
    """
    Amplitude damping with damping parameter gamma.
    Models T1 decay (energy relaxation) common on superconducting qubits.
    gamma is chosen to give approximately the same single-qubit error rate as
    the depolarising model: for small p, amplitude damping with gamma=p
    gives average gate fidelity ~ 1 - p/2, comparable to depolarising.
    We use gamma = min(p, 0.99) to keep physically valid.
    """
    gamma = min(p, 0.99)
    return cirq.ConstantQubitNoiseModel(cirq.amplitude_damp(gamma))


class CorrelatedDepolarisingModel(cirq.NoiseModel):
    """
    Correlated two-qubit depolarising noise.
    After every two-qubit gate (CNOT, CZ, ZZPow), applies a two-qubit
    depolarising channel with strength p_2q = 1-(1-p)^2 to the pair.
    Single-qubit gates receive the standard single-qubit channel at rate p.
    This models hardware where two-qubit gates are the dominant error source.
    """
    def __init__(self, p: float):
        self.p1 = min(p, 0.24)
        self.p2 = min(1 - (1 - p) ** 2, 0.99)  # effective two-qubit rate

    def noisy_operation(self, op: cirq.Operation) -> cirq.OP_TREE:
        yield op
        if len(op.qubits) == 2:
            # Two-qubit depolarising channel
            yield cirq.depolarize(self.p2, n_qubits=2).on(*op.qubits)
        else:
            for q in op.qubits:
                yield cirq.depolarize(self.p1).on(q)


NOISE_MODELS = {
    "depolarising":          depolarising_model,
    "amplitude_damping":     amplitude_damping_model,
    "correlated_depolarising": lambda p: CorrelatedDepolarisingModel(p),
}

# ─────────────────────────────────────────────────────────────────────────────
# Observable and executor
# ─────────────────────────────────────────────────────────────────────────────

def measure_Z0Z1(circuit: cirq.Circuit, noise_model, n_shots: int,
                 rng: np.random.Generator) -> float:
    """
    Measure <Z0 Z1> via shot-based simulation.
    Matches observable definition in v0.4.0 paper (Eq. 5, corrected).
    """
    qubits = sorted(circuit.all_qubits())[:2]
    meas_circuit = circuit + cirq.measure(*qubits, key="m")
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(meas_circuit.moments, qubits=sorted(meas_circuit.all_qubits())))
    sim = cirq.DensityMatrixSimulator(seed=int(rng.integers(2**31)))
    result = sim.run(noisy_circuit, repetitions=n_shots)
    bits = result.measurements["m"]  # shape (n_shots, 2)
    # Z eigenvalues: 0 → +1, 1 → −1
    z0 = 1 - 2 * bits[:, 0].astype(float)
    z1 = 1 - 2 * bits[:, 1].astype(float)
    return float(np.mean(z0 * z1))


def noisy_executor(circuit: cirq.Circuit, scale_factor: float, noise_model,
                   n_shots: int, rng: np.random.Generator) -> float:
    """
    Apply noise scaling by gate folding then measure.
    Uses Mitiq-style global folding: repeat each gate (sf-1)//2 times extra.
    For integer scale factors this is exact; matches main experiment executor.
    """
    import mitiq
    sf_int = int(round(scale_factor))
    if sf_int <= 1:
        scaled = circuit
    else:
        scaled = mitiq.zne.scaling.fold_gates_at_random(circuit, scale_factor=sf_int)
    return measure_Z0Z1(scaled, noise_model, n_shots, rng)

# ─────────────────────────────────────────────────────────────────────────────
# Schedules and extrapolants  (identical to main experiment)
# ─────────────────────────────────────────────────────────────────────────────

SCHEDULES = {
    "fibonacci": [1, 2, 3, 5, 8, 13],
    "linear":    [1, 2, 3, 4, 5, 6],
}

def linear_extrapolant(lambdas: np.ndarray, values: np.ndarray) -> float:
    """OLS linear fit, evaluate at lambda=0."""
    coeffs = np.polyfit(lambdas, values, 1)
    return float(np.polyval(coeffs, 0))

def richardson_extrapolant(lambdas: np.ndarray, values: np.ndarray) -> float:
    """Two-point Richardson: uses first two points only."""
    l1, l2 = lambdas[0], lambdas[1]
    v1, v2 = values[0], values[1]
    return float((l2 * v1 - l1 * v2) / (l2 - l1))

EXTRAPOLANTS = {
    "linear":     lambda l, v: linear_extrapolant(l, v),
    "richardson": lambda l, v: richardson_extrapolant(l, v),
}

# ─────────────────────────────────────────────────────────────────────────────
# Stability filter  (identical to paper Eq. 3)
# ─────────────────────────────────────────────────────────────────────────────

NU    = 0.5
GAMMA = 0.05

def compute_epsilon(lambdas: np.ndarray, values: np.ndarray) -> float:
    """Discrete H^1 energy seminorm (paper Eq. 3)."""
    eps = 0.0
    for k in range(len(lambdas) - 1):
        dl = lambdas[k + 1] - lambdas[k]
        dv = values[k + 1] - values[k]
        eps += NU * (dv / dl) ** 2 * dl
    return float(eps)

# ─────────────────────────────────────────────────────────────────────────────
# Main experiment loop
# ─────────────────────────────────────────────────────────────────────────────

def run_configuration(circuit_type: str, noise_name: str, n: int, depth: int,
                      p: float, schedule_name: str, extrapolant_name: str,
                      n_trials: int, n_shots: int, seed: int) -> Dict:
    """Run one (circuit, noise, n, d, p, schedule, extrapolant) configuration."""
    rng = np.random.default_rng(seed)
    lambdas = np.array(SCHEDULES[schedule_name], dtype=float)
    noise_model = NOISE_MODELS[noise_name](p)
    extrapolant_fn = EXTRAPOLANTS[extrapolant_name]

    zne_vals, eps_vals, flag_vals = [], [], []

    for _ in range(n_trials):
        # Build circuit (randomised circuits get fresh angles each trial)
        builder = CIRCUIT_BUILDERS[circuit_type]
        if circuit_type in ("vqe", "random", "qaoa"):
            circuit = builder(n, depth, rng)
        else:
            circuit = builder(n, depth)

        # Collect ZNE data points
        measured = np.array([
            noisy_executor(circuit, lam, noise_model, n_shots, rng)
            for lam in lambdas
        ])

        # Extrapolate
        zne_val = extrapolant_fn(lambdas, measured)
        eps_val = compute_epsilon(lambdas, measured)
        flag_val = 1 if eps_val >= GAMMA else 0

        zne_vals.append(zne_val)
        eps_vals.append(eps_val)
        flag_vals.append(flag_val)

    return {
        "circuit_type":    circuit_type,
        "noise_model":     noise_name,
        "n_qubits":        n,
        "depth":           depth,
        "noise_level":     p,
        "schedule":        schedule_name,
        "extrapolant":     extrapolant_name,
        "n_trials":        n_trials,
        "n_shots":         n_shots,
        "zne_mean":        float(np.mean(zne_vals)),
        "zne_std":         float(np.std(zne_vals)),       # ddof=0, matches main experiment
        "epsilon_mean":    float(np.mean(eps_vals)),
        "epsilon_std":     float(np.std(eps_vals)),
        "flag_rate":       float(np.mean(flag_vals)),
        "flagged":         int(np.sum(flag_vals)),
    }


def main():
    args = parse_args()
    N_TRIALS = args.trials
    N_SHOTS  = args.shots
    OUTFILE  = args.outfile

    # Experiment grid -- intentionally focused subset for robustness check
    circuit_types  = ["test", "vqe", "random", "qaoa"]
    noise_names    = ["depolarising", "amplitude_damping", "correlated_depolarising"]
    qubit_counts   = [2, 4]
    depths         = [4]
    noise_levels   = [0.001, 0.005, 0.01, 0.02, 0.05]
    schedule_names = ["fibonacci", "linear"]
    extrap_names   = ["linear", "richardson"]

    configs = list(product(
        circuit_types, noise_names, qubit_counts,
        depths, noise_levels, schedule_names, extrap_names
    ))
    total = len(configs)
    print(f"Noise robustness experiment: {total} configurations × {N_TRIALS} trials = "
          f"{total * N_TRIALS:,} total runs")
    print(f"Shots per run: {N_SHOTS}  |  Output: {OUTFILE}")
    print()

    results = {}
    t0 = time.time()

    for i, (ct, nm, n, d, p, sched, ext) in enumerate(configs):
        key = f"{ct}_{nm}_n{n}_d{d}_p{p}_{sched}_{ext}"
        seed = abs(hash((ct, nm, n, d, p, sched, ext))) % (2**31)

        res = run_configuration(ct, nm, n, d, p, sched, ext, N_TRIALS, N_SHOTS, seed)
        results[key] = res

        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (total - i - 1)
        print(f"  [{i+1:3d}/{total}] {key:<65}  "
              f"ε={res['epsilon_mean']:.5f}  flag={res['flag_rate']:.1f}  "
              f"ETA {eta/60:.1f}min", flush=True)

    # ─── Save results ────────────────────────────────────────────────────────
    with open(OUTFILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTFILE}")

    # ─── Summary table ───────────────────────────────────────────────────────
    print_summary(results)


def print_summary(results: Dict):
    """
    Print three summary checks against the paper's qualitative claims.
    """
    import textwrap

    print()
    print("=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)

    # ── CHECK 1: ε hierarchy per noise model ─────────────────────────────────
    print("\nCHECK 1 — ε hierarchy (VQE > QAOA > random > test) per noise model")
    print(f"  Reference condition: n=2, d=4, p=0.01, linear extrapolant")
    print()

    noise_names    = ["depolarising", "amplitude_damping", "correlated_depolarising"]
    circuit_types  = ["test", "vqe", "random", "qaoa"]
    schedule_names = ["fibonacci", "linear"]

    for nm in noise_names:
        means = {}
        for ct in circuit_types:
            eps_vals = []
            for sched in schedule_names:
                key = f"{ct}_{nm}_n2_d4_p0.01_{sched}_linear"
                if key in results:
                    eps_vals.append(results[key]["epsilon_mean"])
            if eps_vals:
                means[ct] = np.mean(eps_vals)
        if len(means) == 4:
            ordered = sorted(means.items(), key=lambda x: -x[1])
            order_str = " > ".join(f"{k}({v:.4f})" for k, v in ordered)
            # Check if VQE > QAOA > random > test
            vals = [means.get(ct, 0) for ct in ["vqe", "qaoa", "random", "test"]]
            correct = all(vals[i] > vals[i+1] for i in range(len(vals)-1))
            status = "✅ HOLDS" if correct else "⚠️  BROKEN"
            print(f"  {nm:<30}: {order_str}  [{status}]")

    # ── CHECK 2: Zero flag rates at low noise ─────────────────────────────────
    print()
    print("CHECK 2 — Flag rates at p≤0.01, d=4 (should be 0.0%)")
    print()

    for nm in noise_names:
        total_flags = 0
        total_trials = 0
        for key, res in results.items():
            if (res["noise_model"] == nm and
                res["noise_level"] <= 0.01 and
                res["depth"] == 4):
                total_flags  += res["flagged"]
                total_trials += res["n_trials"]
        if total_trials > 0:
            rate = total_flags / total_trials
            status = "✅ HOLDS" if total_flags == 0 else f"⚠️  {total_flags} flags"
            print(f"  {nm:<30}: {total_flags} flags / {total_trials} trials  "
                  f"(rate={rate:.4f})  [{status}]")

    # ── CHECK 3: Monotone qubit scaling ───────────────────────────────────────
    print()
    print("CHECK 3 — Monotone ε decrease n=2 → n=4 (should hold in most cases)")
    print()

    for nm in noise_names:
        monotone_count = 0
        total_count    = 0
        for ct in circuit_types:
            for sched in schedule_names:
                key2 = f"{ct}_{nm}_n2_d4_p0.01_{sched}_linear"
                key4 = f"{ct}_{nm}_n4_d4_p0.01_{sched}_linear"
                if key2 in results and key4 in results:
                    e2 = results[key2]["epsilon_mean"]
                    e4 = results[key4]["epsilon_mean"]
                    total_count += 1
                    if e2 > e4:
                        monotone_count += 1
        if total_count > 0:
            frac = monotone_count / total_count
            status = "✅ HOLDS" if frac >= 0.80 else "⚠️  WEAK"
            print(f"  {nm:<30}: {monotone_count}/{total_count} monotone  "
                  f"({frac*100:.0f}%)  [{status}]")

    print()
    print("=" * 70)
    print("Robustness experiment complete.")
    print("If all three checks HOLD under amplitude_damping and")
    print("correlated_depolarising, the paper's qualitative claims are robust")
    print("to noise model mis-specification.")
    print("=" * 70)


if __name__ == "__main__":
    main()
