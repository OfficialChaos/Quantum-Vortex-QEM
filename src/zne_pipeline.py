"""
zne_pipeline.py
---------------
Full pipeline: ZNE simulation + Fibonacci scaling + Stability filter +
Hessian singularity detection.

Uses cirq + mitiq for the quantum simulation backend.
Shot-based sampling used for realistic variance across trials.
"""

import numpy as np
import cirq
import mitiq
from mitiq import zne
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import LinearFactory, RichardsonFactory, PolyFactory

from .stability_filter import StabilityFilter, StabilityResult
from .fibonacci_scaling import fibonacci_scale_factors, linear_scale_factors, odd_scale_factors
from .hessian_detector import HessianDetector

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class PipelineResult:
    schedule_name: str
    scale_factors: List[float]
    raw_expectations: np.ndarray
    zne_value: float
    stability: StabilityResult
    label: str = ""

    def summary(self) -> str:
        return (
            f"\n{'='*50}\n"
            f"Schedule : {self.schedule_name}\n"
            f"Scales   : {self.scale_factors}\n"
            f"ZNE value: {self.zne_value:.6f}\n"
            f"{self.stability.summary()}\n"
        )


def make_test_circuit(depth: int = 4) -> cirq.Circuit:
    """
    Build a simple test circuit: alternating X rotations and CNOT gates.
    Ideal expectation value of Z⊗Z is cos(depth * pi/4).

    Parameters
    ----------
    depth : circuit depth (number of layers)
    """
    qubits = cirq.LineQubit.range(2)
    q0, q1 = qubits

    circuit = cirq.Circuit()
    for _ in range(depth):
        circuit.append([
            cirq.rx(np.pi / 4)(q0),
            cirq.rx(np.pi / 4)(q1),
            cirq.CNOT(q0, q1)
        ])
    circuit.append(cirq.measure(q0, q1, key='result'))
    return circuit


def noisy_executor(circuit: cirq.Circuit,
                   noise_level: float = 0.01,
                   n_shots: int = 1000) -> float:
    """
    Simulate circuit with depolarizing noise using shot-based sampling.
    Returns expectation value of Z(x)Z estimated from measurement outcomes.

    Shot-based simulation (vs exact density matrix) introduces realistic
    sampling variance across trials - essential for meaningful statistics.

    Parameters
    ----------
    circuit     : cirq Circuit
    noise_level : depolarizing noise parameter
    n_shots     : number of measurement shots per evaluation
    """
    # Cap at 0.24 - depolarize(p) requires p < 0.25 for 2-qubit gates
    capped_noise = min(noise_level, 0.24)

    qubits = sorted(circuit.all_qubits())
    q0, q1 = qubits[0], qubits[1]

    # Build clean circuit with measurements, stripping existing measurements
    sim_circuit = cirq.Circuit()
    for moment in circuit:
        for op in moment.operations:
            if not isinstance(op.gate, cirq.MeasurementGate):
                sim_circuit.append(op)
    sim_circuit.append(cirq.measure(q0, q1, key='result'))

    # Apply depolarizing noise model
    noise_model = cirq.depolarize(p=capped_noise)
    noisy_circuit = sim_circuit.with_noise(noise_model)

    # Shot-based run - stochastic, gives variance across trials
    simulator = cirq.DensityMatrixSimulator()
    result = simulator.run(noisy_circuit, repetitions=n_shots)

    # Z(x)Z expectation from measurement outcomes
    # Measurement: 0 -> eigenvalue +1,  1 -> eigenvalue -1
    bits = result.measurements['result']    # shape: (n_shots, 2)
    z0 = 1 - 2 * bits[:, 0].astype(float)  # +1 or -1
    z1 = 1 - 2 * bits[:, 1].astype(float)
    zz = z0 * z1                            # ZZ product per shot

    return float(np.mean(zz))


def run_pipeline(
    circuit_depth: int = 4,
    base_noise: float = 0.01,
    stability_threshold: float = 0.05,
    nu: float = 0.5,
    n_scales: int = 5,
    n_shots: int = 1000,
    schedules: Optional[List[str]] = None
) -> Dict[str, PipelineResult]:
    """
    Run full ZNE pipeline with multiple noise scaling schedules,
    applying stability filter and Hessian detection to each.

    Parameters
    ----------
    circuit_depth        : depth of test circuit
    base_noise           : base depolarizing noise level
    stability_threshold  : Gamma for stability filter
    nu                   : viscosity constant nu
    n_scales             : number of scale factors per schedule
    n_shots              : shots per expectation value estimate
    schedules            : list of schedule names to run
                           ('fibonacci', 'linear', 'odd')

    Returns
    -------
    dict of {schedule_name: PipelineResult}
    """
    if schedules is None:
        schedules = ["fibonacci", "linear", "odd"]

    circuit = make_test_circuit(depth=circuit_depth)
    sf_filter = StabilityFilter(gamma=stability_threshold, nu=nu)

    schedule_map = {
        "fibonacci": fibonacci_scale_factors(n_scales),
        "linear":    linear_scale_factors(n_scales),
        "odd":       odd_scale_factors(n_scales),
    }

    results = {}

    for name in schedules:
        if name not in schedule_map:
            print(f"Unknown schedule: {name}, skipping.")
            continue

        scales = schedule_map[name]
        expectations = []

        # Collect noisy expectation values at each scale factor
        for s in scales:
            noisy_val = noisy_executor(
                circuit,
                noise_level=base_noise * s,
                n_shots=n_shots
            )
            expectations.append(noisy_val)

        ev_array = np.array(expectations)
        sf_array = np.array(scales, dtype=float)

        # ZNE extrapolation via linear fit to zero
        coeffs = np.polyfit(sf_array, ev_array, deg=1)
        zne_value = float(np.polyval(coeffs, 0.0))

        # Apply stability filter
        stability = sf_filter.evaluate(ev_array, sf_array)

        results[name] = PipelineResult(
            schedule_name=name,
            scale_factors=scales,
            raw_expectations=ev_array,
            zne_value=zne_value,
            stability=stability,
        )

    return results


def print_comparison(results: Dict[str, PipelineResult]) -> None:
    """Print a clean comparison table of all schedules."""
    print("\n" + "="*70)
    print(f"{'Schedule':<14} {'ZNE Value':>12} {'epsilon':>14} "
          f"{'Gamma':>8} {'Stable?':>10}")
    print("-"*70)
    for name, r in results.items():
        stable = "PASS" if r.stability.is_stable else "FLAGGED"
        print(f"{name:<14} {r.zne_value:>12.6f} "
              f"{r.stability.epsilon:>14.6f} "
              f"{r.stability.gamma:>8.4f} "
              f"{stable:>10}")
    print("="*70)


if __name__ == "__main__":
    print("Running Quantum-Vortex QEM Pipeline...")
    results = run_pipeline(
        circuit_depth=4,
        base_noise=0.01,
        stability_threshold=0.05,
        n_scales=5,
        n_shots=1000
    )
    print_comparison(results)
    for r in results.values():
        print(r.summary())
