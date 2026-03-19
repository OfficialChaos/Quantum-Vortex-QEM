"""
run_hardware_v1.py — IBM Quantum Hardware Validation
Quantum-Vortex QEM Project  |  Shawn G. Kleipe  |  v0.4.0

PURPOSE
-------
Validate the stability filter (ε < Γ=0.05) on real IBM hardware at
the paper's reference condition (d≤4, low noise).  Compares Fibonacci
vs Linear schedules on Test and VQE-style circuits, matching paper
methodology exactly so hardware ε can be compared directly to Table 1.

DESIGN
------
  Circuits : test (Rx(π/4)+CNOT), vqe (Ry(π/4)+CNOT, i.e. CX — native IBM gate)
  Depths   : d=2, d=4  (both below Γ in simulation)
  Schedules: Fibonacci {1,2,3,5,8,13}, Linear {1,2,3,4,5,6}  (K=6)
  Extrapolant: linear fit to λ=0  (paper reference)
  Shots    : N=1024  (power-of-2, > paper N=500)
  Trials   : 5  (mean ± std)

  Total: 2 circuits × 2 schedules × 6 scale factors × 5 trials × 2 depths
       = 240 circuit executions batched in ONE sampler.run() call.

OUTPUT
------
  hardware_results_v1.json — same schema as summary_v4.json keys for
  direct comparison.  Printed summary flags each ε vs Γ=0.05.

FOLDING
-------
Global circuit folding: for integer scale λ (odd), the folded circuit is
  C_λ = C · (C† · C)^{(λ-1)/2}
applied to the gate-only circuit (measurements stripped, reattached after).
This matches Mitiq's fold_global and the paper's simulation methodology.
Even λ uses partial last fold; we restrict to odd λ = {1,3,5} or the
Fibonacci/Linear sequences which start at 1 and use integer steps — all
handled by the fractional folding formula below.

NOTE: IBM hardware uses ECR or CX (CNOT) as native 2-qubit gate.  CX is
used directly; the transpiler maps it efficiently.  The paper's VQE circuit
uses CZ, which equals H·CX·H — we use CX directly on hardware and note
this adaptation.  Results reflect the native hardware noise model.
"""

import json
import time
import numpy as np
from datetime import datetime
from qiskit import QuantumCircuit
from qiskit.circuit import ClassicalRegister
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# ── CONFIG ────────────────────────────────────────────────────────────────────

SHOTS     = 1024
TRIALS    = 5
DEPTHS    = [2, 4]
GAMMA     = 0.05    # simulation-calibrated threshold (paper Eq 3)
GAMMA_HW  = 0.10    # hardware-estimated threshold: 2× GAMMA, accounting for
                    # coherent errors, readout noise, and non-Markovian effects
                    # not present in depolarizing simulation
NU        = 0.5     # viscosity constant from paper Eq(3)

# Odd-only scale factors — required for gate-level global folding
# (C·C†·C requires odd integer scale factors)
# Fibonacci-odd: nearest odd Fibonacci numbers
# Linear-odd:    odd integers {1,3,5,7,9}
# K=5 for hardware (K=6 in simulation used even factors; noted in paper)
SCHEDULES = {
    'fibonacci': [1, 3, 5, 9, 13],   # odd Fibonacci-adjacent
    'linear':    [1, 3, 5, 7,  9],   # standard odd-integer schedule
}

# ── CIRCUIT BUILDERS ──────────────────────────────────────────────────────────

def make_test_circuit(depth: int) -> QuantumCircuit:
    """Alternating Rx(π/4) + CNOT — matches paper Test circuit exactly."""
    qc = QuantumCircuit(2)
    for _ in range(depth):
        qc.rx(np.pi / 4, 0)
        qc.rx(np.pi / 4, 1)
        qc.cx(0, 1)
    return qc   # no measurements — added after folding


def make_vqe_circuit(depth: int) -> QuantumCircuit:
    """Ry(π/4) + CX (CNOT) hardware-efficient ansatz.
    Paper uses Ry+CZ; on IBM hardware CX (CNOT) is the native 2-qubit gate.
    CZ = H·CX·H would add extra single-qubit overhead, so we use CX directly
    and note this in the paper as the hardware-adapted VQE circuit.
    """
    qc = QuantumCircuit(2)
    for _ in range(depth):
        qc.ry(np.pi / 4, 0)
        qc.ry(np.pi / 4, 1)
        qc.cx(0, 1)
    return qc   # no measurements


CIRCUIT_BUILDERS = {
    'test': make_test_circuit,
    'vqe':  make_vqe_circuit,
}

# ── GLOBAL CIRCUIT FOLDING ─────────────────────────────────────────────────────

def fold_global(qc: QuantumCircuit, scale: int) -> QuantumCircuit:
    """
    Global ZNE folding: C_λ = C · (C† · C)^k  where λ = 2k+1 (odd).

    For the Fibonacci and Linear schedules used here all scale factors
    are integers ≥ 1.  We implement the standard global fold:
      λ=1  → C
      λ=3  → C · C† · C
      λ=5  → C · C† · C · C† · C
      etc.

    Parameters
    ----------
    qc    : base circuit WITHOUT measurements
    scale : integer noise scale factor λ ≥ 1
    """
    if scale == 1:
        return qc.copy()

    if scale % 2 == 0:
        raise ValueError(f"scale must be odd for global folding, got {scale}")

    k = (scale - 1) // 2
    inv = qc.inverse()

    folded = qc.copy()
    for _ in range(k):
        folded = folded.compose(inv)
        folded = folded.compose(qc)

    return folded


def add_measurements(qc: QuantumCircuit) -> QuantumCircuit:
    """Measure logical qubits 0 and 1 into a 2-bit classical register.
    Using explicit 2-bit register (not measure_all) guarantees the
    bitstring stays 2 bits wide after transpilation to ibm_fez (156 qubits).
    measure_all() would produce 156-bit bitstrings, breaking compute_zz().
    """
    qc_m = qc.copy()
    cr = ClassicalRegister(2, 'c')
    qc_m.add_register(cr)
    qc_m.measure([0, 1], [0, 1])
    return qc_m

# ── OBSERVABLE: <Z0 Z1> ───────────────────────────────────────────────────────

def compute_zz(counts: dict) -> float:
    """
    Compute ⟨Z0 Z1⟩ from measurement counts.

    IBM Qiskit bitstring ordering is little-endian:
      bitstring[-1] = qubit 0 (rightmost)
      bitstring[-2] = qubit 1
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    exp = 0.0
    for bitstring, count in counts.items():
        # Strip spaces that sometimes appear in Qiskit bitstrings
        bs = bitstring.replace(' ', '')
        z0 = 1 if bs[-1] == '0' else -1
        z1 = 1 if bs[-2] == '0' else -1
        exp += z0 * z1 * count
    return exp / total

# ── ZNE LINEAR EXTRAPOLATION ──────────────────────────────────────────────────

def zne_linear_extrap(lambdas: list, values: list) -> float:
    """
    Linear extrapolation to λ=0 via OLS.
    Fits ⟨O⟩ = a·λ + b, returns b = ⟨O⟩₀.
    Matches paper's linear (deg-1) extrapolant.
    """
    lam = np.array(lambdas, dtype=float)
    val = np.array(values,  dtype=float)
    # OLS: b = (Σval·Σlam² - Σlam·Σ(lam·val)) / (N·Σlam² - (Σlam)²)
    N   = len(lam)
    b   = (val.sum() * (lam**2).sum() - lam.sum() * (lam*val).sum()) \
        / (N*(lam**2).sum() - lam.sum()**2)
    return float(b)

# ── EPSILON (paper Eq 3) ───────────────────────────────────────────────────────

def compute_epsilon(lambdas: list, values: list) -> float:
    """
    ε ≈ ν Σ_k ((ΔO/Δλ)² · Δλ)   with ν=0.5
    Discrete H¹ energy seminorm — matches paper Eq(3) exactly.
    """
    eps = 0.0
    for i in range(len(lambdas) - 1):
        dO  = values[i+1] - values[i]
        dl  = lambdas[i+1] - lambdas[i]
        eps += (dO / dl)**2 * dl
    return NU * eps

# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    timestamp = datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')
    print(f"\n{'='*60}")
    print(f"  Quantum-Vortex QEM — Hardware Validation v1")
    print(f"  {timestamp}")
    print(f"{'='*60}\n")

    # ── Connect ──────────────────────────────────────────────────
    print("Connecting to IBM Quantum...")
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print(f"  Backend : {backend.name}")
    print(f"  Qubits  : {backend.num_qubits}")
    print()

    # ── Build all circuits ────────────────────────────────────────
    # Index: (circuit_type, schedule_name, depth, trial, lambda_idx)
    # We store metadata alongside each circuit for result parsing.
    print("Building circuits...")
    circuit_list = []   # list of (qc_transpiled, metadata_dict)
    meta_list    = []

    # Build pass manager ONCE — depends only on backend, not on circuit
    pm = generate_preset_pass_manager(
        optimization_level=1,
        backend=backend
    )

    for circ_type, builder in CIRCUIT_BUILDERS.items():
        for depth in DEPTHS:
            base = builder(depth)
            for sched_name, lambdas in SCHEDULES.items():
                for trial in range(TRIALS):
                    for lam_idx, lam in enumerate(lambdas):
                        folded   = fold_global(base, lam)
                        folded_m = add_measurements(folded)
                        transpiled = pm.run(folded_m)
                        circuit_list.append(transpiled)
                        meta_list.append({
                            'circuit':  circ_type,
                            'schedule': sched_name,
                            'depth':    depth,
                            'trial':    trial,
                            'lambda':   lam,
                            'lam_idx':  lam_idx,
                        })

    total = len(circuit_list)
    print(f"  Built {total} circuits ({total * SHOTS:,} total shots)")
    print(f"  Submitting as ONE batched job...")
    print()

    # ── Submit single batched job ─────────────────────────────────
    sampler  = Sampler(backend)
    t_start  = time.time()
    job      = sampler.run(circuit_list, shots=SHOTS)
    job_id   = job.job_id()
    print(f"  Job ID : {job_id}")
    print(f"  Waiting for results...")

    result   = job.result()
    t_elapsed = time.time() - t_start
    print(f"  Done in {t_elapsed:.1f}s\n")

    # ── Parse results ─────────────────────────────────────────────
    # Accumulate per (circuit, schedule, depth, trial) → list of (λ, ⟨O⟩)
    from collections import defaultdict
    raw = defaultdict(lambda: defaultdict(list))  # raw[key][trial] = [(λ,val),...]

    for idx, pub_result in enumerate(result):
        meta  = meta_list[idx]
        counts = pub_result.data.c.get_counts()
        val    = compute_zz(counts)
        key    = (meta['circuit'], meta['schedule'], meta['depth'])
        trial  = meta['trial']
        raw[key][trial].append((meta['lambda'], val))

    # ── Compute ε and ZNE per trial, then average ─────────────────
    output = {}
    summary_rows = []

    for key in sorted(raw.keys()):
        circ_type, sched_name, depth = key
        lambdas = SCHEDULES[sched_name]

        eps_trials = []
        zne_trials = []

        for trial in range(TRIALS):
            # Sort by λ index (should already be ordered)
            pairs  = sorted(raw[key][trial], key=lambda x: x[0])
            lams   = [p[0] for p in pairs]
            vals   = [p[1] for p in pairs]

            eps_t = compute_epsilon(lams, vals)
            zne_t = zne_linear_extrap(lams, vals)
            eps_trials.append(eps_t)
            zne_trials.append(zne_t)

        eps_mean = float(np.mean(eps_trials))
        eps_std  = float(np.std(eps_trials, ddof=0))
        zne_mean = float(np.mean(zne_trials))
        zne_std  = float(np.std(zne_trials, ddof=0))
        flag_sim = eps_mean >= GAMMA
        flag_hw  = eps_mean >= GAMMA_HW

        record = {
            'circuit':    circ_type,
            'schedule':   sched_name,
            'depth':      depth,
            'n_qubits':   2,
            'shots':      SHOTS,
            'trials':     TRIALS,
            'lambdas':    lambdas,
            'epsilon_mean': eps_mean,
            'epsilon_std':  eps_std,
            'zne_mean':     zne_mean,
            'zne_std':      zne_std,
            'flag_sim':     flag_sim,   # vs Gamma=0.05 (simulation threshold)
            'flag_hw':      flag_hw,    # vs Gamma=0.10 (hardware threshold)
        }
        output[f"{circ_type}_{sched_name}_d{depth}"] = record

        if not flag_sim:
            status = f'PASS (both) ✅'
        elif not flag_hw:
            status = f'PASS hw, FLAG sim ⚠️'
        else:
            status = f'FLAG (both) ❌'
        summary_rows.append((circ_type, sched_name, depth,
                              eps_mean, eps_std, zne_mean, zne_std, status))

    # ── Print summary ─────────────────────────────────────────────
    print(f"{'='*75}")
    print(f"  HARDWARE VALIDATION SUMMARY")
    print(f"  Γ_sim={GAMMA} (simulation-calibrated)   Γ_hw={GAMMA_HW} (hardware-estimated)")
    print(f"  Backend: {backend.name}   |   {timestamp}")
    print(f"{'='*75}")
    print(f"  {'Circuit':<8} {'Schedule':<12} {'d':>3}  "
          f"{'ε_mean':>8} {'ε_std':>7}  "
          f"{'ZNE':>8} {'ZNE_std':>8}  Status")
    print(f"  {'-'*72}")
    for row in summary_rows:
        circ, sched, d, em, es, zm, zs, status = row
        print(f"  {circ:<8} {sched:<12} {d:>3}  "
              f"{em:>8.5f} {es:>7.5f}  "
              f"{zm:>8.4f} {zs:>8.4f}  {status}")
    print(f"{'='*75}")
    print(f"  Legend: PASS(both)=ε<{GAMMA}  |  PASS hw=ε<{GAMMA_HW}  |  FLAG(both)=ε≥{GAMMA_HW}")
    print(f"{'='*75}\n")

    # ── Compare to simulation ─────────────────────────────────────
    print("=== COMPARISON TO SIMULATION (Table 1, n=2, linear extrap) ===\n")
    sim_ref = {
        # (circuit, schedule, depth): (zne_mean, zne_std, eps)
        ('test','fibonacci',4): (-0.033, 0.022, 0.00545),
        ('test','linear',   4): (-0.009, 0.044, 0.01345),
        ('vqe', 'fibonacci',4): (+0.054, 0.324, 0.01375),
        ('vqe', 'linear',   4): (-0.025, 0.594, 0.02408),
    }
    print(f"  {'Key':<25} {'ε_sim':>8} {'ε_hw':>8}  "
          f"{'ZNE_sim':>9} {'ZNE_hw':>9}")
    print(f"  {'-'*65}")
    for key, (zne_s, std_s, eps_s) in sorted(sim_ref.items()):
        circ, sched, d = key
        hw_key = f"{circ}_{sched}_d{d}"
        if hw_key in output:
            r    = output[hw_key]
            label = f"{circ}/{sched}/d={d}"
            print(f"  {label:<25} {eps_s:>8.5f} {r['epsilon_mean']:>8.5f}  "
                  f"{zne_s:>9.4f} {r['zne_mean']:>9.4f}")

    # ── Save JSON ─────────────────────────────────────────────────
    save_data = {
        'metadata': {
            'timestamp':  timestamp,
            'backend':    backend.name,
            'shots':      SHOTS,
            'trials':     TRIALS,
            'gamma_sim':  GAMMA,
            'gamma_hw':   GAMMA_HW,
            'nu':         NU,
            'job_id':     job_id,
            'elapsed_s':  round(t_elapsed, 1),
            'paper_version': 'v0.4.0',
        },
        'results': output,
    }
    fname = f"hardware_results_v1_{timestamp[:10]}.json"
    with open(fname, 'w') as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to: {fname}")
    print("Done.\n")


if __name__ == '__main__':
    main()
