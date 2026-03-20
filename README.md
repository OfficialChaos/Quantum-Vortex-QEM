# Fluid-Dynamic Stability Filtering for Zero-Noise Extrapolation in NISQ-Era Quantum Circuits

**A physics-informed post-hoc reliability framework for Zero-Noise Extrapolation**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.18827720-blue.svg)](https://doi.org/10.5281/zenodo.18827720)
[![arXiv](https://img.shields.io/badge/arXiv-pending-red.svg)]()
[![ORCID](https://img.shields.io/badge/ORCID-0009--0002--2480--2430-green.svg)](https://orcid.org/0009-0002-2480-2430)

---

## Overview

Zero-Noise Extrapolation (ZNE) is among the most widely deployed quantum error mitigation
techniques for NISQ devices. However, two fundamental questions remain open: (1) which
noise scaling schedule is optimal, and (2) how to validate whether a given ZNE output
is physically reliable or has diverged into an artifact.

This repository introduces:

1. **A fluid-dynamic stability filter** — a post-hoc reliability criterion based on the
   Navier-Stokes energy dissipation norm. Formally, ε is a discrete H¹ Sobolev
   regularity functional on the ZNE curve; high ε indicates ill-conditioned extrapolation.
2. **Five structured noise scaling schedules** — Fibonacci, Lucas, prime-anchored,
   linear, and odd-integer — systematically compared across all four standard
   extrapolants (linear, Richardson, poly-2, poly-3).

Both contributions are validated across **153,600 simulation runs** and **120 real
hardware trials** on IBM Quantum processors ibm\_fez (Heron r2) and ibm\_torino (Heron r1).

---

## Stack

[![Cirq](https://img.shields.io/badge/Cirq-1.3+-lightgrey)](https://quantumai.google/cirq)
[![Mitiq](https://img.shields.io/badge/Mitiq-0.38+-lightgrey)](https://mitiq.readthedocs.io/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4)](https://www.ibm.com/quantum/qiskit)
[![IBM Quantum](https://img.shields.io/badge/IBM%20Quantum-0268C2)](https://quantum.cloud.ibm.com)
[![IBM Torino](https://img.shields.io/badge/IBM_Torino-133_Qubits_Heron_r1-161616?logo=ibm&logoColor=white)](https://quantum.cloud.ibm.com)
[![IBM Fez](https://img.shields.io/badge/IBM_Fez-156_Qubits_Heron_r2-161616?logo=ibm&logoColor=white)](https://quantum.cloud.ibm.com)

---

## Key Results

| Finding | Result |
|---|---|
| Simulation: zero flags at p≤0.01, d≤4 | 0 / 14,400 trials (95% C-P bound: p_flag < 0.00022) |
| Hardware: zero flags, all conditions | 0 / 120 trials (24 conditions × 5 trials, 95% C-P bound: p_flag < 0.025) |
| Max hardware ε observed | 0.00285 — 5.7% of Γ=0.05 |
| Fibonacci schedule advantage | Lowest mean ε across all circuit types |
| Best extrapolant (lowest variance) | Linear extrapolation (wins 13/20 circuit/schedule combinations) |
| Lucas schedule | Lowest ZNE variance on random and QAOA circuits |
| Qubit scaling | ε decreases monotonically with qubit count in 19/20 cases |
| ε hierarchy (simulation, p=0.01) | VQE > QAOA > random > test |

---

## Installation

```bash
git clone https://github.com/OfficialChaos/Quantum-Vortex-QEM.git
cd Quantum-Vortex-QEM
pip install -r requirements.txt
```

### Requirements

```
cirq>=1.3.0
mitiq>=0.38.0
numpy>=1.26.0
scipy>=1.12.0
matplotlib>=3.8.0
jupyter>=1.0.0
notebook>=7.0.0
qiskit>=1.0.0               # required for hardware experiments only
qiskit-ibm-runtime>=0.20.0  # required for hardware experiments only
```

---

## Repository Structure

```
Quantum-Vortex-QEM/
│
├── README.md
├── LICENSE
├── requirements.txt
├── project_config.yaml
│
├── run_experiments_v4.py          # Full simulation (153,600 runs)
├── run_experiments_v3.py          # v3 predecessor (provenance)
├── run_hardware_v1.py             # Hardware: test + VQE circuits
├── run_hardware_v2.py             # Hardware: random + QAOA circuits
├── run_hardware_v3.py             # Hardware: d=6, d=8 depth sweep
├── noise_robustness_v4.py         # Noise model robustness analysis
│
├── src/
│   ├── __init__.py
│   ├── stability_filter.py        # ε < Γ criterion (paper Eq. 3)
│   ├── fibonacci_scaling.py       # All 5 schedule generators
│   ├── hessian_detector.py        # Hessian singularity detection
│   └── zne_pipeline.py            # Full ZNE + filter pipeline
│
├── experiments/
│   └── results/
│       └── summary_v4.json        # Full simulation results (153,600 runs)
│
├── hardware_results_v1_2026-03-19.json   # ibm_fez: test + VQE, d=2,4
├── hardware_results_v2_2026-03-19.json   # ibm_fez: random + QAOA, d=2,4
├── hardware_results_v3_2026-03-19.json   # ibm_torino: test + VQE, d=6,8
│
└── paper/
    ├── kleipe2026_v4.tex        # Paper source (LaTeX)
    ├── kleipe2026_v4.pdf        # Compiled paper
    ├── figB_v4_eps_vs_noise_qubit_scaling.png
    ├── figC_v4_flag_rate_heatmaps.png
    ├── figD_v4_zne_std_extrapolant_comparison.png
    ├── figE_v4_eps_vs_depth.png
    └── fig9_v4_schedule_comparison.png
```

---

## Reproducing the Simulation

```bash
# Full v4 experiment suite (153,600 runs — takes several hours)
python run_experiments_v4.py

# Results saved to experiments/results/summary_v4.json
```

### Experiment scope

```
4 circuit types   × test, VQE, random, QAOA
5 schedules       × Fibonacci, linear, odd, Lucas, prime-anchored
4 extrapolants    × linear, Richardson, poly-2, poly-3
3 qubit counts    × n = 2, 4, 6
8 circuit depths  × d = 2, 4, 6, 8, 10, 12, 16, 20
8 noise levels    × p = 0.001 to 0.2
10 trials
─────────────────────────────────────────────────────
Total             153,600 runs
```

---

## Reproducing the Hardware Experiments

IBM Quantum access required. Save your API credentials once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(
    channel='ibm_quantum_platform',
    token='YOUR_API_TOKEN',
    instance='YOUR_CRN',
    set_as_default=True
)
```

Then run the three hardware experiments:

```bash
# Experiment 1: test + VQE circuits, d=2,4 (ibm_fez, ~81s)
python run_hardware_v1.py

# Experiment 2: random + QAOA circuits, d=2,4 (ibm_fez, ~102s)
python run_hardware_v2.py

# Experiment 3: test + VQE, d=6,8 — depth sweep (ibm_torino, ~67s)
python run_hardware_v3.py
```

All three use a single batched job submission (200 circuits each) and save results
to `hardware_results_vN_YYYY-MM-DD.json`.

---

## The Stability Filter

The filter computes a discrete H¹ Sobolev regularity functional on the ZNE curve:

```
ε ≈ ν Σ_k ((ΔO/Δλ)² · Δλ)    with ν = 0.5
```

A ZNE output is flagged as potentially unreliable when `ε ≥ Γ` (default `Γ = 0.05`).

This is a **formal analogy** to the Navier-Stokes energy dissipation norm — not a claim
that ZNE dynamics are governed by fluid equations. The connection is a standard
condition-number argument: when `|∂⟨O⟩/∂λ|` is large over the fitting interval,
small measurement errors are amplified by the extrapolation, producing large uncertainty
in the zero-noise estimate `⟨O⟩₀`. High ε directly measures this steepness.

```python
from src.stability_filter import compute_epsilon

lambdas = [1, 2, 3, 5, 8, 13]   # Fibonacci schedule
values  = [0.82, 0.71, 0.63, 0.51, 0.38, 0.21]   # measured ⟨ZZ⟩

eps = compute_epsilon(lambdas, values)
print(f"ε = {eps:.5f}  {'PASS' if eps < 0.05 else 'FLAG'}")
```

---

## The Five Noise Scaling Schedules

| Schedule | K=6 scale factors | Mean ε (sim, p=0.01) |
|---|---|---|
| Fibonacci | {1, 2, 3, 5, 8, 13} | 0.010 ← lowest |
| Lucas | {1, 2, 3, 4, 7, 11} | 0.014 |
| Prime-anchored | {1, 2, 3, 5, 7, 11} | 0.014 |
| Odd-integer | {1, 3, 5, 7, 9, 11} | 0.013 |
| Linear | {1, 2, 3, 4, 5, 6} | 0.019 ← highest |

**Recommendation:** Fibonacci minimises ε (best filter stability). Lucas minimises ZNE
output variance on unstructured circuits (random, QAOA). Linear extrapolation produces
the lowest and most consistent ZNE variance across all schedules and is the recommended
extrapolant for reliability-focused deployment.

---

## Paper

**Fluid-Dynamic Stability Filtering for Zero-Noise Extrapolation in NISQ-Era Quantum Circuits**
Shawn G. Kleipe — v0.4.0

- **Preprint:** arXiv (pending, quant-ph)
- **Zenodo:** [10.5281/zenodo.18827720](https://doi.org/10.5281/zenodo.18827720)
- **ORCID:** [0009-0002-2480-2430](https://orcid.org/0009-0002-2480-2430)

```bibtex
@article{kleipe2026fluiddynamic,
  author  = {Kleipe, Shawn G.},
  title   = {Fluid-Dynamic Stability Filtering for Zero-Noise Extrapolation
             in {NISQ}-Era Quantum Circuits},
  year    = {2026},
  note    = {arXiv preprint (pending)},
  doi     = {10.5281/zenodo.18827720},
  url     = {https://github.com/OfficialChaos/Quantum-Vortex-QEM},
  orcid   = {0009-0002-2480-2430}
}
```

---

## Status

- [x] Core modules implemented (stability filter, schedules, Hessian detector)
- [x] v4 simulation complete — 153,600 runs across full parameter space
- [x] Hardware validation — ibm\_fez (Heron r2) + ibm\_torino (Heron r1)
- [x] All 24 hardware conditions pass Γ\_sim=0.05; max ε=0.00285
- [x] Zenodo DOI minted — 10.5281/zenodo.18827720
- [x] Paper finalised — v0.4.0
- [ ] arXiv submission (quant-ph) — in progress

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Shawn G. Kleipe**
ORCID: [0009-0002-2480-2430](https://orcid.org/0009-0002-2480-2430)
GitHub: [OfficialChaos](https://github.com/OfficialChaos)
