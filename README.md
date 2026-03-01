# Fluid-Dynamic Stability Filtering for Quantum Error Mitigation

**A physics-informed validation framework for Zero-Noise Extrapolation in NISQ circuits**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/1169766876.svg)](https://doi.org/10.5281/zenodo.18827720)
[![arXiv](https://img.shields.io/badge/arXiv-pending-red.svg)]()

---

## Overview

Standard Zero-Noise Extrapolation (ZNE) can produce divergent or "hallucinated" results
when applied to short-depth circuits in high-noise NISQ environments. This repository
implements a **fluid-dynamic stability criterion** as a post-hoc validation filter for
ZNE outputs, alongside a **Fibonacci-sequenced noise scaling schedule** as an alternative
to conventional linear scaling.

### Core Hypothesis

> A Navier-Stokes energy dissipation norm applied to the ZNE extrapolation residuals
> can distinguish stable (physically meaningful) corrections from unstable
> (hallucinated) ones — analogous to detecting blow-up singularities in fluid flow.

---

## The Three Contributions

### 1. Stability Criterion Filter
```
epsilon = nu * integral ||grad u||^2 dOmega < Gamma
```
Applied to ZNE extrapolation curves: if the energy norm of the residual exceeds
threshold Gamma, the correction is flagged as potentially hallucinated.

### 2. Fibonacci Noise Scaling Schedule
Instead of linear scale factors `[1, 2, 3, 4, 5]`, we use:
```
tau_n = tau_{n-1} + tau_{n-2}  ->  [1, 2, 3, 5, 8, 13]
```
Hypothesis: Fibonacci spacing reduces interpolation artifacts in the extrapolation fit.

### 3. Hessian Singularity Detection
```
det(H(Phi)) = 0
```
Applied to the expectation value landscape to identify degenerate critical points
where ZNE is most likely to fail.

---

## Installation

```bash
git clone https://github.com/OfficialChaos/Quantum-Vortex-QEM.git
cd Quantum-Vortex-QEM
pip install -r requirements.txt
```

### Requirements
```
cirq>=1.3
mitiq>=0.38
numpy>=1.26
scipy>=1.12
matplotlib>=3.8
jupyter>=1.0
```

---

## Repository Structure

```
Quantum-Vortex-QEM/
|
+-- README.md
+-- LICENSE
+-- requirements.txt
+-- run_experiments.py        # Reproducible experiment runner
+-- quick_test.py             # Verify install without quantum deps
|
+-- src/
|   +-- __init__.py
|   +-- stability_filter.py       # epsilon < Gamma criterion
|   +-- fibonacci_scaling.py      # Fibonacci noise schedule
|   +-- hessian_detector.py       # Singularity detection
|   +-- zne_pipeline.py           # Full ZNE + filter pipeline
|
+-- experiments/
|   +-- results/
|       +-- summary.json          # Full experimental results
|
+-- paper/
|   +-- main.tex                  # Overleaf-ready LaTeX (RevTeX4-2)
|   +-- references.bib
|   +-- figures/                  # Generated plots (300 dpi)
|       +-- fig1_stability_and_accuracy.png
|       +-- fig2_flag_rate_heatmap.png
|
+-- notebooks/                    # Coming soon
```

---

## Quickstart

```python
import sys
sys.path.insert(0, '.')

from src.zne_pipeline import run_pipeline, print_comparison
from src.fibonacci_scaling import fibonacci_scale_factors

# Generate Fibonacci scale factors
scales = fibonacci_scale_factors(n=6)  # [1, 2, 3, 5, 8, 13]
print(scales)

# Run ZNE with stability filtering across all three schedules
results = run_pipeline(
    circuit_depth=4,
    base_noise=0.01,
    stability_threshold=0.05,  # Gamma
    n_shots=1000
)

print_comparison(results)
```

---

## Reproducing Experiments

```bash
python run_experiments.py
```

Outputs saved to `experiments/results/summary.json` and `paper/figures/`.

---

## Motivation & Background

Zero-Noise Extrapolation is one of the most widely used QEM techniques, implemented
in Google's Mitiq library. However, the choice of noise scaling schedule is typically
arbitrary (linear), and there is currently no standard method for validating whether
a given ZNE result is physically reliable.

This work draws on:
- **Mitiq** (Larose et al., 2022) — ZNE implementation framework
- **Fibonacci pulse sequences** (Viola & Lloyd, 1998; Ezzell et al., 2023) — dynamical decoupling
- **Navier-Stokes stability theory** — energy dissipation as a convergence criterion
- **DeepMind singularity work** (2024) — AI-assisted blow-up detection

---

## Status

- [x] Repository scaffolded
- [x] Core modules implemented
- [x] Baseline experiments complete
- [x] Fibonacci vs linear vs odd comparison
- [x] Figures generated (fig1, fig2)
- [ ] Paper draft finalized on Overleaf
- [ ] Zenodo DOI minted
- [ ] arXiv submission (quant-ph)

---

## Citation

```bibtex
@software{kleipe2026qvmqem,
  author = {Kleipe, Shawn G.},
  title  = {Fluid-Dynamic Stability Filtering for Quantum Error Mitigation},
  year   = {2026},
  doi    = {10.5281/zenodo.18827721},
  url    = {https://github.com/OfficialChaos/Quantum-Vortex-QEM},
  orcid  = {0009-0002-2480-2430}
}
```

---

## Author

**Shawn G. Kleipe**
ORCID: [0009-0002-2480-2430](https://orcid.org/0009-0002-2480-2430)
GitHub: [OfficialChaos](https://github.com/OfficialChaos)

---

## License

MIT License — see [LICENSE](LICENSE) for details.
