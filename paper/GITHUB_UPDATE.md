# GitHub Repo Update Instructions

## New Repository Title
Fluid-Dynamic Stability Filtering for Quantum Error Mitigation

## New Description (160 chars max for GitHub About)
Physics-informed ZNE validation using Navier-Stokes energy norms and Fibonacci noise scaling. Flags hallucinated corrections in NISQ circuits.

## Topics (add these in GitHub Settings > About)
quantum-computing
quantum-error-mitigation
zero-noise-extrapolation
navier-stokes
physics-informed
nisq
cirq
mitiq
fibonacci
python

## Old vs New

| Field       | Old                                              | New                                                        |
|-------------|--------------------------------------------------|------------------------------------------------------------|
| Title       | Quantum-Vortex-QEM                               | Quantum-Vortex-QEM  (keep slug, update display name)       |
| Description | Implementation of Z13/Z32 singularity mapping... | Physics-informed ZNE validation using Navier-Stokes...     |
| Website     | (blank)                                          | Link to Zenodo DOI once minted                             |

## Commit Message for First Push
feat: initial implementation of stability filter, Fibonacci scaling, Hessian detector

- StabilityFilter: ε = ν∫||∇u||²dΩ < Γ applied to ZNE residuals
- fibonacci_scaling: Fibonacci vs linear vs odd schedule comparison  
- HessianDetector: det(H(Φ)) = 0 singularity scanning
- zne_pipeline: full cirq + mitiq integration
