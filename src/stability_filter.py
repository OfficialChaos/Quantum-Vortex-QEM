"""
stability_filter.py
-------------------
Implements the fluid-dynamic stability criterion as a validation filter
for Zero-Noise Extrapolation (ZNE) outputs.

Criterion:
    ε = ν ∫ ||∇u||² dΩ < Γ

Where:
    ν  = viscosity constant (set to noise level proxy)
    u  = ZNE extrapolation curve (discretized)
    Γ  = stability threshold (user-defined)
    ε  = energy dissipation norm of the residuals
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class StabilityResult:
    epsilon: float          # Computed energy norm
    gamma: float            # Threshold
    is_stable: bool         # ε < Γ
    gradient_norm: float    # ||∇u||² integrated
    flagged: bool           # True if hallucination suspected

    def summary(self) -> str:
        status = "STABLE" if self.is_stable else "FLAGGED (possible hallucination)"
        return (
            f"Stability Filter Result\n"
            f"  ε (energy norm)  : {self.epsilon:.6f}\n"
            f"  Γ (threshold)    : {self.gamma:.6f}\n"
            f"  Status           : {status}\n"
            f"  ||∇u||² integral : {self.gradient_norm:.6f}"
        )


class StabilityFilter:
    """
    Applies a Navier-Stokes-inspired energy dissipation norm to ZNE
    extrapolation curves to flag potentially hallucinated corrections.

    Parameters
    ----------
    gamma : float
        Stability threshold Γ. Corrections with ε >= Γ are flagged.
    nu : float
        Viscosity constant ν. Default: entropy proxy (0.5 bits normalized).
    """

    def __init__(self, gamma: float = 0.05, nu: float = 0.5):
        self.gamma = gamma
        self.nu = nu

    def compute_epsilon(self, expectation_values: np.ndarray,
                        scale_factors: np.ndarray) -> float:
        """
        Compute ε = ν ∫ ||∇u||² dΩ over the extrapolation curve.

        Parameters
        ----------
        expectation_values : array of ZNE expectation values at each scale factor
        scale_factors      : corresponding noise scale factors (the Ω discretization)

        Returns
        -------
        epsilon : float — the energy dissipation norm
        """
        # Compute gradient of the expectation value curve: ∇u
        gradients = np.gradient(expectation_values, scale_factors)

        # Compute ||∇u||² pointwise
        grad_sq = gradients ** 2

        # Integrate over Ω using trapezoidal rule
        integral = np.trapz(grad_sq, scale_factors)

        # Apply viscosity constant
        epsilon = self.nu * integral
        return epsilon

    def evaluate(self, expectation_values: np.ndarray,
                 scale_factors: np.ndarray,
                 gamma: Optional[float] = None) -> StabilityResult:
        """
        Evaluate stability of a ZNE result.

        Parameters
        ----------
        expectation_values : array of expectation values at each noise level
        scale_factors      : noise scale factors used in ZNE
        gamma              : override threshold (uses self.gamma if None)

        Returns
        -------
        StabilityResult dataclass
        """
        threshold = gamma if gamma is not None else self.gamma

        gradients = np.gradient(expectation_values, scale_factors)
        grad_sq = gradients ** 2
        integral = np.trapz(grad_sq, scale_factors)
        epsilon = self.nu * integral

        is_stable = epsilon < threshold

        return StabilityResult(
            epsilon=epsilon,
            gamma=threshold,
            is_stable=is_stable,
            gradient_norm=integral,
            flagged=not is_stable
        )

    def batch_evaluate(self, results_dict: dict) -> dict:
        """
        Evaluate stability across multiple circuits/schedules.

        Parameters
        ----------
        results_dict : {label: (expectation_values, scale_factors)}

        Returns
        -------
        {label: StabilityResult}
        """
        return {
            label: self.evaluate(ev, sf)
            for label, (ev, sf) in results_dict.items()
        }
