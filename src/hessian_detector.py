"""
hessian_detector.py
-------------------
Detects degenerate critical points (Hessian singularities) in the
ZNE expectation value landscape.

Criterion:
    det(H(Φ)) = 0

Where H(Φ) is the Hessian matrix of the expectation value surface
evaluated over noise scale factors and circuit parameters.

Physical interpretation:
    A zero determinant indicates a saddle point or flat direction —
    a region where ZNE extrapolation is geometrically degenerate
    and most likely to produce unreliable corrections.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class HessianResult:
    determinant: float
    eigenvalues: np.ndarray
    is_singular: bool           # det ≈ 0
    is_positive_definite: bool  # all eigenvalues > 0 (stable minimum)
    condition_number: float     # ratio of max/min eigenvalue

    def summary(self) -> str:
        status = "SINGULAR (degenerate)" if self.is_singular else "Non-singular"
        pd = "Positive definite (stable)" if self.is_positive_definite else "Not positive definite"
        return (
            f"Hessian Analysis\n"
            f"  det(H)           : {self.determinant:.6e}\n"
            f"  Eigenvalues      : {np.round(self.eigenvalues, 6)}\n"
            f"  Condition number : {self.condition_number:.4f}\n"
            f"  Singularity      : {status}\n"
            f"  Stability        : {pd}"
        )


class HessianDetector:
    """
    Computes and analyzes the Hessian of a ZNE expectation value
    surface to identify degenerate critical points.

    Parameters
    ----------
    singular_threshold : float
        |det(H)| below this value is considered singular. Default: 1e-10.
    """

    def __init__(self, singular_threshold: float = 1e-10):
        self.singular_threshold = singular_threshold

    def compute_hessian_1d(self, values: np.ndarray,
                           x: np.ndarray) -> np.ndarray:
        """
        Compute finite-difference Hessian for a 1D function.
        Returns a 1x1 matrix (second derivative).

        Parameters
        ----------
        values : function values f(x)
        x      : independent variable (scale factors)
        """
        d2 = np.gradient(np.gradient(values, x), x)
        # Return as 2D array for consistency
        return d2

    def compute_hessian_2d(self, surface: np.ndarray) -> np.ndarray:
        """
        Compute the 2x2 Hessian matrix at the center of a 2D surface.

        Parameters
        ----------
        surface : 2D array of expectation values over (scale_factor, parameter) grid

        Returns
        -------
        2x2 Hessian matrix
        """
        # Second derivatives via finite differences
        fxx = np.gradient(np.gradient(surface, axis=0), axis=0)
        fyy = np.gradient(np.gradient(surface, axis=1), axis=1)
        fxy = np.gradient(np.gradient(surface, axis=0), axis=1)

        # Evaluate at center point
        cx, cy = surface.shape[0] // 2, surface.shape[1] // 2
        H = np.array([
            [fxx[cx, cy], fxy[cx, cy]],
            [fxy[cx, cy], fyy[cx, cy]]
        ])
        return H

    def analyze(self, hessian_matrix: np.ndarray) -> HessianResult:
        """
        Analyze a Hessian matrix for singularity and stability.

        Parameters
        ----------
        hessian_matrix : square numpy array

        Returns
        -------
        HessianResult dataclass
        """
        det = np.linalg.det(hessian_matrix)
        eigenvalues = np.linalg.eigvalsh(hessian_matrix)

        is_singular = abs(det) < self.singular_threshold
        is_pd = bool(np.all(eigenvalues > 0))

        # Condition number: ratio of largest to smallest absolute eigenvalue
        abs_eigs = np.abs(eigenvalues)
        if abs_eigs.min() < 1e-15:
            cond = np.inf
        else:
            cond = abs_eigs.max() / abs_eigs.min()

        return HessianResult(
            determinant=float(det),
            eigenvalues=eigenvalues,
            is_singular=is_singular,
            is_positive_definite=is_pd,
            condition_number=float(cond)
        )

    def scan_landscape(self, surface: np.ndarray,
                       window: int = 5) -> np.ndarray:
        """
        Scan a 2D expectation value surface for singular regions.

        Parameters
        ----------
        surface : 2D array of expectation values
        window  : local window size for Hessian computation

        Returns
        -------
        singularity_map : 2D boolean array, True where det(H) ≈ 0
        """
        rows, cols = surface.shape
        singularity_map = np.zeros((rows, cols), dtype=bool)
        hw = window // 2

        for i in range(hw, rows - hw):
            for j in range(hw, cols - hw):
                patch = surface[i-hw:i+hw+1, j-hw:j+hw+1]
                H = self.compute_hessian_2d(patch)
                result = self.analyze(H)
                singularity_map[i, j] = result.is_singular

        return singularity_map
