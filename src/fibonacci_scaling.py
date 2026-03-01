"""
fibonacci_scaling.py
--------------------
Implements Fibonacci-sequenced noise scaling for Zero-Noise Extrapolation.

Instead of linear scale factors [1, 2, 3, 4, 5], uses:
    τ_n = τ_{n-1} + τ_{n-2}  →  [1, 2, 3, 5, 8, 13, ...]

Hypothesis: Fibonacci spacing reduces interpolation artifacts in the
extrapolation fit due to the golden-ratio distribution of sample points.
"""

import numpy as np
from typing import List


def fibonacci_scale_factors(n: int = 6, start: int = 1) -> List[int]:
    """
    Generate n Fibonacci scale factors starting from (1, 2).

    Parameters
    ----------
    n     : number of scale factors to generate
    start : starting value (default 1, giving sequence 1,2,3,5,8,13...)

    Returns
    -------
    List of integers
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [1]

    seq = [1, 2]
    while len(seq) < n:
        seq.append(seq[-1] + seq[-2])
    return seq[:n]


def linear_scale_factors(n: int = 6, start: int = 1) -> List[int]:
    """
    Generate n linear scale factors [1, 2, 3, ..., n].
    Baseline comparison for Fibonacci schedule.
    """
    return list(range(start, start + n))


def odd_scale_factors(n: int = 6) -> List[int]:
    """
    Generate n odd scale factors [1, 3, 5, 7, ...].
    Standard Mitiq default for gate-level noise scaling.
    """
    return [2 * i + 1 for i in range(n)]


def compare_schedules(n: int = 6) -> dict:
    """
    Return all three schedules for comparison.

    Returns
    -------
    dict with keys: 'fibonacci', 'linear', 'odd'
    """
    return {
        "fibonacci": fibonacci_scale_factors(n),
        "linear": linear_scale_factors(n),
        "odd": odd_scale_factors(n),
    }


def golden_ratio_density(scales: List[float]) -> float:
    """
    Compute how closely the spacing of scale factors approximates
    the golden ratio φ = 1.618...

    A lower score means spacing is closer to φ-distributed.
    Used as a diagnostic for schedule quality.

    Parameters
    ----------
    scales : list of scale factors

    Returns
    -------
    mean absolute deviation from golden ratio spacing
    """
    phi = (1 + np.sqrt(5)) / 2
    if len(scales) < 3:
        return 0.0

    ratios = [scales[i + 1] / scales[i] for i in range(len(scales) - 1)]
    deviations = [abs(r - phi) for r in ratios]
    return float(np.mean(deviations))


if __name__ == "__main__":
    schedules = compare_schedules(6)
    print("Scale factor schedules (n=6):")
    for name, factors in schedules.items():
        density = golden_ratio_density(factors)
        print(f"  {name:12s}: {factors}  | φ-deviation: {density:.4f}")
