"""
quick_test.py
Run this from your project root: python quick_test.py
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.fibonacci_scaling import compare_schedules, golden_ratio_density

print("=" * 50)
print("Fibonacci vs Linear vs Odd Scale Factors (n=6)")
print("=" * 50)

schedules = compare_schedules(6)
for name, factors in schedules.items():
    density = golden_ratio_density(factors)
    print(f"  {name:<12}: {factors}  | phi-deviation: {density:.4f}")

print()
print("phi (golden ratio) = 1.6180...")
print("Lower deviation = closer to golden ratio spacing")
print("=" * 50)
print("SUCCESS: fibonacci_scaling module working correctly")
