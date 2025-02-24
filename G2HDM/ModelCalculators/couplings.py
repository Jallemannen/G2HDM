# Functions for calculating couplings
# - Optimized for speed using Numba and SymEngine

# import packages
import numpy as np
import symengine as se
import numba as nb


def prepare_coefficients3(V, fields1, fields2):
    """Precompute coefficient dictionary before JIT compilation"""
    fields1 = [se.sympify(f) for f in fields1]
    fields2 = [se.sympify(f) for f in fields2]
    
    coeff_dict = V.as_coefficients_dict()
    field_map = {f: i for i, f in enumerate(fields1 + fields2)}

    precomputed = []
    for term, coeff in coeff_dict.items():
        factors = term.as_ordered_factors()
        indices = [field_map[f] for f in factors if f in field_map]
        precomputed.append((tuple(indices), float(coeff)))  # Store as tuples for Numba

    return precomputed, len(fields1), len(fields2)

@nb.njit
def calculate_couplings3(n1, n2, precomputed_coeffs):
    """JIT-compiled numerical calculation of L3 (without parallel execution)."""
    L3 = np.zeros((n1, n1, n2), dtype=np.float64)

    for idx in range(len(precomputed_coeffs)):  # No parallelism
        indices, coeff = precomputed_coeffs[idx]
        if len(indices) == 3:
            i, j, k = indices
            L3[i, j, k] = coeff
            if i != j:
                L3[j, i, k] = coeff  # Use symmetry optimization

    return L3


def prepare_coefficients4(V, fields1, fields2):
    """Precompute coefficient dictionary before JIT compilation"""
    fields1 = [se.sympify(f) for f in fields1]
    fields2 = [se.sympify(f) for f in fields2]
    
    coeff_dict = V.as_coefficients_dict()
    field_map = {f: i for i, f in enumerate(fields1 + fields2)}

    precomputed = []
    for term, coeff in coeff_dict.items():
        factors = term.as_ordered_factors()
        indices = [field_map[f] for f in factors if f in field_map]
        precomputed.append((tuple(indices), float(coeff)))  # Store as tuples for Numba

    return precomputed, len(fields1), len(fields2)

@nb.njit
def calculate_couplings4(n1, n2, precomputed_coeffs):
    """JIT-compiled numerical calculation of L4 (without parallel execution)."""
    L4 = np.zeros((n1, n1, n2, n2), dtype=np.float64)

    for idx in range(len(precomputed_coeffs)):  # No parallelism
        indices, coeff = precomputed_coeffs[idx]
        if len(indices) == 4:
            i, j, k, l = indices
            L4[i, j, k, l] = coeff

            # Fill in symmetric entries
            if i != j:
                L4[j, i, k, l] = coeff
            if k != l:
                L4[i, j, l, k] = coeff
            if i != j and k != l:
                L4[j, i, l, k] = coeff

    return L4