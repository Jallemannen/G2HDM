# Functions for calculating couplings
# - Optimized for speed using Numba and SymEngine

# import packages
import numpy as np
import symengine as se
import numba as nb


"""
def get_potentials_simplified(V0, )
    V0_se = se.sympify(MDC.V0_simplified).xreplace(subs_bgfield_values) 
    V0_se = V0_se.subs(subs_massStates_higgs).expand()
    self.V0 = se.sympify(sp.sympify(V0_se).xreplace({sp.I:0})) #V0_se #sp.sympify(V0_se)
    #self.V0 = V0_se
    #print("Time for V0: ", time.time()-t0)
    #display(self.V0)
    #display(V0_se.as_coefficients_dict())
    
    L_kin_se = se.sympify(MDC.L_kin_simplified).xreplace(subs_bgfield_values | subs_massStates_higgs | subs_massStates_gauge)
    L_kin_se = L_kin_se.expand()#.evalf()
    self.L_kin = se.sympify(sp.sympify(L_kin_se).xreplace({sp.I:0}))

"""


def calculate_couplings3(V, fields1, fields2):
    """Make sure that there are no imaginary terms in the potential. 
    Otherwise will symengine will raise "not implemented" error."""
    n1 = len(fields1)
    n2 = len(fields2)
    fields1 = [se.sympify(f) for f in fields1]
    fields2 = [se.sympify(f) for f in fields2]
    #L3 = np.empty((n1, n1, n2), dtype=object)
    L3 = np.zeros((n1, n1, n2), dtype=np.float64)
    fields_to_zero = {f: 0 for f in (fields1 + fields2)}
    # A cache to avoid recomputing the same fieldterm.
    cache = {}
    # Create a Poly object over all the fields.
    ##polyV = V.as_poly() #sp.Poly(V, list(set(fields1 + fields2)))
    coeff_dict = V.as_coefficients_dict()
    
    
    # Loop only over i <= j assuming symmetry in the first two indices.
    for i in range(n1):
        for j in range(i, n1):
            prod1 = fields1[i] * fields1[j]  # common product
            for k in range(n2):
                fieldterm = prod1 * fields2[k]
                if fieldterm not in cache:
                    # Expand and substitute only once per unique fieldterm.
                    ##term = V.coeff(fieldterm).subs(fields_to_zero) #Costly part
                    ##coeff = polyV.coeff_monomial(fieldterm)
                    ##term = coeff.subs(fields_to_zero)
                    term = coeff_dict.get(fieldterm, 0)
                    cache[fieldterm] = np.float64(term) #sp.re(term)
                value = cache[fieldterm]
                L3[i, j, k] = value
                if i != j:
                    # Use symmetry to assign the swapped indices.
                    L3[j, i, k] = value
    return L3

# calculate quartic couplings (Optimized)
#@jit
def calculate_couplings4(V, fields1, fields2):
    """Make sure that there are no imaginary terms in the potential. 
    Otherwise will symengine will raise "not implemented" error."""
    n1 = len(fields1)
    n2 = len(fields2)
    fields1 = [se.sympify(f) for f in fields1]
    fields2 = [se.sympify(f) for f in fields2]
    #L4 = np.empty((n1, n1, n2, n2), dtype=object)
    L4 = np.zeros((n1, n1, n2, n2), dtype=np.float64)
    fields_to_zero = {f: 0 for f in (fields1 + fields2)}
    # A cache to avoid recomputing the same fieldterm.
    cache = {}
    # Create a Poly object over all the fields.
    ##polyV = V.as_poly() #sp.Poly(V, list(set(fields1 + fields2)))
    coeff_dict = V.as_coefficients_dict()
    
    # Loop over indices in the first group assuming symmetry: i <= j.
    for i in range(n1):
        for j in range(i, n1):
            prod1 = fields1[i] * fields1[j]
            # Loop over indices in the second group assuming symmetry: k <= l.
            for k in range(n2):
                for l in range(k, n2):
                    fieldterm = prod1 * fields2[k] * fields2[l]
                    if fieldterm not in cache:
                        ##term = V.coeff(fieldterm).subs(fields_to_zero) #Costly part
                        ##coeff = polyV.coeff_monomial(fieldterm)
                        ##term = coeff.subs(fields_to_zero)
                        term = coeff_dict.get(fieldterm, 0)
                        cache[fieldterm] = np.float64(term)
                    value = cache[fieldterm]
                    L4[i, j, k, l] = value
                    # Now fill in all the symmetric entries.
                    if i != j:
                        L4[j, i, k, l] = value
                    if k != l:
                        L4[i, j, l, k] = value
                    if i != j and k != l:
                        L4[j, i, l, k] = value
    return L4




#####################################
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
def calculate_couplings3_old(n1, n2, precomputed_coeffs):
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
def calculate_couplings4_old(n1, n2, precomputed_coeffs):
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