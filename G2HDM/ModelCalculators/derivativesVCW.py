
import numpy as np
import symengine as se
import numba as nb
import time
import sympy as sp

# Custom imports
from ..Model2HDM import Model2HDM
#from .ModelDataCalculator import ModelDataCalculator
from .couplings import calculate_couplings3, calculate_couplings4


# Class for calculating the Coleman-Weinberg potential derivatives
class CWPotentialDerivativeCalculator:
    """ Calculates the Coleman-Weinberg potential derivatives for a given model 
    and a specific background field value, given at the initialization of the object."""
    
    # Constants
    threshold = 1e-10  
    epsilon = 1/(4*np.pi)**2
    
    def __init__(self, MDC:object, model:Model2HDM, subs_bgfield_values:dict, regulator=246**2): #
        
        # Model and MDC
        self.model = model
        self.MDC = MDC
        
        # Constants
        self.mu = MDC.mu
        
        # Substitutions
        self.subs_bgfield_values = subs_bgfield_values
        
        # Masses
        self.masses_higgs, self.R_higgs = MDC.calculate_masses_higgs(subs_bgfield_values) #diagonalize_numerical_matrix(M0_numerical, sorting=True)
        self.masses_gauge, self.R_gauge = MDC.calculate_masses_gauge(subs_bgfield_values)
        #self.masses_fermions, self.R_fermions = MDC.calculate_masses_fermions(subs_bgfield_values)

        # Add a regulator to the masses
        self.regulator = regulator
        self.masses_higgs = self.masses_higgs #+ self.regulator

        # Fields
        self.massStates_higgs = self.R_higgs @ model.massfields
        self.massStates_gauge = self.R_gauge @ model.massfields_gauge
        #self.massStates_fermions = sp.Matrix(self.R_fermions) * sp.Matrix(model.massfields_fermions)

        # Potentials (This is a time sink)
        subs_massStates_higgs = {field:state for field,state in zip(model.fields, self.massStates_higgs)}
        subs_massStates_gauge = {field:state for field,state in zip(model.fields_gauge, self.massStates_gauge)}

        # Potentials (This is a time sink), computed with symengine
        V0_se = se.sympify(MDC.V0_simplified).xreplace(subs_bgfield_values) 
        V0_se = V0_se.subs(subs_massStates_higgs).expand()
        self.V0 = V0_se #sp.sympify(V0_se)
        
        self.L_kin = MDC.L_kin_simplified.xreplace(subs_bgfield_values | subs_massStates_gauge).expand()

        # Couplings
        self.L3_higgs = self.calculate_couplings3(self.V0, model.massfields, model.massfields)
        self.L3_gauge = self.calculate_couplings3(self.L_kin, model.fields_gauge, model.massfields)
        #self.L3_fermions = self.calculate_couplings3(self.L_yuk, model.fields_fermions, model.massfields)
        
        self.L4_higgs = self.calculate_couplings4(self.V0, model.massfields, model.massfields) # ca 0.15s
        self.L4_gauge = self.calculate_couplings4(self.L_kin, model.fields_gauge, model.massfields)
        #self.L4_fermions = self.calculate_couplings4()

    def first_derivative(self):
        # First derivative terms
        NCW_higgs = self.Ni(L3=self.L3_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        NCW_gauge = self.Ni(L3=self.L3_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        NCW_fermions = np.zeros(8) #Ni(L3, masses, kT, coeff)
        NCW = NCW_higgs + NCW_gauge + NCW_fermions
        NCW = self.epsilon * self.R_higgs @ NCW # numpy matrix mul (j to i)  #sp.Matrix(NCW)
        NCW = np.where(np.abs(NCW) < self.threshold, 0, NCW)
        return NCW #sp.Matrix(NCW) 
    
    def second_derivative(self):
        MCW_higgs = self.Hij(L3=self.L3_higgs, L4=self.L4_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        MCW_gauge = self.Hij(L3=self.L3_gauge, L4=self.L4_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        MCW_fermions = np.zeros((8,8)) # Hij(L3, L4, masses, kT, coeff)
        MCW = MCW_higgs + MCW_gauge + MCW_fermions
        MCW = self.epsilon * self.R_higgs @ (MCW+MCW.T)/2 @ self.R_higgs.T
        MCW = np.where(np.abs(MCW) < self.threshold, 0, MCW)
        

        return MCW #sp.Matrix(MCW)
  
        
####### Helper functions #######

# log term
def f1(msq, mu, regulator):
    sign = 1
    if msq < 0:
        msq = -msq
        sign = -1
        
    if msq==0: #.is_zero:
        result = 0
    else:
        # Not IR div since this will be mult by the mass later
        # Add threshold here?
        # Apply IR regulator 
        msq += regulator
        result = sign*np.log(msq/mu**2) #m * (sp.log(m/mu**2)-kT+1/2)
    return result

# log term 2, with regulator f2
def f2(self, m1sq, m2sq, mu, regulator):
    #return 1
    sign1 = 1
    sign2 = 1
    if m1sq < 0:
        m1sq = -m1sq
        sign1 = -1
    if m2sq < 0:
        m2sq = -m2sq
        sign2 = -1
    
    # Apply IR regulator to all masses
    m1sq_R = m1sq + regulator
    m2sq_R = m2sq + regulator
    
    # calc
    log1 = 0
    log2 = 0
    """if m1sq == 0 and m2sq == 0:
        return 1
    if np.abs(m1sq-m2sq)<1e-5:
        return 1
    if m1sq != 0:
        log1 = sign1*np.log(m1sq_R/mu**2)
    if m2sq != 0: #and sp.Abs(m1sq-m2sq)>1e-5:
        log2 = sign2*np.log(m2sq_R/mu**2)
    else:
        return 1 + log1
        
    return (m1sq*log1 - m2sq*log2)/(m1sq-m2sq)
    """
    
    if m1sq == 0 and m2sq == 0:
        return 1
    if m1sq != 0:
        log1 = sign1*sp.log(m1sq_R/mu**2)
        log2 = 0
    
    if sp.Abs(m1sq-m2sq)>1e-5: # add threshold
        log2 = 0  
        if m2sq != 0:
            log2 = sign2*sp.log(m2sq_R/mu**2)
        if m1sq == 0:
            return log2
        elif m2sq == 0:
            return log1
        else:
            #print(log1, m2sq/m1sq, (m1sq*sp.Abs(log1) - m2sq*sp.Abs(log2))/(m1sq-m2sq))
            return (m1sq*sp.Abs(log1) - m2sq*sp.Abs(log2))/(m1sq-m2sq) #log1/(1-m2sq/m1sq)
    else: 
        return 1 + log1
    

#@jit
def Ni(L3, masses, kT, coeff, mu):
    """
    Compute the quantity Ncw[j] = sum_a L3[a,a,j] * masses[a] * (f1(masses[a], mu) - kT + 0.5)
    and return coeff * Ncw.
    
    L3 is assumed to be a NumPy array of shape (n, n, 8), and masses is a list or array.
    """
    masses = np.array(masses)  # ensure it's a NumPy array
    n = len(masses)
    N = 8
    
    # Precompute f1 for each mass
    f1_vals = np.array([f1(m, mu) for m in masses])
    
    # Extract the diagonal elements L3[a,a,:] for each a (resulting in an array of shape (n, N))
    # One could also use np.einsum('aaj->aj', L3) if L3 is a full array.
    L3_diag = np.array([L3[a, a, :] for a in range(n)])
    # Compute contributions: for each a, shape (n, N)
    contributions = L3_diag * masses[:, None] * (f1_vals - kT + 0.5)[:, None]
    
    # Sum over a to get an array of shape (N,)
    Ncw = contributions.sum(axis=0)
    return coeff * Ncw

#@jit
def Hij(L3, L4, masses, kT, coeff, mu):
    """
    Compute a symmetric matrix Mcw[i,j] from the L3 and L4 couplings.
    
    The first term is:
    sum_{a,b} L3[a,b,i] * L3[b,a,j] * (f2(masses[a], masses[b], mu) - kT + 0.5)
    The second term (only if masses[a] != 0) is:
    sum_{a} L4[a,a,i,j] * masses[a] * (f1(masses[a], mu) - kT + 0.5)
    
    L3 is assumed to have shape (n, n, 8) and L4 shape (n, n, 8, 8).
    """
    masses = np.array(masses)
    n = len(masses)
    N = 8
    
    # -----------------------------
    # First term: use np.einsum for double sum over a and b.
    # Precompute the f2 matrix F2[a,b] = self.f2(masses[a], masses[b], mu)
    F2 = np.empty((n, n))
    for a in range(n):
        for b in range(n):
            F2[a, b] = f2(masses[a], masses[b], mu)
    factor = F2 - kT + 0.5
    
    # L3 has shape (n, n, N). We need L3[a,b,i] and L3[b,a,j].
    # Use swapaxes to get L3_swapped[b,a,j] = L3[a,b,j].
    L3_swapped = np.swapaxes(L3, 0, 1)
    
    # Compute the double sum over a and b:
    # term1[i,j] = sum_{a,b} L3[a,b,i] * L3[b,a,j] * factor[a,b]
    term1 = np.einsum('abi,abj,ab->ij', L3, L3_swapped, factor)
    
    # -----------------------------
    # Second term: contributions from L4
    # Precompute f1 for each mass, and only add if the mass is nonzero.
    # (You might want to decide what to do when mass==0; here we simply set f1 to 0.)
    f1_vals = np.array([f1(m, mu) if m != 0 else 0 for m in masses])
    
    # Extract the diagonal of L4: L4_diag[a,:,:] = L4[a,a,:,:] (shape: (n, N, N))
    L4_diag = np.array([L4[a, a, :, :] for a in range(n)])
    
    # term2[i,j] = sum_{a} L4[a,a,i,j] * masses[a] * (f1_vals[a] - kT + 0.5)
    term2 = np.einsum('aij,a->ij', L4_diag, masses * (f1_vals - kT + 0.5))
    
    # -----------------------------
    # Combine both contributions
    Mcw = term1 + term2
    
    
    return coeff * Mcw