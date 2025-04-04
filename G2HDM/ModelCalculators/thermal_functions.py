import numpy as np
from scipy.integrate import quad


# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *
from ..utils import constants
from .couplings import *

# ========= Thermal potentials =========
# Arnold–Espinosa method: VT -> VT + V_daisy (no replacement of M -> M + PI)
# Parwani method: First replace M -> M + PI = MT, then do all calculations with MT instead 
# (fermions do not receive mass corrections, only bosons)

# Need to run this once for each set of masses (higgs, gauge bosons, fermions)
def V_T(masses_higgs, masses_gauge, masses_quarks, T):

    ####################################
    # TESTING
    """
    new_masses_higgs = []
    for i, mass in enumerate(masses_higgs):
        if i in [0,4,5,6,7]:
            new_masses_higgs.append(0)
        else:
            new_masses_higgs.append(mass)
    masses_higgs = new_masses_higgs
    """
    ####################################
    N_h = 8
    N_g = 4
    N_q = 1
    
    n_h = 1
    n_g = 2
    n_q = 9
    
    int_range = [0, 10]
    coeff = T**4/(2*np.pi**2)
    result = 0
    for mass in masses_higgs:
        x = mass/T**2
        result += n_h * J_boson_integral(x, int_range)
    for mass in masses_gauge:
        x = mass/T**2
        result += n_g * J_boson_integral(x, int_range)
    """
    for i in range(N_q):
        x = masses_quarks[i]/T**2
        result += n_q * J_fermion_integral(x, range=[0, np.inf])
    """
        
    return coeff * result
        

# Need to run this once for each set of masses (higgs, gauge bosons, fermions?)
# Accounts for the transversal and longitudunal masses
# Arnold–Espinosa method uses this
def V_daisy(masses, masses_T, T):
    result = 0
    coeff = -T/(12*np.pi)
    for mass, mass_t in zip(masses, masses_T):
        result += mass_t**(1.5) - mass**(1.5)
        
    return coeff * result


# ========== Mass corrections ==========

def daisy_mass_corrections_higgs(L4_higgs, L4_gauge, L3_fermion, T):
    coeff = T**2/24
    N_h = 8
    N_g = 4
    N_f = 1
    result = np.zeros((N_h,N_h))
    n_h = 1
    n_g = 3
    n_f = 9
    for i in range(N_h):
        for j in range(N_h):
            for a in range(N_h):
                result[i,j] += coeff * n_h * (L4_higgs[i,j,a,a])
            for b in range(N_g):
                result[i,j] += coeff * n_g * (L4_gauge[b,b,i,j])    
            """
            for I in range(N_f):
                for J in range(N_f):
                    result[i,j] += coeff * 1/2 * (np.conjugate(L3_fermion[I,J,j])*L3_fermion[I,J,i] + np.conjugate(L3_fermion[I,J,i])*L3_fermion[I,J,j])
            """
    return result    

    
def daisy_mass_corrections_gauge(L4_gauge, T, use_debye_mass=True):
    N = 4
    n = 4 # Number of higgs fields that couples to the gauge boson n leq n_higgs
    coeff = T**2 * 2/3 * (n/8 + 5) / n
    result = np.zeros((N,N))
    for a in range(N):
        # a = b, kronecker delta
        for m in range(N):
            result[a,a] = coeff * L4_gauge[a,a,m,m]
    
    # Debye mass corrections
    if use_debye_mass:
        g = 1
        db_0 = 1/24 * T**2 * g**2
        Pi_0 = np.diag([db_0, db_0, db_0, db_0])
        result += Pi_0

    return result
 
# ========== Thermal functions ==========

# J-
def J_boson_integral(x, range=[0, np.inf]):
    sign = np.sign(x)
    x = np.abs(x)
    def f(k):
        return k**2 * np.log(1 - np.exp(-np.sqrt(k**2+x)))
    
    
    result, error = quad(f, range[0], range[1])
    return result

# J+
def J_fermion_integral(x, range=[0, np.inf]):
    x = np.abs(x)
    def f(k):
        return -k**2 * np.log(1 + np.exp(-np.sqrt(k**2+x)))
    
    
    result, error = quad(f, range[0], range[1])
    return result

# ========== Series expansion for low and high temperature limits (faster evaluation) ==========

#constants
gammaE = 1
cp = 1
cm = 1
xsqp = 1
ssqm = 1
dp = 1
dm = 1

def J_boson_small():
    pass

def J_boson_large():
    pass

def J_fermion_small():
    pass

def J_fermion_large():
    pass