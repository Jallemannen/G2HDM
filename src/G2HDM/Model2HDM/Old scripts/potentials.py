# Methods for the model class

# Calculate = Numerical solution
# Generate / Solve = Analytical solution

# Imports
import os
import sympy as sp  
import numpy as np  
import matplotlib.pyplot as plt  
import time as time
from IPython.display import display, Math  

# Custom imports
#from .class_Model2HDM import Model2HDM
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from src.utils import constants as const

#################### Gauge bosons and fermions ####################

def generate_masses_gaugebosons_OLD(model, T=0, show_procedure=False, show_solution=False):
    
    # unpacking
    bgfield1 = model.bgfield1
    bgfield2 = model.bgfield2 
    
    # Symbols
    W1, W2, W3 = sp.symbols("W1 W2 W3", real=True)
    B = sp.Symbol("B", real=True)
    g1 = sp.Symbol("g_1", real=True)
    g2 = sp.Symbol("g_2", real=True)
    Wp = sp.symbols("W^+", real=True)
    Wm = sp.symbols("W^-", real=True)
    Z = sp.symbols("Z", real=True) 
    A = sp.symbols("A", real=True)
    
    # Gauge boson substitutions
    #WW = g2**2 * (W1 + sp.I*W2) * (W1 - sp.I*W2)
    WP = (W1 + sp.I*W2)
    WM = (W1 - sp.I*W2)
    ZZ = (g1*B - g2*W3)**2 
    AA = (g1*B + g2*W3)**2
    
    subs_gauge = {WP: Wp, 
                  WM:Wm, 
                  ZZ: Z**2* sp.sqrt(g1**2 + g2**2), 
                  AA: A**2* sp.sqrt(g1**2 + g2**2)}

    # Calculation

    D = sp.Matrix([[g1*B+g2*W3, g2*(W1 - sp.I*W2)],
                    [g2*(W1 + sp.I*W2), g1*B-g2*W3]])
        
    result1 = dagger(D*bgfield1) * (D*bgfield1)
    new_result1 = sp.Rational(1,2) * result1[0].subs(subs_gauge)
    result2 = dagger(D*bgfield2) * (D*bgfield2)
    new_result2 = sp.Rational(1,2) * result2[0].subs(subs_gauge)
    
    result = new_result1 + new_result2
    
    # masses
    mWp_sq = result.diff(Wp, Wp)/2
    mWm_sq = result.diff(Wm, Wm)/2
    mWpm_sq = result.diff(Wp, Wm)
    mZ_sq = result.diff(Z, Z)/2
    mA_sq = result.diff(A, A)/2
    
    # Display
    if show_procedure or show_solution:
        print(" Kinetic term ".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{kin}} =" + sp.latex(result)))
        
        print("========== Gauge boson masses ==========".center(50, "="))
        present(Math("m_{W^+}^2" + "=" + sp.latex(mWp_sq)))
        present(Math("m_{W^-}^2" + "=" + sp.latex(mWm_sq)))
        present(Math("m_{W_\pm}^2" + "=" + sp.latex(mWpm_sq)))
        present(Math("m_{Z}^2" + "=" + sp.latex(mZ_sq)))
        present(Math("m_{A}^2" + "=" + sp.latex(mA_sq)))
        print("Mass ratio:")
        present(Math(r"\cos^2(\theta_W) = \frac{m^2_{W_\pm}}{m^2_{Z}}" + "=" + sp.latex(mWpm_sq/mZ_sq)))
    
    
    """if T==0:
        m_W = const.g1/sp.sqrt(2) * (h1_bg.T * h1_bg)
        m_Z = sp.sqrt(const.g1**2 + const.g2**2)/sp.sqrt(2) * (h1_bg.T * h1_bg)
        m_A = 0
        return m_W, m_Z, m_A
    else:
        return 0,0,0"""
        
    return [mWpm_sq, mZ_sq, mA_sq]
    
def generate_masses_fermions_OLD(model, T=0, type="I", only_top=True):
    
    # coupling constants
    yt = sp.Symbol("y_t", real=True)
    yb = sp.Symbol("y_b", real=True)

    bgfield1 = model.bgfield1
    bgfield2 = model.bgfield2

    bgfield1_c = sp.Matrix([bgfield1[1], bgfield1[0]])
    bgfield2_c = sp.Matrix([bgfield2[1], bgfield2[0]])

    # Symbols
    tL = sp.symbols("t_L", real=True)
    bL = sp.symbols("b_L", real=True)
    tR = sp.symbols("t_R", real=True)
    bR = sp.symbols("b_R", real=True)
    QLt = sp.Matrix([[tL], [bL]])  
    
    if T==0:
        if only_top:
            QLt = sp.Matrix([[tL], [0]])  
            
            Lt1 = yb * dagger(QLt) * bgfield1 * bR + yt * dagger(QLt) * bgfield1_c * tR
            Lt2 = yb * dagger(QLt) * bgfield2 * bR + yt * dagger(QLt) * bgfield2_c * tR
            if type == "I":
                L_Yuk = Lt1[0]
            if type == "II":
                L_Yuk = Lt1[0] + Lt2[0]
            mt_sq = L_Yuk.diff(tR, tL)
            
            return [mt_sq]
        
    return 0


def generate_kinetic_term_gauge(model, using_SM_fields=False):
    # unpacking
    field1 = model.field1
    field2 = model.field2 
    
    # Symbols
    W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
    W0, Wp, Wm = sp.symbols("W_0 W^+ W^-", real=True)
    B = sp.Symbol("B", real=True)
    g1 = sp.Symbol("g_1", real=True)
    g2 = sp.Symbol("g_2", real=True)
    
    # mass states
    Wp = sp.symbols("W^+", real=True)
    Wm = sp.symbols("W^-", real=True)
    Z = sp.symbols("Z^0", real=True) 
    A = sp.symbols("A", real=True)
    
    # Gauge boson substitutions
    if using_SM_fields:
        WP =  g2*Wp #g2*(-W1 + sp.I*W2)
        WM =  g2*Wm #g2*(-W1 - sp.I*W2)
        Z0 = Z * (g1**2 + g2**2)**sp.Rational(1,2) #(-g1*B + g2*W3) * (g1**2 + g2**2)**sp.Rational(1,2)
        A0 = A * (g1**2 + g2**2)**sp.Rational(1,2) #(g1*B + g2*W3) * (g1**2 + g2**2)**sp.Rational(1,2)

        D = sp.I*sp.Rational(1,2)*sp.Matrix([[A0, -WP],
                                            [-WM, -Z0]])
        D_dagger = -sp.I*sp.Rational(1,2)*sp.Matrix([[A0, -WP],
                                                    [-WM, -Z0]]) # transpose + complex conjugate Wm->Wp
    else:
        D = sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, g2*(W1 - sp.I*W2)],
                    [g2*(W1 + sp.I*W2), g1*B-g2*W3]])
        D_dagger = dagger(D)
        
        #D = sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, -sp.sqrt(2)*Wp],
        #                                    [-sp.sqrt(2)*Wm, g1*B-g2*W0,]])
        #D_dagger = -sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, -sp.sqrt(2)*Wp],
        #                                    [-sp.sqrt(2)*Wm, g1*B-g2*W0,]])
        
    result1 = (dagger(field1)*D_dagger) * (D*field1)
    
    result2 = (dagger(field2)*D_dagger) * (D*field2)
    
    result = result1[0] + result2[0]
    
    return result


def generate_yukawa_term_fermions(model, type="I", only_top=True):
    
    # coupling constants
    yt = sp.Symbol("y_t", real=True, positive=True)
    yb = sp.Symbol("y_b", real=True, positive=True)

    field1 = model.field1
    field2 = model.field2

    field1_c = sp.Matrix([field1[1], field1[0]])
    field2_c = sp.Matrix([field2[1], field2[0]])

    # Symbols
    tL = sp.symbols("t_L")
    bL = sp.symbols("b_L")
    tR = sp.symbols("t_R")
    bR = sp.symbols("b_R")
    ctL = sp.Symbol(r"\bar{t}_L")
    cbL = sp.Symbol(r"\bar{b}_L")
    ctR = sp.Symbol(r"\bar{t}_R")
    cbR = sp.Symbol(r"\bar{b}_R")
    QLt_dagger = sp.Matrix([[ctL], [cbL]]).T
    

    if only_top:
        QLt_dagger = sp.Matrix([[ctL], [0]]).T
    
    L_Yuk = 0
    Lt1 = yb * QLt_dagger * field1 * bR + yt * QLt_dagger * field1_c * tR
    Lt2 = yb * QLt_dagger * field2 * bR + yt * QLt_dagger * field2_c * tR
        
    if type == "I":
        L_Yuk += Lt1[0]
    if type == "II":
        L_Yuk += Lt1[0] + Lt2[0]
        
    return L_Yuk


def generate_masses_gaugebosons(model, show_procedure=False, show_solution=False):
    
    L_kin = generate_kinetic_term_gauge(model, using_SM_fields=False)
    subs_fields = {f:0 for f in model.fields}
    
    W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
    #W0, Wp, Wm = sp.symbols("W_0 W^+ W^-", real=True)
    B = sp.Symbol("B", real=True)
    gaugefields = [B, W1, W2, W3]
    #gaugefields = [B, W0, Wp, Wm]
    #fields_gauge = [B, W0, Wp/sp.sqrt(2), Wm/sp.sqrt(2)]
    
    H = sp.hessian(L_kin.subs(subs_fields).expand(), gaugefields)
    #display(H)
    eigen_data = H.eigenvects()
    eigenvalues = [ev[0] for ev in eigen_data] #*sp.Rational(1,2) 
    eigenvectors = [ev[2] for ev in eigen_data]
    #display(eigenvalues, eigenvectors)
    #display(eigenvectors)

    evals = []
    evects = []
    states = []
    for eval, evec in zip(eigenvalues, eigenvectors):
        for vec in evec:
            evals.append(eval)
            evects.append(vec)
            states.append((sp.Matrix(vec).T * sp.Matrix(gaugefields))[0])
            
    R = sp.Matrix.hstack(*evects)
    masses = evals
    # is eval is neg --> set neg eigenstate instead
    
    # Filter negative masses 
    """masses1 = masses.copy()
    masses2 = masses.copy()
    states_sorted = []
    masses_sorted = []
    for i, mass1 in enumerate(masses1):
        term_mass = 0
        term_state = 0
        for j, mass2 in enumerate(masses2):
            if mass2.coeff(mass1) == -1:
                #assuming two states are mixed:
                term_mass = sp.Abs(mass2)*2
                term_state = (states[i]*-states[j]/2).expand()
        masses_sorted.append(term_mass)    
        states_sorted.append(term_state)   
    """          

    if show_procedure or show_solution:
        print(" Kinetic term ".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{kin}} =" + sp.latex(L_kin.subs(subs_fields))))
        
        print("========== Gauge boson masses ==========".center(50, "="))
        for i in range(int(len(masses))):
            present(Math("m^2" + "=" + sp.latex(masses[i])))
            present(Math(sp.latex(states[i]**2)))
        
        
        """# Mult fields if they have the same eigenvalue. 
        print("========== Gauge boson masses ==========".center(50, "="))
        present(Math("m_{W_\pm}^2" + "=" + sp.latex(masses[1]+masses[2])))
        present(Math("m_{Z}^2" + "=" + sp.latex(masses[3])))
        present(Math("m_{A}^2" + "=" + sp.latex(masses[0])))
        print("Mass ratio:")
        present(Math(r"\cos^2(\theta_W) = \frac{m^2_{W_\pm}}{m^2_{Z}}" + "=" + sp.latex((masses[1]+masses[2])/masses[3])))
        """
        
        model.DATA["M_gauge"] = H
        model.DATA["M_gauge_eigenvalues"] = masses
        
    return masses, states, R 
    

def generate_masses_fermions(model, type="I", only_top=True, show_procedure=False, show_solution=False): 
    # coupling constants

    subs_fields = {f:0 for f in model.fields}
    L_yuk = generate_yukawa_term_fermions(model, type="I", only_top=only_top).subs(subs_fields)
    display(L_yuk)
    # Symbols
    tL = sp.symbols("t_L")
    bL = sp.symbols("b_L")
    tR = sp.symbols("t_R")
    bR = sp.symbols("b_R")
    ctL = sp.Symbol(r"\bar{t}_L")
    cbL = sp.Symbol(r"\bar{b}_L")
    ctR = sp.Symbol(r"\bar{t}_R")
    cbR = sp.Symbol(r"\bar{b}_R")
    
    if only_top:
        fields = sp.Matrix([tL, tR, bL, bR])
        fields_conjugated = sp.Matrix([ctL, ctR, cbL, cbR])
        H = sp.Matrix([[0 for _ in fields] for _ in fields])
        
        #display(L_yuk.diff(ctL, tR), L_yuk.diff(ctL, tR))
        for i in range(len(fields)):
            #display(L_yuk.diff(fields_conjugated[i]), fields_conjugated[i])
            for j in range(len(fields)):
                H[i,j] += L_yuk.diff(fields[i], fields_conjugated[j])/2
                H[i,j] += L_yuk.diff(fields_conjugated[i], fields[j])/2
        #display(H)
        eigen_data = H.eigenvects()
        eigenvalues = [ev[0]*sp.Rational(1,2) for ev in eigen_data]
        eigenvectors = [ev[2] for ev in eigen_data]
        
    masses = []
    states = []
    states_conjugated = []
    for eval, evec in zip(eigenvalues, eigenvectors):
        for vec in evec:
            masses.append(eval)
            states.append(vec.T * sp.Matrix(fields))
            states_conjugated.append(vec.T * sp.Matrix(fields_conjugated))
            
    if show_procedure or show_solution:
        print("========== Yukawa term ==========".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{Yuk}} =" + sp.latex(L_yuk.subs(subs_fields))))
        # Mult fields if they have the same eigenvalue. 
        for i in range(int(len(masses))):
            print("========== Fermion masses ==========".center(50, "="))
            present(Math("m^2" + "=" + sp.latex(masses[i])))
            state = (states_conjugated[i]*states[i]).expand()
            display(Math(sp.latex(state)))
        
    return masses, states
#################### Potentials (Temperature independent) ####################

# Generate the analytical solution of the Coleman-Weinberg potential
def generate_VCW(model, only_top=True):
    
    
    # Constants
    v = const.v
    mu = v
    
    # Coleman weinberg potential function
    def V_CW(mass, dof, C):

        mass_squared = mass
        display(mass_squared)
        # Handle zero and negative masses
        if mass_squared.is_zero:
            log_term = 0
        else:
            #log_term = sp.sign(mass_squared)*sp.log(sp.Abs(mass_squared) / mu**2)
            log_term = sp.log(mass_squared / mu**2)

        factor = dof / (64 * sp.pi**2)

        # Add this particle's contribution to the effective potential
        return factor * mass_squared**2 * (log_term - C)
    
    
    #
    masses_gauge, _ = generate_masses_gaugebosons(model)
    masses_fermions, _ = generate_masses_fermions(model, type="I", only_top=only_top)
    
    
    #display(masses_gauge, masses_fermions)
    
    
    
    # Unpacking
    masses_higgs = model.DATA["M0_eigenvalues"]
    
    # mass eigenvalues
    masses = masses_higgs + masses_gauge + masses_fermions
    
    # Spins
    spins_higgs = [0 for _ in masses_higgs]
    spins_gauge = [1 for _ in masses_gauge]
    spins_fermions = [1/2 for _ in masses_fermions]

    # Dof
    dofs_higgs = [1 for _ in masses_higgs] 
    dofs_gauge = []
    for mass in masses_gauge:
        if mass.is_zero:
            dofs_gauge.append(2)
        else:
            dofs_gauge.append(3)
            
    dofs_fermions = [12]
    
    # Scheme constants
    scheme_constants_higgs = [3/2 for _ in masses_higgs]
    scheme_constants_gauge = [3/2 for _ in masses_gauge]
    scheme_constants_fermions = [5/6 for _ in masses_fermions]

    # Generate the potential
    Masses = masses_higgs + masses_gauge #+ masses_fermions
    Dofs = dofs_higgs + dofs_gauge #+ dofs_fermions
    Scheme_constants = scheme_constants_higgs + scheme_constants_gauge #+ scheme_constants_fermions
    V_cw_expr = 0
    for masses, dofs, scheme_constants in zip(Masses, Dofs, Scheme_constants):
        V_cw_expr += V_CW(masses, dofs, scheme_constants)
    
    return V_cw_expr

# Generate analytical effective potential
def generate_Veff(T=0):
    pass


def calculate_VCW():
    pass

def calculate_Veff():
    pass

#################### Potential derivatives ####################

# Calc N_i and Hij from roman 2016 paper
def calculate_CW_potential_derivatives_numerical(M0_numerical, V0_fdep, fields, fields_mass,
                                                 show_procedure=False):
    """_summary_

    Args:
        M0_numerical: _description_
        V0: Should be evaluated for all values except for the fields
        fields: _description_
        fields_mass: _description_
        show_procedure: _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    v = const.v
    mu = v
    
    #M0_neutal = M0[:4, :4]
    #M0_charged = M0[4:, 4:]
    #R_n = compute_diagonalization_matrix(M0_neutal)
    #R_c = compute_diagonalization_matrix(M0_charged)
    
    #, params_values_subs, bgfields_values_subs, VEV_values_subs
    #M0 = M0.subs(params_values_subs)
    #M0 = M0.subs(VEV_values_subs)
    #M0 = M0.subs(bgfields_values_subs)
    
    masses, field_states, R = diagonalize_numerical_matrix(M0_numerical)
    
    # Check for imaginary masses
    for mass in masses:
        if isinstance(mass, complex):
            if mass.imag < mass.real*1e-5:
                mass = mass.real
            print("Imaginary mass found!:", mass)
                

    ### Calculate mass eigenstates in terms of the fields
    fields_states_mass = sp.Matrix(R) * sp.Matrix(fields_mass)
    fields_states = sp.Matrix(R).inv() * sp.Matrix(fields)
    fields_states_mass_subs = {f:fm for f,fm in zip(fields, fields_states_mass)}
    fields_mass_to_zero = {f:0 for f in fields_mass}
    fields_to_zero = {f:0 for f in fields_states}
    
    # Potential in terms of mass eigenstates
    V0_mass = V0_fdep.subs(fields_states_mass_subs).expand()

    #for key, value in fields_states_mass_subs.items():
    #    display(Math(f"{key} = {value}"))
    
    #display(*fields_states_mass_subs.items())
    
    ### Calculate the trilinear and quartic couplings
    from itertools import permutations
    L3 = np.zeros((8,8,8))
    L4 = np.zeros((8,8,8,8))
    
    cprint("Calculating the trilinear couplings...", show = show_procedure)
    combinations3, S3 = compute_unique_permutations(8,3)
    
    n=0
    for i,j,k in combinations3:
        fieldterm = fields_mass[i]*fields_mass[j]*fields_mass[k]
        term = V0_mass.coeff(fieldterm).expand().subs(fields_mass_to_zero)
        for a,b,c in permutations([i,j,k]):
            L3[a,b,c] = sp.re(term)
        n+=1
        if n % 32 == 0:
            cprint(n, f"/ {len(combinations3)} ", end="\r", show = show_procedure)
    
    cprint("\nCalculating the quartic couplings...", show = show_procedure)
    combinations4, S4 = compute_unique_permutations(8,4)
    for i,j,k,l in combinations4:
        fieldterm = fields_mass[i]*fields_mass[j]*fields_mass[k]*fields_mass[l]
        term = V0_mass.coeff(fieldterm).expand().subs(fields_mass_to_zero)
        for a,b,c,d in permutations([i,j,k,l]):
            L4[a,b,c,d] = term
        n+=1
        if n % 32 == 0:
            cprint(n, f"/ {len(combinations4)} ", end="\r", show = show_procedure)
    
    ### Calculate the derivatives
    NCW = np.zeros(8)
    MCW = np.zeros((8,8))
    
    kT = 3/2
    sT = 0 # for all scalar particles
    coeff = (-1)**(2*sT)*(1+2*sT) / 2
    coeff = 1/2
    epsilon = 1/(4*np.pi)**2
    
    # log term 1, with regulator
    def g(m):
        is_negative = False
        sign = 1
        if m < 0:
            is_negative = True
            m = -m
            sign = -1
            
        if m==0: #.is_zero:
            result = 0
        else:
            result =  m * (sp.log(m/mu**2)-kT+1/2)
            
        if is_negative:
            return -result
        else:
            return result
    
    # log term 2, with regulator
    def f(m1, m2):
        m1_is_negative = False
        m2_is_negative = False
        sign1 = 1
        sign2 = 1
        if m1 < 0:
            m1_is_negative = True
            m1 = -m1
            sign1 = -1
        if m2 < 0:
            m2_is_negative = True
            m2 = -m2
            sign2 = -1
        
        # calc
        if m1 == 0 and m2 == 0:
            return 1
        if m1 != 0:
            log1 = sign1*sp.log(m1/mu**2)
            log2 = 0
            
        if sp.Abs(m1-m2)>1e-5: # add threshold
            if m2 != 0:
                log2 = sign2*sp.log(m2/mu**2)
            if m1 == 0:
                log1 = 0
            if m2 == 0:
                log2 = 0        
        else: 
            return 1 + sign1*sp.log(m1/mu**2)
            
        return (m1*log1 - m2*log2)/(m1-m2)
            
        """elif sp.Eq(m1,m2) or m2 == 0: #m2.is_zero: # IR divergency with regulator
            if m1_is_negative:
                return 1 - sp.log(m1/mu**2)
            else:
                return 1 + sp.log(m1/mu**2)
        elif m1 == 0:  # m1.is_zero: 
            if m2_is_negative:
                return 1 - sp.log(m2/mu**2)
            else:
                return 1 + sp.log(m2/mu**2)"""
        #return (m2*sp.log(m2/mu**2))/(-m2)
        """else:
            term1 = m1*sp.log(m1/mu**2) / (m1-m2)
            term2 = m2*sp.log(m2/mu**2) / (m2-m1)
            if m1_is_negative:
                term1 = -term1
            if m2_is_negative:
                term2 = -term2
            return term1 + term2"""
            #return (m1*sp.log(m1/mu**2)-m2*sp.log(m2/mu**2))/(m1-m2)

    # Calculate the Ni derivatives
    cprint(r"Calculating the $N_i$ derivatives...", show = show_procedure)
    n=0
    
    for i in range(8):
        for j in range(8):
            for a in range(8):
                NCW[i] += epsilon * coeff * R[i,j] * L3[a][a][j] * g(masses[a]) #M[a] * (sp.log(M[a]/mu**2)-kT+1/2)
                n += 1
                if n % 64 == 0:
                    cprint(n, "/ 512 ", end="\r", show = show_procedure)
      

    cprint(r"\nCalculating the $H_{ij}$ derivatives...", show = show_procedure)
    n=0
    
    Terms1 = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            term = 0
            for c in range(8):
                term += L4[c][c][i][j] * g(masses[c]) #M[c] * (sp.log(M[c]/mu**2)-kT+1/2)
                n += 1
                if n % 64 == 0:
                    cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            Terms1[i][j] = term
        
    Terms2 = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            term = 0
            for a in range(8):
                for b in range(8):
                    F = f(masses[a],masses[b])
                    term += L3[a][b][i] * L3[a][b][j] * (F-kT+1/2) 
                    n += 1
                    if n % 64 == 0:
                        cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            Terms2[i][j] = term 
           
    for k in range(8):
        for l in range(8):
            term = 0
            for i in range(8):
                for j in range(8):
                    term += coeff * epsilon * R[k,i] * R[l,j]  * (Terms2[i][j]+Terms1[i][j])
                    n += 1
                    if n % 64 == 0:
                        cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            MCW[k][l] = term



    return sp.Matrix(NCW), sp.Matrix(MCW)

# Calculate trilinear couplings
def calculate_couplings3(V, fields1, fields2):
    
    fields_to_zero = {f:0 for f in fields1+fields2}
    from itertools import permutations, product
    L3 = np.zeros((len(fields1),len(fields1),len(fields2)))
    #combinations3, S3 = compute_unique_permutations(8,3)
    range1 = range(0,len(fields1))
    range2 = range(0,len(fields2))
    combinations3 = set(product(range1, range1, range2))
    
    for i,j,k in combinations3:
        fieldterm = fields1[i]*fields1[j]*fields2[k]
        term = V.coeff(fieldterm).expand().subs(fields_to_zero)
        L3[i,j,k]  = sp.re(term)
        #for a,b in permutations([i,j]):
        #   L3[a,b,k] = sp.re(term)
    
    return L3

# Calculate quartic couplings
def calculate_couplings4(V, fields1, fields2):

    from itertools import permutations, product
    L4 = np.zeros((len(fields1),len(fields1),len(fields2),len(fields2)))
    fields_to_zero = {f:0 for f in fields1+fields2}
    #combinations4, S4 = compute_unique_permutations(8,4)
    range1 = range(0,len(fields1))
    range2 = range(0,len(fields2))
    combinations4 = set(product(range1, range1, range2, range2))


    for i,j,k,l in combinations4:
        fieldterm = fields1[i]*fields1[j]*fields2[k]*fields2[l]
        term = V.coeff(fieldterm).expand().subs(fields_to_zero)
        L4[i,j,k,l] = term
        #for a,b in permutations([i,j]):
        #    L4[a,b,k,l] = term

    return L4


# log term
def f1(msq, mu):

    sign = 1
    if msq < 0:
        msq = -msq
        sign = -1
        
    if msq==0: #.is_zero:
        result = 0
    else:
        result = sign*sp.log(msq/mu**2) #m * (sp.log(m/mu**2)-kT+1/2)
    return result
    
# log term 2, with regulator f2
def f2(m1sq, m2sq, mu):
    
    sign1 = 1
    sign2 = 1
    if m1sq < 0:
        m1sq = -m1sq
        sign1 = -1
    if m2sq < 0:
        m2sq = -m2sq
        sign2 = -1
    
    # calc
    log1 = 0
    log2 = 0
    if m1sq == 0 and m2sq == 0:
        return 1
    if m1sq != 0:
        log1 = sign1*sp.log(m1sq/mu**2)
        log2 = 0
        
    if sp.Abs(m1sq-m2sq)>1e-5: # add threshold
        log2 = 0  
        if m2sq != 0:
            log2 = sign2*sp.log(m2sq/mu**2)
        if m1sq == 0:
            return log2
        if m2sq == 0:
            return log1
        else:
            return (m1sq*log1 - m2sq*log2)/(m1sq-m2sq) 
    else: 
        return 1 + log1


def calculate_VCW_derivatives(model, omega, M0_numerical, V0_fdep, mu=const.v):
    """If you need both Ni and Hij, use this function, as it is more optimized."""
    
    show_procedure = True

    
    
    # add imag eval check inside here!
    masses_higgs, field_states, R = diagonalize_numerical_matrix(M0_numerical)
    masses_gauge, field_states_gauge, R_gauge = generate_masses_gaugebosons(model)

    ### Calculate mass eigenstates in terms of the fields
    fields = model.fields
    fields_mass = model.massfields
    fields_states_mass = sp.Matrix(R) * sp.Matrix(fields_mass)
    fields_states_mass_subs = {f:fm for f,fm in zip(fields, fields_states_mass)}
    #fields_states = sp.Matrix(R).inv() * sp.Matrix(fields)
    #fields_mass_to_zero = {f:0 for f in fields_mass}
    #fields_to_zero = {f:0 for f in fields_states}
    
    # Subs gauge fields
    W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
    B = sp.Symbol("B", real=True)
    fields_gauge = [B, W1, W2, W3]
    G0, G1, G2, G3 = sp.symbols("G_0 G_1 G_2 G_3", real=True)
    fields_gauge_mass = [G0, G1, G2, G3]
    fields_gauge_states_mass = sp.Matrix(R_gauge) * sp.Matrix(fields_gauge_mass)
    fields_gauge_states_mass_subs = {f:fm for f,fm in zip(fields_gauge, fields_gauge_states_mass)}
    g1 = sp.Symbol("g_1", real=True)
    g2 = sp.Symbol("g_2", real=True)
    subs_const = {g1:const.g1, g2:const.g2}
    
    ### Calc Couplings
    # Potential in terms of mass eigenstates
    V0_mass = V0_fdep.subs(fields_states_mass_subs).expand()
    V0_gauge = generate_kinetic_term_gauge(model, using_SM_fields=False)
    V0_gauge = V0_gauge.subs(fields_gauge_states_mass_subs).subs({model.symbol("omega"):omega}).expand()
    V0_gauge = V0_gauge.subs(fields_states_mass_subs).expand().subs(subs_const)
    masses_gauge = [m.subs(subs_const).subs({model.symbol("omega"):omega}) for m in masses_gauge]
    
    #display(V0_gauge.coeff(G1*G1*fields[0]), V0_gauge)
    # Calculate the couplings
    L3 = calculate_couplings3(V0_mass, fields_mass, fields_mass)
    L4 = calculate_couplings4(V0_mass, fields_mass, fields_mass)
    L3_gauge = calculate_couplings3(V0_gauge, fields_gauge_mass, fields_mass)
    L4_gauge = calculate_couplings4(V0_gauge, fields_gauge_mass, fields_mass)

    # Constants
    threshold = 1e-10  
    kT = 3/2
    sT = 0 # for all scalar particles
    coeff = (-1)**(2*sT)*(1+2*sT) / 2
    coeff_n = 1/2
    epsilon = 1/(4*np.pi)**2   
    
    # First derivative
    def Ni(L3, masses, kT, coeff):
        N = len(masses_higgs)
        n = len(masses)
        Ncw = np.zeros(N) 
        for j in range(N):
            for a in range(n):
                Ncw[j] += L3[a][a][j] * masses[a] * (f1(masses[a],mu)-kT+1/2) #M[a] * (sp.log(M[a]/mu**2)-kT+1/2)

        return coeff * Ncw

    # First derivative terms
    NCW_higgs = Ni(L3=L3, masses=masses_higgs, kT=3/2, coeff=1/2)
    NCW_gauge = Ni(L3=L3_gauge, masses=masses_gauge, kT=3/2, coeff=3/2)
    NCW_fermions = np.zeros(8) #Ni(L3, masses, kT, coeff)
    NCW = NCW_higgs + NCW_gauge + NCW_fermions
    NCW = epsilon * R @ NCW # numpy matrix mul (j to i)  #sp.Matrix(NCW)
    NCW = np.where(np.abs(NCW) < threshold, 0, NCW)

    # Second derivative
    def Hij(L3, L4, masses, kT, coeff):
        N = len(masses_higgs)
        n = len(masses)
        Mcw = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if j >= i: # use symmetry
                    for a in range(n):
                        for b in range(n):
                            logf2 = f2(masses[a],masses[b],mu)
                            Mcw[i,j] += L3[a][b][i] * L3[b][a][j] * (logf2-kT+1/2) 
                            if masses[a] != 0:
                                logf1 = f1(masses[a],mu)
                                Mcw[i,j] += L4[a][a][i][j] * masses[a] * (logf1-kT+1/2)
        return coeff * Mcw  
        
    # Second derivative terms
    MCW_higgs = Hij(L3=L3, L4=L4, masses=masses_higgs, kT=3/2, coeff=1/2)
    MCW_gauge = Hij(L3=L3_gauge, L4=L4_gauge, masses=masses_gauge, kT=3/2, coeff=3/2)
    MCW_fermions = np.zeros((8,8)) # Hij(L3, L4, masses, kT, coeff)
    MCW = MCW_higgs + MCW_gauge + MCW_fermions
    MCW = epsilon * R.T @ (MCW+MCW.T)/2 @ R 
    MCW = np.where(np.abs(MCW) < threshold, 0, MCW)
    #display(sp.Matrix(MCW))
    #display(MCW_higgs, MCW_gauge)

    return sp.Matrix(NCW), sp.Matrix(MCW)
        
        

    

    
def calculate_VCW_firstDerivative(M0_numerical, V0_fdep, fields, fields_mass, mu=const.v,
                                  L3=None, L4=None):
    pass

def calculate_VCW_secondDerivative(M0_numerical, V0_fdep, fields, fields_mass):
    pass


#################### Temperature-dependent Potentials ####################

def generate_VT_potental():
    pass

