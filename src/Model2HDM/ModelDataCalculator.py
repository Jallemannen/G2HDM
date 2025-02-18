# Imports
import os
import sympy as sp  
import numpy as np  
import matplotlib.pyplot as plt  
import time as time
from IPython.display import display, Math  

# Custom imports
from .class_Model2HDM import Model2HDM
from .potentials import *
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from src.utils import constants as const

####################################################################################

# Main/parent class
class ModelDataCalculator:
    """Generates numerical solutions for a given set of parameters and VEVs, which depends on omega."""
    def __init__(self, model:Model2HDM, subs_V0_params_values:dict, subs_VEV_values:dict, subs_bgfield_values:dict={}):
        
        self.model = model
        assert model.getdata("VCT_params_solution") is not None, "The analytical counterterm solutions has not been solved yet"
        
        # Constants
        self.mu = 246
        
        # Numerical values
        self.subs_VEV_values = subs_VEV_values
        self.subs_bgfield_values = subs_bgfield_values #Default values
        self.subs_V0_params_values = subs_V0_params_values
        
        # Precompute common substitutions and expressions.
        #self.fields_to_zero = {f: 0 for f in model.fields} | {f: 0 for f in model.fields_gauge}
        #self.subs_VEV_values = {symb:val for symb,val in zip(model.VEVs, VEV_values)}
        self.subs_bgfields_to_VEV = {symb:val for symb,val in zip(model.bgfields, model.VEVs)}
        self.subs_fields_to_zero = {field:0 for field in model.fields}
        #self.subs_V0_params_values = {param:value for param, value in zip(model.V0_params, V0_params_values)}
        
        # Gauge subs
        g1 = sp.Symbol("g_1", real=True)
        g2 = sp.Symbol("g_2", real=True)
        self.subs_constants = {g1:const.g1, g2:const.g2}

        # Simplified solutions
        self.V0_simplified = model.V0.subs(self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.L_kin_simplified = generate_kinetic_term_gauge(model).subs(self.subs_V0_params_values | self.subs_VEV_values | self.subs_constants).evalf()
        self.L_yuk_simplified = generate_yukawa_term_fermions(model).subs(self.subs_V0_params_values | self.subs_VEV_values | self.subs_constants).evalf()
        
        # Precompute expressions
        self.M0 = sp.hessian(self.V0_simplified, model.fields).subs(self.subs_fields_to_zero)
        self.M_kin = sp.hessian(self.L_kin_simplified, model.fields_gauge).subs(self.subs_fields_to_zero) #.subs(self.subs_constants).evalf()
        self.M_yuk = None

        # Counterterm values
        VCT_params_values = self.counterterm_values()
        self.subs_VCT_params_values = {param:value for param, value in zip(model.VCT_params, VCT_params_values)}
        self.VCT_simplified = model.VCT.subs(self.subs_VCT_params_values | self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.MCT = sp.hessian(self.VCT_simplified, model.fields).subs(self.subs_fields_to_zero)
        

    ########## Numerical Solutions ##########
    
    def calculate_masses_higgs(self, subs_bgfield_values):
        M_numerical = self.M0.subs(subs_bgfield_values).evalf()
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_gauge(self, subs_bgfield_values):
        M_numerical = self.M_kin.subs(subs_bgfield_values).evalf()
        display(M_numerical)
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_fermions(self, subs_bgfield_values):
        pass  
    
    ########## Numerical Solutions ##########
    
    # Calculate and assign the VCT parameter values
    def counterterm_values(self):
        #VCT_params, VCT_params_sol, VEV, VEV_values, M0, V0, fields, fields_mass, show_solution=False, 
        """ 
        V0: Should be evaluated for all values except for the fields
        """

        # Unpacking
        VCT_params = self.model.VCT_params
        VCT_params_eqs_sol = self.model.DATA.get("VCT_params_solution",None)
        assert VCT_params_eqs_sol is not None, "The analytical counterterm solutions has not been assigned"
        
        # Calculate the derivatives
        subs_bgfield_values = {bgfield:VEV_value for bgfield, VEV_value in zip(self.model.bgfields, self.subs_VEV_values.values())}
        VCW_deriv_calc = CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values)
        NCW = VCW_deriv_calc.first_derivative()
        MCW = VCW_deriv_calc.second_derivative()
        
        # OBS! Also need to consider extra constraints values later
        
        # create Ni and Hij subs dict
        Ni = generate_vector(8, "N")
        Hij = generate_matrix(8, "H", symmetric=True)
        
        equations_Ni = [sp.Eq(Ni[i], NCW[i]) for i in range(8)]
        equations_Hij = [sp.Eq(Hij[i,j], MCW[i,j]) for i in range(8) for j in range(8) if i <= j]
        
        subs_Ni = eqs_to_dict(equations_Ni)
        subs_Hij = eqs_to_dict(equations_Hij)
        
        # Assign the values
        VCT_params_values = [0 for i in range(len(VCT_params))]
        for i, eq in enumerate(VCT_params_eqs_sol):
            VCT_params_values[i] = eq.rhs.subs(self.subs_VEV_values).subs(subs_Ni | subs_Hij).evalf()
                    
        """# Display the results
        if show_solution:
            print("========== Counterterm values ==========".center(50, "="))
            for symb, value in zip(VCT_params, VCT_params_values):
                display(Math(sp.latex(symb) + "=" + sp.latex(value)))"""
            
        return VCT_params_values
    
    
    def VCW_first_derivative(self, subs_bgfield_values):
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values).first_derivative()

    def VCW_second_derivative(self, subs_bgfield_values):  
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values).second_derivative()
    
    ########## Mass Data set calculators ##########
     
    def calculate_massdata2D_level0(self, free_bgfield,
                          N, Xrange, sorting=True):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = [0 for i in range(N)]
        for i,x in enumerate(X):
            M0_numerical = self.M0.subs({free_bgfield: x})
            M0_numerical = np.array(M0_numerical).astype(np.float64)
            masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting)   
            masses_new = np.zeros(8)
            for j, m in enumerate(masses):
                masses_new[j] = np.sign(m)*np.sqrt(np.abs(m))
            Y[i] = masses_new
        Y = np.transpose(Y)

        return X, Y   
    
    def calculate_massdata2D_level1(self, model, free_bgfield,
                          N, Xrange,
                          calc_potential=True, calc_mass=True, sorting=True):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = [0 for i in range(N)]
        for i,x in enumerate(X):
            subs_bgfield_values = {free_bgfield: x}
            CWcalculator = CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values)
            MCW = CWcalculator.second_derivative()
            MCT = self.MCT.subs(subs_bgfield_values).evalf()
            M0 = self.M0.subs(subs_bgfield_values).evalf()
            Meff = M0 + MCT + MCW
            masses, field_states, R = diagonalize_numerical_matrix(Meff, sorting=sorting)
            masses_new = np.zeros(8)
            for j, m in enumerate(masses):
                masses_new[j] = np.sign(m)*np.sqrt(np.abs(m))
            Y[i] = masses_new
        Y = np.transpose(Y)
        
        return X, Y
    
          
    def calculate_massdata3D_level0(model, free_bgfield,
                          N, Xrange,
                          calc_potential=True, calc_mass=True, sorting=True):
        pass
    
    def calculate_massdata3D_level1(self, model):
        pass
    
    
    ########## Potential Data set calculators ##########
     
    def calculate_Vdata2D_level0(self, N, free_bgfield, Xrange):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = [0 for i in range(N)]
        
        for i,x in enumerate(X):
            V0_numerical = self.V0_simplified.subs({free_bgfield: x})
            Y[i] = V0_numerical
        
        return X, Y
    
    def calculate_Vdata3D_level0(self, N, free_bgfields, Xranges):
        Xrange = Xranges[0]
        Yrange = Xranges[1]
        omega1 = free_bgfields[0]
        omega2 = free_bgfields[1]
        # Define grid points
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.linspace(Yrange[0], Yrange[1], N)
        X, Y = np.meshgrid(X, Y)
        
        Z = np.zeros((N,N))
        # Iterate over grid points
        for i in range(N):
            for j in range(N):
                x, y = X[i, j], Y[i, j]
                V0_numerical = self.V0_simplified.subs({omega1: x, omega2: y})
                Z[i,j] = V0_numerical
                
        return X, Y, Z
    
    def calculate_Vdata2D_level1(self, model):
        pass
    
    def calculate_Vdata3D_level1(self, model):
        pass

####################################################################################
# Function/Method classes
####################################################################################

# Class for calculating the Coleman-Weinberg potential derivatives
class CWPotentialDerivativeCalculator:
    """ Calculates the Coleman-Weinberg potential derivatives for a given model 
    and a specific background field value, given at the initialization of the object."""
    
    # Constants
    threshold = 1e-10  
    epsilon = 1/(4*np.pi)**2
    
    def __init__(self, MDC:ModelDataCalculator, model:Model2HDM, subs_bgfield_values:dict):
        
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

        # Fields
        self.massStates_higgs = sp.Matrix(self.R_higgs) * sp.Matrix(model.massfields)
        self.massStates_gauge = sp.Matrix(self.R_gauge) * sp.Matrix(model.massfields_gauge)
        #self.massStates_fermions = sp.Matrix(self.R_fermions) * sp.Matrix(model.massfields_fermions)

        # Potentials
        self.V0 = MDC.V0_simplified.subs(subs_bgfield_values).evalf().subs({field:state for field,state in zip(model.massfields, self.massStates_higgs)})
        self.L_kin = MDC.L_kin_simplified.subs(subs_bgfield_values).evalf().subs({field:state for field,state in zip(model.massfields_gauge, self.massStates_gauge)})
        #self.L_yuk = MDC.L_yuk_simplified.subs(subs_bgfield_values).evalf().subs({field:state for field,state in zip(model.massfields_fermions, self.massStates_fermions)})
        
        # Couplings
        self.L3_higgs = self.calculate_couplings3(self.V0, model.massfields, model.massfields)
        self.L3_gauge = self.calculate_couplings3(self.L_kin, model.fields_gauge, model.massfields)
        #self.L3_fermions = self.calculate_couplings3(self.L_yuk, model.fields_fermions, model.massfields)
        
        self.L4_higgs = self.calculate_couplings4(self.V0, model.massfields, model.massfields)
        self.L4_gauge = self.calculate_couplings4(self.L_kin, model.fields_gauge, model.massfields)
        #self.L4_fermions = self.calculate_couplings4()

    # Calculate trilinear couplings
    def calculate_couplings3(self, V, fields1, fields2):
        
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
    def calculate_couplings4(self, V, fields1, fields2):

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
    def f1(self, msq, mu):

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
    def f2(self, m1sq, m2sq, mu):
        
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

    def Ni(self, L3, masses, kT, coeff):
        N = 8
        n = len(masses)
        Ncw = np.zeros(N) 
        for j in range(N):
            for a in range(n):
                Ncw[j] += L3[a][a][j] * masses[a] * (self.f1(masses[a],self.mu)-kT+1/2) #M[a] * (sp.log(M[a]/mu**2)-kT+1/2)

        return coeff * Ncw
        pass

    def Hij(self, L3, L4, masses, kT, coeff):
        N = 8
        n = len(masses)
        Mcw = np.zeros((N,N))
        for i in range(N):
            for j in range(N):
                if j >= i: # use symmetry
                    for a in range(n):
                        for b in range(n):
                            logf2 = self.f2(masses[a],masses[b],self.mu)
                            Mcw[i,j] += L3[a][b][i] * L3[b][a][j] * (logf2-kT+1/2) 
                            if masses[a] != 0:
                                logf1 = f1(masses[a],self.mu)
                                Mcw[i,j] += L4[a][a][i][j] * masses[a] * (logf1-kT+1/2)
        return coeff * Mcw  
    

    def first_derivative(self):
        # First derivative terms
        NCW_higgs = self.Ni(L3=self.L3_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        NCW_gauge = self.Ni(L3=self.L3_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        NCW_fermions = np.zeros(8) #Ni(L3, masses, kT, coeff)
        NCW = NCW_higgs + NCW_gauge + NCW_fermions
        NCW = self.epsilon * self.R_higgs @ NCW # numpy matrix mul (j to i)  #sp.Matrix(NCW)
        NCW = np.where(np.abs(NCW) < self.threshold, 0, NCW)
        return sp.Matrix(NCW)
    
    def second_derivative(self):
        MCW_higgs = self.Hij(L3=self.L3_higgs, L4=self.L4_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        MCW_gauge = self.Hij(L3=self.L3_gauge, L4=self.L4_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        MCW_fermions = np.zeros((8,8)) # Hij(L3, L4, masses, kT, coeff)
        MCW = MCW_higgs + MCW_gauge + MCW_fermions
        MCW = self.epsilon * self.R_higgs.T @ (MCW+MCW.T)/2 @ self.R_higgs
        MCW = np.where(np.abs(MCW) < self.threshold, 0, MCW)

        return sp.Matrix(MCW)
        
# Class for calculating the Branching ratios
class BranchingRatios:
    pass