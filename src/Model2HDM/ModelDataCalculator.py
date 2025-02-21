# Imports
import os
import sympy as sp  
import symengine as se
import numpy as np  
import matplotlib.pyplot as plt  
import time as time
from IPython.display import display, Math  

# Utils
from numba import jit, njit
from functools import lru_cache
import psutil

# Custom imports
from .class_Model2HDM import Model2HDM
from .methods_Model2HDM import *
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from src.utils import constants as const

####################################################################################

# Main/parent class
class ModelDataCalculator:
    """
    Generates numerical solutions for a given set of parameters and VEVs, which depends on omega.
        
    * Does not modify the model itself, thus multiple calculators can be created for different parameter sets.
    """
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
        self.subs_bgfields_to_VEV_values = {symb:val for symb,val in zip(model.bgfields, subs_VEV_values.values())}
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
        #sp.hessian(self.V0_simplified, model.fields).subs(self.subs_fields_to_zero)
        self.M0 = model.getdata("M0").subs(self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.M_kin = sp.hessian(self.L_kin_simplified, model.fields_gauge).subs(self.subs_fields_to_zero) #.subs(self.subs_constants).evalf()
        self.M_yuk = None

        # Counterterm values
        self.MCT = None


    def assign_counterterm_values(self):
        # Calculate the VCT values
        VCT_params_values = self.counterterm_values()
        self.subs_VCT_params_values = {param:value for param, value in zip(self.model.VCT_params, VCT_params_values)}
        self.VCT_simplified = self.model.VCT.subs(self.subs_VCT_params_values | self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.MCT = sp.hessian(self.VCT_simplified, self.model.fields).subs(self.subs_fields_to_zero)


    ########## Numerical Solutions ##########
    
    def calculate_masses_higgs(self, subs_bgfield_values=None):
        if subs_bgfield_values is None:
            subs_bgfield_values = self.subs_bgfields_to_VEV_values
        M_numerical = self.M0.subs(subs_bgfield_values).evalf()
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_gauge(self, subs_bgfield_values):
        M_numerical = self.M_kin.subs(subs_bgfield_values).evalf()
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_fermions(self, subs_bgfield_values):
        pass  
    
    ########## Numerical Solutions ##########
    
    

    def calculate_VCW_point(self, subs_bgfield_values, only_top=True):
        
        
        # Constants
        v = const.v
        mu = v
        
        # Coleman weinberg potential function
        def V_CW(mass, dof, C):

            mass_squared = mass
            # Handle zero and negative masses
            if mass_squared < 1e-5: #.is_zero:
                log_term = 0
            else:
                #log_term = sp.sign(mass_squared)*sp.log(sp.Abs(mass_squared) / mu**2)
                log_term = sp.log(mass_squared / mu**2)

            factor = dof / (64 * sp.pi**2)

            # Add this particle's contribution to the effective potential
            return factor * mass_squared**2 * (log_term - C)
        
        
        masses_higgs, _ = self.calculate_masses_higgs(subs_bgfield_values)
        masses_gauge, _ = self.calculate_masses_gauge(subs_bgfield_values)
        #masses_fermions = mdc.calculate_masses_fermions(subs_bgfield_values)
        
        # mass eigenvalues
        masses = list(np.concatenate((masses_higgs,masses_gauge))) #+ masses_fermions
        
        # Spins
        spins_higgs = [0 for _ in masses_higgs]
        spins_gauge = [1 for _ in masses_gauge]
        #spins_fermions = [1/2 for _ in masses_fermions]

        # Dof
        dofs_higgs = [1 for _ in masses_higgs] 
        dofs_gauge = []

        for mass in masses_gauge:
            if mass==0:
                dofs_gauge.append(2)
            else:
                dofs_gauge.append(3)
                
        dofs_fermions = [12]
        
        # Scheme constants
        scheme_constants_higgs = [3/2 for _ in masses_higgs]
        scheme_constants_gauge = [3/2 for _ in masses_gauge]
        #scheme_constants_fermions = [5/6 for _ in masses_fermions]

        # Generate the potential
        #Masses = masses_higgs + masses_gauge #+ masses_fermions
        Dofs = dofs_higgs + dofs_gauge #+ dofs_fermions
        Scheme_constants = scheme_constants_higgs + scheme_constants_gauge #+ scheme_constants_fermions
        V_cw_expr = 0
        
        for masses, dofs, scheme_constants in zip(masses, Dofs, Scheme_constants):
            V_cw_expr += V_CW(masses, dofs, scheme_constants)
        
        return V_cw_expr
    
    
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
    
    
    def VCW_first_derivative(self, subs_bgfield_values, regulator=246**2):
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values, regulator).first_derivative()

    def VCW_second_derivative(self, subs_bgfield_values, regulator=246**2): 
    
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values, regulator).second_derivative()
    
    ########## Mass Data set calculators ##########
     
    
    def calculate_massdata2D_level0(self, N, Xrange, free_bgfield, sorting=True):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros((N,8))
        #Y = [0 for i in range(N)]
        
        #Test
        #M0_numerical = self.M0.subs({free_bgfield: 246})
        #M0_numerical = np.array(M0_numerical).astype(np.float64)
        #masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting)   
        #display(self.M0, sp.Matrix(M0_numerical))
        #display(masses)
        
        
        for i,x in enumerate(X):
            M0_numerical = self.M0.subs({free_bgfield: x})
            M0_numerical = np.array(M0_numerical).astype(np.float64)
            masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting)   
            #masses_new = np.zeros(8)
            for j, m in enumerate(masses):
                Y[i,j] = np.sign(m)*np.sqrt(np.abs(m))
            #Y[i] = masses_new
        Y = np.transpose(Y)

        return X, Y   
    
    def calculate_massdata2D_level1(self, N, Xrange, free_bgfield, sorting=True):
        import gc
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros((N,8))
        #Y = [0 for i in range(N)]
        
        # Precompile functions
        self.assign_counterterm_values()
        MCT_func = sp.lambdify(free_bgfield, self.MCT, "numpy")
        M0_func  = sp.lambdify(free_bgfield, self.M0, "numpy")
        Meff = np.zeros((8, 8), dtype=np.float64)
        #gc.collect()
        gc.disable()
        
        t0 = time.time()
        for i,x in enumerate(X):
            t_last = time.time()
            # Calculations
            #subs_bgfield_values = {free_bgfield: x}
            #CWcalculator = CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values)
            #tt0 = time.time()
            MCW = self.VCW_second_derivative({free_bgfield: x}) #CWcalculator.second_derivative()
            #tt1 = time.time()
            #print("Time for MCW: ", tt1-tt0)
            #del CWcalculator
            #MCT = self.MCT.subs(subs_bgfield_values).evalf()
            #M0 = self.M0.subs(subs_bgfield_values).evalf()
            #M0 = np.array(M0_func(x)).astype(np.float64)
            #MCT = np.array(MCT_func(x)).astype(np.float64)
            M0 = M0_func(x)
            MCT = MCT_func(x)
            np.add(M0, MCT, out=Meff)  # Meff = M0 + MCT
            np.add(Meff, MCW, out=Meff)  # Meff += MCW
            
            #Meff = M0 + MCT + MCW
            masses, field_states, R = diagonalize_numerical_matrix(Meff, sorting=sorting)
            #masses_new = np.zeros(8)
            
            #np.sqrt(np.abs(masses), out=Y[i])  # Directly modify Y
            #Y[i] *= np.sign(masses)  # Apply sign directly
            for j, m in enumerate(masses):
                Y[i,j] = np.sign(m)*np.sqrt(np.abs(m))
                #Y[i,j] = masses_new
            
            # Print progress
            ti = time.time()
            t = ti-t0
            t_avg = t/(i+1)
            t_now = ti - t_last
            print(f"| Progress: {i+1}/{N} at x={x:0.1f}={x/246:0.2f}v | Estimated time left: {sec_to_hms(t_avg*(N-i-1))} | Elapsed time: {sec_to_hms(t)} // {sec_to_hms(t_avg*N)} | Time per point: {t_avg:0.3f}s | " , end="\r")
            #print("Progress: ", i+1, "/", N, " at x=", x, "=", x/246, "v | Estimated time left: ", sec_to_hms(t_avg*(N-i-1)), " | Elapsed time: ", sec_to_hms(t), " // ", sec_to_hms(t_avg*N), " | Time for point: ", t_now, "s , avg: ", t_avg, "s | " , end="\r")
            if i % 10 == 0:
                gc.collect()
            
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
        Y = np.zeros(N)
        
        for i,x in enumerate(X):
            V0_numerical = self.V0_simplified.subs({free_bgfield: x}).subs(self.subs_fields_to_zero).evalf()
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
    
    def calculate_Vdata2D_level1(self, N, free_bgfield, Xrange):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros(N)
        
        self.assign_counterterm_values()
        
        for i,x in enumerate(X):
            print(f"Progress {i+1}/{N} at x=", x, end="\r")
            V0_point = self.V0_simplified.subs({free_bgfield: x}).subs(self.subs_fields_to_zero).evalf()
            Vcw_point = self.calculate_VCW_point({free_bgfield: x}, only_top=True)
            Vct_point = self.VCT_simplified.subs({free_bgfield: x}).subs(self.subs_VCT_params_values).subs(self.subs_fields_to_zero).evalf()
            Y[i] = V0_point + Vcw_point + Vct_point
        
        return X, Y
    
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
    
    def __init__(self, MDC:ModelDataCalculator, model:Model2HDM, subs_bgfield_values:dict, regulator=246**2): #
        
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
        #display(self.masses_higgs)
        self.regulator = regulator
        self.masses_higgs = self.masses_higgs + self.regulator
        #display(self.masses_higgs)

        # Fields
        #self.massStates_higgs = sp.Matrix(self.R_higgs) * sp.Matrix(model.massfields)
        #self.massStates_gauge = sp.Matrix(self.R_gauge) * sp.Matrix(model.massfields_gauge)
        self.massStates_higgs = self.R_higgs @ model.massfields
        self.massStates_gauge = self.R_gauge @ model.massfields_gauge
        #self.massStates_fermions = sp.Matrix(self.R_fermions) * sp.Matrix(model.massfields_fermions)

        # Potentials (This is a time sink)
        subs_massStates_higgs = {field:state for field,state in zip(model.fields, self.massStates_higgs)}
        subs_massStates_gauge = {field:state for field,state in zip(model.fields_gauge, self.massStates_gauge)}
        #self.V0 = MDC.V0_simplified.subs(subs_bgfield_values | subs_massStates_higgs).expand() #).evalf().subs(
        #self.L_kin = MDC.L_kin_simplified.subs(subs_bgfield_values | subs_massStates_gauge).expand()
        #self.L_yuk = MDC.L_yuk_simplified.subs(subs_bgfield_values).evalf().subs({field:state for field,state in zip(model.massfields_fermions, self.massStates_fermions)})
        #self.V0 = MDC.V0_simplified.xreplace(subs_bgfield_values | subs_massStates_higgs).expand() 
        
        
        #self.V0 = MDC.V0_simplified.xreplace(subs_bgfield_values) # ca 0.2s
        #self.V0 = custom_fast_expand_ultra(self.V0.xreplace(subs_massStates_higgs), model.massfields) # This takes the logest (ca 0.8-1s)
        
        #t0 = time.time()
        V0_se = se.sympify(MDC.V0_simplified).xreplace(subs_bgfield_values) 
        V0_se = V0_se.subs(subs_massStates_higgs).expand()
        self.V0 = V0_se #sp.sympify(V0_se)
        #self.V0 = V0_se
        #print("Time for V0: ", time.time()-t0)
        #display(self.V0)
        #display(V0_se.as_coefficients_dict())
        
        self.L_kin = MDC.L_kin_simplified.xreplace(subs_bgfield_values | subs_massStates_gauge).expand()

        # Couplings
        self.L3_higgs = self.calculate_couplings3(self.V0, model.massfields, model.massfields)
        self.L3_gauge = self.calculate_couplings3(self.L_kin, model.fields_gauge, model.massfields)
        #self.L3_fermions = self.calculate_couplings3(self.L_yuk, model.fields_fermions, model.massfields)
        
        self.L4_higgs = self.calculate_couplings4(self.V0, model.massfields, model.massfields) # ca 0.15s
        self.L4_gauge = self.calculate_couplings4(self.L_kin, model.fields_gauge, model.massfields)
        #self.L4_fermions = self.calculate_couplings4()
        
        #print("Debug")
        #display(self.L4_higgs[0,0,0])
        #display(sp.Matrix(self.R_higgs))
        #display(self.masses_higgs)

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
    def f1(self, msq, mu):
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
            msq += self.regulator
            result = sign*np.log(msq/mu**2) #m * (sp.log(m/mu**2)-kT+1/2)
        return result

    # log term 2, with regulator f2
    def f2(self, m1sq, m2sq, mu):
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
        m1sq_R = m1sq + self.regulator
        m2sq_R = m2sq + self.regulator
        
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
    
    # Calculate trilinear couplings (Optimized)
    #@jit
    def calculate_couplings3(self, V, fields1, fields2):
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
    def calculate_couplings4(self, V, fields1, fields2):
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
    
    #@jit
    def Ni(self, L3, masses, kT, coeff):
        """
        Compute the quantity Ncw[j] = sum_a L3[a,a,j] * masses[a] * (f1(masses[a], mu) - kT + 0.5)
        and return coeff * Ncw.
        
        L3 is assumed to be a NumPy array of shape (n, n, 8), and masses is a list or array.
        """
        masses = np.array(masses)  # ensure it's a NumPy array
        n = len(masses)
        N = 8
        
        # Precompute f1 for each mass
        f1_vals = np.array([self.f1(m, self.mu) for m in masses])
        
        # Extract the diagonal elements L3[a,a,:] for each a (resulting in an array of shape (n, N))
        # One could also use np.einsum('aaj->aj', L3) if L3 is a full array.
        L3_diag = np.array([L3[a, a, :] for a in range(n)])
        # Compute contributions: for each a, shape (n, N)
        contributions = L3_diag * masses[:, None] * (f1_vals - kT + 0.5)[:, None]
        
        # Sum over a to get an array of shape (N,)
        Ncw = contributions.sum(axis=0)
        return coeff * Ncw

    #@jit
    def Hij(self, L3, L4, masses, kT, coeff):
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
                F2[a, b] = self.f2(masses[a], masses[b], self.mu)
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
        f1_vals = np.array([self.f1(m, self.mu) if m != 0 else 0 for m in masses])
        
        # Extract the diagonal of L4: L4_diag[a,:,:] = L4[a,a,:,:] (shape: (n, N, N))
        L4_diag = np.array([L4[a, a, :, :] for a in range(n)])
        
        # term2[i,j] = sum_{a} L4[a,a,i,j] * masses[a] * (f1_vals[a] - kT + 0.5)
        term2 = np.einsum('aij,a->ij', L4_diag, masses * (f1_vals - kT + 0.5))
        
        # -----------------------------
        # Combine both contributions
        Mcw = term1 + term2
        
        
        return coeff * Mcw

    ####### "Raw" unoptimized helper functions #######

    # log term 2, IR divergent
    def f2_IRDIV(self, m1sq, m2sq, mu):
        
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

    # Calculate trilinear couplings
    def calculate_couplings3_unoptimized(self, V, fields1, fields2):
        
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
    def calculate_couplings4_unoptimized(self, V, fields1, fields2):

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

    def Ni_unoptimized(self, L3, masses, kT, coeff):
        N = 8
        n = len(masses)
        Ncw = np.zeros(N) 
        for j in range(N):
            for a in range(n):
                Ncw[j] += L3[a][a][j] * masses[a] * (self.f1(masses[a],self.mu)-kT+1/2) #M[a] * (sp.log(M[a]/mu**2)-kT+1/2)

        return coeff * Ncw
        pass

    def Hij_unoptimized(self, L3, L4, masses, kT, coeff):
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
                                logf1 = self.f1(masses[a],self.mu)
                                Mcw[i,j] += L4[a][a][i][j] * masses[a] * (logf1-kT+1/2)
        return coeff * Mcw  
    

# Class for calculating the Branching ratios
class BranchingRatios:
    pass





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
    #masses_gauge, _ = generate_masses_gaugebosons(model)
    #masses_fermions, _ = generate_masses_fermions(model, type="I", only_top=only_top)
    
    
    #display(masses_gauge, masses_fermions)
    
    
    
    # Unpacking
    #masses_higgs = model.DATA["M0_eigenvalues"]
    
    masses_higgs = mdc.calculate_masses_higgs(subs_bgfield_values)
    masses_gauge = mdc.calculate_masses_gauge(subs_bgfield_values)
    #masses_fermions = mdc.calculate_masses_fermions(subs_bgfield_values)
    
    # mass eigenvalues
    masses = masses_higgs + masses_gauge + masses_fermions
    
    # Spins
    spins_higgs = [0 for _ in masses_higgs]
    spins_gauge = [1 for _ in masses_gauge]
    #spins_fermions = [1/2 for _ in masses_fermions]

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
    #scheme_constants_fermions = [5/6 for _ in masses_fermions]

    # Generate the potential
    Masses = masses_higgs + masses_gauge #+ masses_fermions
    Dofs = dofs_higgs + dofs_gauge #+ dofs_fermions
    Scheme_constants = scheme_constants_higgs + scheme_constants_gauge #+ scheme_constants_fermions
    V_cw_expr = 0
    
    for masses, dofs, scheme_constants in zip(Masses, Dofs, Scheme_constants):
        V_cw_expr += V_CW(masses, dofs, scheme_constants)
    
    return V_cw_expr



############# Helper functions #############


####################################################
def poly_to_dict(expr, gens):
    """
    Convert a polynomial expression in generators `gens` into a dictionary.
    Each key is a tuple of exponents (in the order of gens) and the value is the coefficient.
    """
    # Base case: if expr is an addition, process each term.
    if expr.is_Add:
        poly_dict = {}
        for term in expr.args:
            term_dict = poly_to_dict(term, gens)
            for mono, coeff in term_dict.items():
                poly_dict[mono] = poly_dict.get(mono, 0) + coeff
        return poly_dict
    # Multiplication: multiply the dictionary representations.
    elif expr.is_Mul:
        poly_dict = {tuple([0]*len(gens)): 1}  # represents the constant 1
        for factor in expr.args:
            factor_dict = poly_to_dict(factor, gens)
            poly_dict = multiply_poly_dict(poly_dict, factor_dict)
        return poly_dict
    # Power: handle integer exponents.
    elif expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        base_dict = poly_to_dict(expr.base, gens)
        return poly_pow_dict(base_dict, int(expr.exp), len(gens))
    else:
        # If expr is one of the generators, represent it with exponent 1.
        if expr in gens:
            mono = [0] * len(gens)
            mono[gens.index(expr)] = 1
            return {tuple(mono): 1}
        # Otherwise, assume it's a numeric constant.
        return {tuple([0]*len(gens)): expr}

def multiply_poly_dict(p1, p2):
    """
    Multiply two polynomials represented as dictionaries.
    """
    result = {}
    for m1, c1 in p1.items():
        for m2, c2 in p2.items():
            # Sum the exponents elementwise.
            m = tuple(a + b for a, b in zip(m1, m2))
            result[m] = result.get(m, 0) + c1 * c2
    return result

def poly_pow_dict(poly, n, num_gens):
    """
    Raise a polynomial (in dict form) to the power n.
    """
    result = {tuple([0]*num_gens): 1}  # 1
    for _ in range(n):
        result = multiply_poly_dict(result, poly)
    return result

def dict_to_expr(poly_dict, gens):
    """
    Convert a dictionary representation back into a Sympy expression.
    """
    expr = 0
    for mono, coeff in poly_dict.items():
        term = coeff
        for g, exp in zip(gens, mono):
            if exp:
                term *= g**exp
        expr += term
    return expr

def custom_fast_expand_ultra(expr, gens):
    """
    Fully expand a polynomial expression by converting it to a dictionary and back.
    Assumes the expression is a polynomial in the given generators.
    """
    poly_dict = poly_to_dict(expr, gens)
    return dict_to_expr(poly_dict, gens)


##############################################################################
def custom_fast_expand(expr):
    """
    Recursively expand a polynomial expression by distributing products.
    Assumes that powers are nonnegative integers.
    """
    # Base case: if the expression is atomic, return it.
    if expr.is_Atom:
        return expr

    # If it's an addition, recursively expand each term.
    if expr.is_Add:
        return sp.Add(*[custom_fast_expand(arg) for arg in expr.args])
    
    # If it's a multiplication, recursively expand each factor
    # then distribute pairwise.
    if expr.is_Mul:
        # First, expand each factor
        factors = [custom_fast_expand(arg) for arg in expr.args]
        result = factors[0]
        for factor in factors[1:]:
            result = distribute(result, factor)
        return result
    
    # If it's a power with an integer exponent, use repeated multiplication.
    if expr.is_Pow and expr.exp.is_Integer and expr.exp >= 0:
        base = custom_fast_expand(expr.base)
        n = int(expr.exp)
        if n == 0:
            return sp.Integer(1)
        result = base
        for _ in range(1, n):
            result = distribute(result, base)
        return result

    # Otherwise, process the arguments and reassemble.
    return expr.func(*[custom_fast_expand(arg) for arg in expr.args])

def distribute(expr1, expr2):
    """
    Distribute multiplication over addition for two expanded expressions.
    Assumes expr1 and expr2 are already expanded (or atomic).
    """
    # If an expression is not an Add, treat it as a single term.
    terms1 = expr1.args if expr1.is_Add else [expr1]
    terms2 = expr2.args if expr2.is_Add else [expr2]
    return sp.Add(*[t1 * t2 for t1 in terms1 for t2 in terms2])