# Imports
import os
import sympy as sp  
import symengine as se
import numpy as np  
import matplotlib.pyplot as plt  
import time as time
from IPython.display import display, Math  
import warnings

# Utils
from numba import jit, njit
from functools import lru_cache
import psutil

# Custom imports
from ..Model2HDM.Model2HDM import Model2HDM
from ..Model2HDM.methods_Model2HDM import *

# Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *
from ..utils import constants as const

# Methods
from .derivativesVCW import CWPotentialDerivativeCalculator
from .couplings import calculate_couplings3, calculate_couplings4
from .thermal_functions import V_T, V_daisy, daisy_mass_corrections_higgs, daisy_mass_corrections_gauge

####################################################################################

# Main/parent class
class Model2HDMCalculator:
    """
    Generates numerical solutions for a given set of parameters and VEVs, which depends on omega.
        
    * Does not modify the model itself, thus multiple calculators can be created for different parameter sets.
    """
    def __init__(self, model:Model2HDM, 
                 subs_V0_params_values:dict=None, 
                 subs_VEV_values:dict=None, 
                 V0_params_values:list=None,
                 VEV_values:list=None):
        
        self.is_renormalized = None
        self.model = model
        assert model.getdata("VCT_params_solution") is not None, "The analytical counterterm solutions has not been solved yet"
        
        # Numerical values for parameters
        if V0_params_values is None and subs_V0_params_values is None:
            raise ValueError("No values for V0 parameters has been assigned")
        elif V0_params_values is not None and subs_V0_params_values is not None:
            raise ValueError("Both list and dict values for V0 parameters has been assigned")
        elif V0_params_values is not None:
            subs_V0_params_values = {param:value for param, value in zip(model.V0_params, V0_params_values)}
            self.subs_V0_params_values = subs_V0_params_values
        elif subs_V0_params_values is not None:
            self.subs_V0_params_values = subs_V0_params_values
            
        # Numerical values for VEVs
        if VEV_values is None and subs_VEV_values is None:
            raise ValueError("No values for VEVs has been assigned")
        elif VEV_values is not None and subs_VEV_values is not None:
            raise ValueError("Both list and dict values for VEVs has been assigned")
        elif VEV_values is not None:
            subs_VEV_values = {symb:val for symb,val in zip(model.VEVs, VEV_values)}
            self.subs_VEV_values = subs_VEV_values
        elif subs_VEV_values is not None:
            self.subs_VEV_values = subs_VEV_values
        
        #self.subs_VEV_values = subs_VEV_values
        #self.subs_bgfield_values = subs_VEV_values #Default values for unassigned bgfields
        #self.subs_V0_params_values = subs_V0_params_values # add support for both dict and list
        
        # Precompute common substitutions and expressions.
        #self.fields_to_zero = {f: 0 for f in model.fields} | {f: 0 for f in model.fields_gauge}
        #self.subs_VEV_values = {symb:val for symb,val in zip(model.VEVs, VEV_values)}
        self.subs_bgfields_to_VEV = {symb:val for symb,val in zip(model.bgfields, model.VEVs)}
        self.subs_bgfields_to_VEV_values = {symb:val for symb,val in zip(model.bgfields, VEV_values)}
        self.subs_fields_to_zero = {field:0 for field in model.fields}
        self.bgfields_to_zero = {bgfield:0 for bgfield in model.bgfields}
        #self.subs_V0_params_values = {param:value for param, value in zip(model.V0_params, V0_params_values)}
        
        # Gauge subs
        g1 = model.symbol("g1")
        g2 = model.symbol("g2")
        self.subs_constants_gauge = {g1:const.g1, g2:const.g2}

        # quark subs
        #yt = model.symbol("yt")

        # Simplified solutions
        self.V0_simplified = model.V0.subs(self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.L_kin_simplified = generate_kinetic_term_gauge(model).subs(self.subs_V0_params_values | self.subs_VEV_values | self.subs_constants_gauge).evalf()
        self.L_yuk_simplified = generate_yukawa_term_fermions(model).subs(self.subs_V0_params_values | self.subs_VEV_values | self.subs_constants_gauge).evalf()
        
        # Precompute expressions
        #sp.hessian(self.V0_simplified, model.fields).subs(self.subs_fields_to_zero)
        self.M0 = model.getdata("M0").subs(self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.M_kin = sp.hessian(self.L_kin_simplified, model.fields_gauge).subs(self.subs_fields_to_zero) #.subs(self.subs_constants).evalf()
        self.M_yuk = None

        # Counterterm values
        self.MCT = None
        self.is_renormalized = None

    ########## Methods for renormalization and counterterms ##########

    def assign_counterterm_values(self, raise_warning=True):
        # Calculate the VCT values
        VCT_params_values = self.counterterm_values()
        self.VCT_params_values = VCT_params_values
        self.subs_VCT_params_values = {param:value for param, value in zip(self.model.VCT_params, VCT_params_values)}
        self.VCT_simplified = self.model.VCT.subs(self.subs_VCT_params_values | self.subs_V0_params_values | self.subs_VEV_values).evalf()
        self.MCT = sp.hessian(self.VCT_simplified, self.model.fields).subs(self.subs_fields_to_zero)

        # Also make a check if the model is renormalized
        if self.is_renormalized is None:
            self.check_renormalization()
        if not self.is_renormalized and raise_warning:
            warnings.warn("The model is not renormalized", UserWarning)

    def check_renormalization(self, debug=False):
        if self.is_renormalized is None:
            MCW = self.VCW_second_derivative(self.subs_bgfields_to_VEV_values)
            # Check so that the renormalization works
            diff = self.MCT.subs(self.subs_bgfields_to_VEV_values).evalf() + sp.Matrix(MCW)
            threshold = 1e-5
            is_renormalized = all(element < threshold for element in diff)
            self.is_renormalized = is_renormalized
        if not self.is_renormalized and debug:
            print("The model is not renormalized!")
            print("See, difference: MCT + MCW =")
            display(sp.Matrix(diff))
        return self.is_renormalized
    
    
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
        subs_bgfield_values = {bgfield:sp.sympify(VEV_value).subs(self.subs_VEV_values) for bgfield, VEV_value in zip(self.model.bgfields, self.model.VEVs) if bgfield !=0}

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
    

    ########## Numerical Solutions for masses ##########
    
    def calculate_masses_higgs(self, subs_bgfield_values=None, T=0):
        
        # Default to VEV values
        if subs_bgfield_values is None:
            subs_bgfield_values = self.subs_bgfields_to_VEV_values
        
        # Convert to numerical matrix
        M_numerical = sp.re(self.M0.subs(subs_bgfield_values)).evalf()
        try:
            M_numerical = np.array(M_numerical).astype(np.float64)
        except Exception as e:
            print("Error:", e)
            display(M_numerical)
            raise TypeError("Error in converting to numerical matrix")
        
        
        # Calculate the mass corrections for non-zero temperature
        if T != 0:
            
            V0_se = se.sympify(self.V0_simplified).xreplace(subs_bgfield_values) 
            V0_se = V0_se.expand() #.subs(subs_massStates_higgs).expand()
            V0_se = se.sympify(sp.sympify(V0_se).xreplace({sp.I:0}))
            
            L_kin_se = se.sympify(self.L_kin_simplified).xreplace(subs_bgfield_values) # subs_massStates_higgs | subs_massStates_gauge)
            L_kin_se = L_kin_se.expand()#.evalf()
            L_kin_se = se.sympify(sp.sympify(L_kin_se).xreplace({sp.I:0}))
            
            # Calculate the mass corrections
            L4_higgs = calculate_couplings4(V0_se, self.model.fields, self.model.fields)
            L4_gauge = calculate_couplings4(L_kin_se, self.model.fields_gauge, self.model.fields)
            #L3_fermion = calculate_couplings3(self.L_yuk_simplified, self.model.fields, self.model.fields)
            
            M_T = daisy_mass_corrections_higgs(L4_higgs, L4_gauge, L3_fermion=0, T=T)
            M_numerical += M_T
            
        
        # Diagonalize the matrix
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_gauge(self, subs_bgfield_values=None, T=0, use_debye_mass=True):
        
        # Default to VEV values
        if subs_bgfield_values is None:
            subs_bgfield_values = self.subs_bgfields_to_VEV_values

        # Convert to numpy numerical matrix
        M_numerical = sp.re(self.M_kin.subs(subs_bgfield_values)).evalf()
        try:
            M_numerical = np.array(M_numerical).astype(np.float64)
        except Exception as e:
            print("Error:", e)
            display(M_numerical)
            raise TypeError("Error in converting to numerical matrix")
        
        # Calculate the mass corrections for non-zero temperature
        if T != 0:
            L_kin_se = se.sympify(self.L_kin_simplified).xreplace(subs_bgfield_values) # subs_massStates_higgs | subs_massStates_gauge)
            L_kin_se = L_kin_se.expand()#.evalf()
            L_kin_se = se.sympify(sp.sympify(L_kin_se).xreplace({sp.I:0}))
            
            # Calculate the mass corrections
            L4_gauge = calculate_couplings4(L_kin_se, self.model.fields_gauge, self.model.fields)
            
            M_T = daisy_mass_corrections_gauge(L4_gauge, T, use_debye_mass)
            M_numerical += M_T
        
        # Diagonalize the matrix
        masses, states, R = diagonalize_numerical_matrix(M_numerical, sorting=True)
        return masses, R
    
    def calculate_masses_fermions(self, subs_bgfield_values):
        pass  
    
    ########## Numerical Solutions for CW potential ##########
    
    def calculate_VCW_point(self, subs_bgfield_values, T=0, only_top=True):
        
        
        # Constants
        v = const.v
        mu = v
        
        # Coleman weinberg potential function
        def V_CW(mass, dof, spin, C):

            mass_squared = mass
            sign = 1
            if mass_squared < 0:
                sign = -1
                mass_squared = -mass_squared
            # Handle zero and negative masses
            if np.abs(mass_squared) < 1e-5: #.is_zero:
                log_term = 0
            else:
                #log_term = sp.sign(mass_squared)*sp.log(sp.Abs(mass_squared) / mu**2)
                log_term = sign*sp.log(mass_squared / mu**2)

            factor = dof / (64 * sp.pi**2) * (-1)**(2*spin)

            # Add this particle's contribution to the effective potential
            return factor * mass_squared**2 * (log_term - C)
        
        # Masses
        masses_higgs, _ = self.calculate_masses_higgs(subs_bgfield_values, T)
        masses_gauge, _ = self.calculate_masses_gauge(subs_bgfield_values, T)
        #masses_fermions = mdc.calculate_masses_fermions(subs_bgfield_values)
        
        ####################################
        # TESTING
        """
        new_masses_higgs = []
        for i, mass in enumerate(masses_higgs):
            if i in [0,4,5,6,7]: #[0,1,2,3,4,5,6,7]: # [1,2,3,6,7] # 0 is fine, [0,4,5]
                new_masses_higgs.append(mass)
            else:
                new_masses_higgs.append(0)
        masses_higgs = new_masses_higgs
        """
        #masses_higgs[0] = masses_higgs[0] * 0.5 #2/3
        #masses_higgs[1] = masses_higgs[1] * 0.5
        #masses_higgs[2] = masses_higgs[2] * 0.5 #2/3
        #masses_higgs[3] = masses_higgs[3] * 0.5 #2/3
        #masses_higgs = [mass*0.7 for mass in masses_higgs]
        ####################################
        
        #masses_higgs = [mass for i, mass in enumerate(masses_higgs) if i in [1,2,3,6,7]]  #/ 1.1 #ca
        masses_gauge = masses_gauge #/ 1.1 #ca
        
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

        # Collect the masses, spins, dofs and scheme constants
        masses = list(np.concatenate((masses_higgs,masses_gauge))) 
        Dofs = dofs_higgs + dofs_gauge #+ dofs_fermions
        Spins = spins_higgs + spins_gauge #+ spins_fermions
        Scheme_constants = scheme_constants_higgs + scheme_constants_gauge #+ scheme_constants_fermions
        
        # Generate the potential
        V_cw_expr = 0
        for masses, dofs, spins, scheme_constants in zip(masses, Dofs, Spins, Scheme_constants):
            V_cw_expr += V_CW(masses, dofs, spins, scheme_constants)
        
        return V_cw_expr
       
    def VCW_first_derivative(self, subs_bgfield_values, regulator=246**2, scale=246): # add mu/scale constatnt as input
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values, regulator, scale).first_derivative()

    def VCW_second_derivative(self, subs_bgfield_values, regulator=246**2, scale=246): 
    
        return CWPotentialDerivativeCalculator(self, self.model, subs_bgfield_values, regulator, scale).second_derivative()
    
    ########## Mass Data set calculators ##########
     
    
    def calculate_massdata2D_level0(self, N, Xrange, free_bgfield, T=0, 
                                    parametrization_bgfields=None, sorting=True):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros((N,8))
        #Y = [0 for i in range(N)]
        """
        if parametrization_bgfields is not None:
            #parametrization_bgfields
            M0 = self.M0.subs(parametrization_bgfields)
        M0 = self.M0
        """
        #Test
        #M0_numerical = self.M0.subs({free_bgfield: 246})
        #M0_numerical = np.array(M0_numerical).astype(np.float64)
        #masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting)   
        #display(self.M0, sp.Matrix(M0_numerical))
        #display(masses)
        
        #masses_gauge = self.calculate_masses_gauge({free_bgfield: x})[0]
        
        for i,x in enumerate(X):
            """
            M0_numerical = M0.subs({free_bgfield: x})
            M0_numerical = np.array(M0_numerical).astype(np.float64)
            masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting) 
            """
            masses = self.calculate_masses_higgs({free_bgfield: x}, T)[0]
              
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
            #np.add(M0, MCT, out=Meff)  # Meff = M0 + MCT
            #np.add(Meff, MCW, out=Meff)  # Meff += MCW
            
            Meff = M0 + MCT + MCW
            #display(sp.Matrix(MCT), sp.Matrix(MCW))
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
     
    def calculate_Vdata2D_level0(self, N, Xrange, free_bgfield=None, free_bg_fields_expr=None, parametrization=None, subs_bgfields_extra={}):
        

        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros(N)
        
        subs_bgfields_remaining = {}
        for bgfield in self.model.bgfields:
            if bgfield != free_bgfield or bgfield not in subs_bgfields_extra:
                subs_bgfields_remaining.update({bgfield:0})
        subs_bgfields_remaining.update(subs_bgfields_extra)
        
        t0 = time.time()
        for i,x in enumerate(X):
            t_last = time.time()
                    
            if free_bgfield != None:
                subs_bg_fields = {free_bgfield: x}
            elif free_bg_fields_expr != None and parametrization != None:
                subs_parametrization = {parametrization:x}
                subs_bg_fields = {self.model.bgfields[i]:sp.Add(free_bg_fields_expr[i]).subs(subs_parametrization) for i in range(8)}
            else:
                raise "Need to choose bgfield/(s)"
                
            V0_numerical = self.V0_simplified.subs(subs_bg_fields).subs(subs_bgfields_remaining|self.subs_fields_to_zero).evalf()
            
            Y[i] = V0_numerical

            # Print progress
            ti = time.time()
            t = ti-t0
            t_avg = t/(i+1)
            t_now = ti - t_last
            print(f"| Progress: {i+1}/{N} at x={x:0.1f}={x/246:0.2f}v | Estimated time left: {sec_to_hms(t_avg*(N-i-1))} | Elapsed time: {sec_to_hms(t)} // {sec_to_hms(t_avg*N)} | Time per point: {t_avg:0.3f}s | " , end="\r")
        
        return X, Y
    
    def calculate_Vdata3D_level0(self, N, Xrange, Yrange, free_bgfields=None, free_bg_fields_expr=None, parametrization=None, subs_bgfields_extra={}):

        #omega1 = free_bgfields[0]
        #omega2 = free_bgfields[1]
        
        # Define grid points
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.linspace(Yrange[0], Yrange[1], N)
        X, Y = np.meshgrid(X, Y)
        
        Z = np.zeros((N,N))
        # Iterate over grid points
        for i in range(N):
            print(f"Progress: {i+1}/{N}", end="\r")
            
            
            for j in range(N):
                subs_bgfields_remaining = {}
                # subs parametrized bgfields
                x, y = X[i, j], Y[i, j]
                if free_bgfields != None:
                    subs_bgfields = {free_bgfields[0]: x, free_bgfields[1]: y}
                    for bgfield in self.model.bgfields:
                        if bgfield not in free_bgfields or bgfield not in subs_bgfields_extra:
                            subs_bgfields_remaining.update({bgfield:0})
                elif free_bg_fields_expr != None and parametrization != None:
                    subs_parametrization = {parametrization[0]:x, parametrization[1]:y}
                    subs_bgfields = {self.model.bgfields[i]:sp.Add(free_bg_fields_expr[i]).subs(subs_parametrization) for i in range(8)}
                else:
                    raise "Need to choose bgfield/(s)"
                
                # subs remaining bgfields
                
                
                subs_bgfields_remaining.update(subs_bgfields_extra)
                
                V0_numerical = self.V0_simplified.subs(subs_bgfields).subs(subs_bgfields_remaining|self.subs_fields_to_zero).evalf()
                Z[i,j] = V0_numerical
                
        return X, Y, Z
    
    def calculate_Vdata2D_level1(self, N, Xrange, free_bgfield=None, free_bg_fields_expr=None, parametrization=None, T=0):
        
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.zeros(N)
        v = 246
        #bgfields_to_zero = self.bgfields_to_zero.copy()
        #bgfields_to_zero.pop(free_bgfield)
        bgfields_to_zero = self.bgfields_to_zero.copy()
        
        self.assign_counterterm_values()
        
        for i,x in enumerate(X):
            
            if free_bgfield != None:
                subs_bg_fields = {free_bgfield: x}

                bgfields_to_zero = {}
                for bgfield in self.model.bgfields:
                    if bgfield != free_bgfield:
                        bgfields_to_zero.update({bgfield:0})
            elif free_bg_fields_expr != None and parametrization != None:
                subs_parametrization = {parametrization:x}
                subs_bg_fields = {self.model.bgfields[i]:sp.Add(free_bg_fields_expr[i]).subs(subs_parametrization) for i in range(8)}
                
                bgfields_to_zero = {}
                for bgfield, bgfield_expr in zip(self.model.bgfields, free_bg_fields_expr):
                    if bgfield_expr == 0:
                        bgfields_to_zero.update({bgfield:0})
            else:
                raise "Need to choose bgfield/(s)"
            
            print(f"Progress {i+1}/{N} at x=", x, end="\r")
            V0_point = self.V0_simplified.subs(subs_bg_fields|bgfields_to_zero).subs(self.subs_fields_to_zero).evalf()
            Vcw_point = self.calculate_VCW_point(subs_bg_fields|bgfields_to_zero, T, only_top=True)
            Vct_point = self.VCT_simplified.subs(subs_bg_fields|bgfields_to_zero).subs(self.subs_VCT_params_values).subs(self.subs_fields_to_zero).evalf()
            if T != 0:
                masses_higgs = self.calculate_masses_higgs(subs_bg_fields|bgfields_to_zero)[0]
                masses_gauge = self.calculate_masses_gauge(subs_bg_fields|bgfields_to_zero)[0]
                #masses_higgs_T = self.calculate_masses_higgs({free_bgfield: x}, T)[0]
                masses_gauge_T = self.calculate_masses_gauge(subs_bg_fields|bgfields_to_zero, T)[0]
                masses_quarks = 0 #[0 for _ in range(8)]
                VT = V_T(masses_higgs, masses_gauge, masses_quarks, T)
                #VT += V_daisy(masses_gauge, masses_gauge_T, T)
            else:
                VT = 0
            
            Y[i] = V0_point + Vcw_point + Vct_point + VT /v # why v here?
        
        return X, Y
    
    def calculate_Vdata3D_level1(self, N, Xrange, Yrange, free_bgfields=None, free_bg_fields_expr=None, parametrization=None, T=0, subs_bgfields_extra={}):
        #omega1 = free_bgfields[0]
        #omega2 = free_bgfields[1]
        # Define grid points
        X = np.linspace(Xrange[0], Xrange[1], N)
        Y = np.linspace(Yrange[0], Yrange[1], N)
        X, Y = np.meshgrid(X, Y)
        
        Z = np.zeros((N,N))
        # Iterate over grid points
        for i in range(N):
            print(f"Progress: {i+1}/{N}", end="\r")
            for j in range(N):
                subs_bgfields_remaining = {}
                # subs parametrized bgfields
                x, y = X[i, j], Y[i, j]
                if free_bgfields != None:
                    subs_bgfields = {free_bgfields[0]: x, free_bgfields[1]: y}
                    for bgfield in self.model.bgfields:
                        if bgfield not in free_bgfields or bgfield not in subs_bgfields_extra:
                            subs_bgfields_remaining.update({bgfield:0})
                elif free_bg_fields_expr != None and parametrization != None:
                    subs_parametrization = {parametrization[0]:x, parametrization[1]:y}
                    subs_bgfields = {self.model.bgfields[i]:sp.Add(free_bg_fields_expr[i]).subs(subs_parametrization) for i in range(8)}
                else:
                    raise "Need to choose bgfield/(s)"
                
                # subs remaining bgfields
                subs_bgfields_remaining.update(subs_bgfields_extra)
                
                V0_point = self.V0_simplified.subs(subs_bgfields).subs(subs_bgfields_remaining|self.subs_fields_to_zero).evalf()
                Vcw_point = self.calculate_VCW_point(subs_bgfields_remaining.update(subs_bgfields), T, only_top=True)
                Vct_point = self.VCT_simplified.subs(subs_bgfields).subs(subs_bgfields_remaining|self.subs_VCT_params_values).subs(self.subs_fields_to_zero).evalf()
                
                if T != 0:
                    masses_higgs = self.calculate_masses_higgs(subs_bgfields|subs_bgfields_remaining)[0]
                    masses_gauge = self.calculate_masses_gauge(subs_bgfields|subs_bgfields_remaining)[0]
                    #masses_higgs_T = self.calculate_masses_higgs({free_bgfield: x}, T)[0]
                    masses_gauge_T = self.calculate_masses_gauge(subs_bgfields|subs_bgfields_remaining, T)[0]
                    masses_quarks = 0 #[0 for _ in range(8)]
                    VT = V_T(masses_higgs, masses_gauge, masses_quarks, T)
                    #VT += V_daisy(masses_gauge, masses_gauge_T, T)
                else:
                    VT = 0
                
                
                Z[i,j] = V0_point + Vcw_point + Vct_point + VT /246 # why v here?
                
        return X, Y, Z



    ########## Temperature dependent plots ##########
    
    