
# Default Packages
import os 
import inspect

#
import numpy as np
import sympy as sp
from IPython.display import display, Math # type: ignore
import time

# Custom Packages
from .methods_Model2HDM import *
#from .ParameterSearch import *
#from .constraints import *

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from ..utils import constants as const
from .standard_symbols import STANDARD_SYMBOLS

class Model2HDM:
    """Class for the 2HDM model. Internal methods only include analtyical solvers
    """
    def __init__(self, 
                 symbols:dict, 
                 #potentials:dict, 
                 model_data:dict, #replace as main data input for non-symbols
                 name:str="Model",
                 project_root:str=None,
                 evaluate_now:bool=False
                 ):
        
        # Auto evaluate enviroment to determine print method. Update whenever the model is loaded.
        
        # Paths for saving and loading data
        self.name = name
        if project_root == None:
            project_root = os.path.abspath(os.path.join(os.getcwd(), ""))
        self.path_root = os.path.join(project_root, "saved_models", self.name)
        self.path_modeldata = os.path.join(self.path_root, "model_data")
        self.path_model = os.path.join(self.path_modeldata, "model.pkl")
        self.path_data = os.path.join(self.path_root, "Data")
        self.path_plots = os.path.join(self.path_root, "Plots")
        
        # Create directories
        os.makedirs(os.path.join(self.path_root), exist_ok=True)
        os.makedirs(os.path.join(self.path_modeldata), exist_ok=True)
        os.makedirs(os.path.join(self.path_data), exist_ok=True)
        os.makedirs(os.path.join(self.path_plots), exist_ok=True)
        
        # Symbols library
        self.SYMBOLS = symbols
        self.SYMBOLS.update(STANDARD_SYMBOLS)
        
        # Field lists (non-optional)
        self.fields = model_data["fields"]
        self.bgfields = model_data["bgfields"]
        self.VEVs = model_data["VEVs"]
        
        # Construct fields from fields
        assert isinstance(self.fields, list), "The fields must be a list."
        assert len(self.fields) == 8, "The number of fields must be 8."
        assert isinstance(self.bgfields, list), "The bgfields must be a list."
        assert len(self.bgfields) == 8, "The number of bgfields must be 8."
        assert isinstance(self.VEVs, list), "The VEVs must be a list."
        assert len(self.VEVs) == 8, "The number of VEVs must be 8."
        f1, f2, f3, f4, f5, f6, f7, f8 = self.fields
        w1, w2, w3, w4, w5, w6, w7, w8 = self.bgfields
        #v1, v2, v3, v4, v5, v6, v7, v8 = self.VEVs
        
        self.bgfield1 = sp.simplify(sp.Matrix([w5+sp.I*w7, w1+sp.I*w3])/sp.sqrt(2))
        self.bgfield2 = sp.simplify(sp.Matrix([w6+sp.I*w8, w2+sp.I*w4])/sp.sqrt(2))
        self.field1 = sp.simplify(sp.Matrix([f5+sp.I*f7, f1+sp.I*f3])/sp.sqrt(2) + self.bgfield1)
        self.field2 = sp.simplify(sp.Matrix([f6+sp.I*f8, f2+sp.I*f4])/sp.sqrt(2) + self.bgfield2)
        
        
        # Optional kwargs. Otherwise use default values
        if "massfields" in model_data:
            self.massfields = model_data["massfields"]
            assert isinstance(self.massfields, list), "The massfields must be a list."
            assert len(self.massfields) == 8, "The number of massfields must be 8."
        else:
            model_data["massfields"] = [self.SYMBOLS["h1"], self.SYMBOLS["h2"], self.SYMBOLS["h3"], self.SYMBOLS["h4"], self.SYMBOLS["h5"], self.SYMBOLS["h6"], self.SYMBOLS["h7"], self.SYMBOLS["h8"]]
        if "field1_matrixsymbol" in model_data:
            self.field1_matrixsymbol = model_data["field1_matrixsymbol"]
        else:
            self.field1_matrixsymbol = self.SYMBOLS["Phi1_matrix"]
        if "field2_matrixsymbol" in model_data:
            self.field2_matrixsymbol = model_data["field2_matrixsymbol"]
        else:
            self.field2_matrixsymbol = self.SYMBOLS["Phi2_matrix"]
        
        # Parameters
        self.V0_params_symbols = model_data["V0_params"]
        self.VCT_params_symbols = model_data["VCT_params"]
        self.V0_params = subs_complex_params(self.V0_params_symbols)
        self.VCT_params = subs_complex_params(self.VCT_params_symbols)
        
        # Values
        """self.VEV_values = None
        self.bgfield_values = None
        self.V0_params_values = None
        self.VCT_params_values = None"""
        
        # Potentials
        # * Add a check that the potentials are valid functions
        # and that they match the number of parameters
        V0_func = model_data["V0"]
        VCT_func = model_data["VCT"]
        self.V0, self.V0_display = self.__validate_and_unpack_potential(V0_func, self.V0_params_symbols)
        self.VCT, self.VCT_display = self.__validate_and_unpack_potential(VCT_func, self.VCT_params_symbols)
        #self.V0 = subs_complex_expressions([V0_func(self.field1, self.field2, self.V0_params_symbols).expand()])[0].expand()
        #self.VCT = subs_complex_expressions([VCT_func(self.field1, self.field2, self.VCT_params_symbols , self.fields, self.bgfields).expand()])[0].expand()
        
        #self.V0_display = V0_func(self.field1_matrixsymbol, self.field2_matrixsymbol, self.V0_params_symbols)
        #self.VCT_display1, self.VCT_display2 = VCT_func(self.field1_matrixsymbol, self.field2_matrixsymbol, self.VCT_params_symbols, self.fields, self.bgfields)
        
        # Gauge bosons
        W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
        B = sp.Symbol("B", real=True)
        self.fields_gauge = [B, W1, W2, W3]
        G0, G1, G2, G3 = sp.symbols("G_0 G_1 G_2 G_3", real=True)
        self.massfields_gauge = [G0, G1, G2, G3]
        
        # Dict for diverse data 
        # Make private?
        self.DATA = {}
        """
        for key, value in kwargs.items():
            self.DATA[key] = value
            """
            
        if evaluate_now:
            pass
            
        # Save the model
        #self.save()
    
    #################### Setters ####################
        
    def assign_VCT_constraints(self, constraints:list):
        """Constraints as a list of eqs"""
        self.DATA["VCT_constraints"] = constraints
        self.save()
                
    #################### Getters ####################
    
    def symbol(self, symbol:str, raise_error:bool=True):
        output = self.SYMBOLS.get(symbol)
        if output == None and raise_error:
            raise ValueError(f"Symbol '{symbol}' not found.")
        return output
    
    def getdata(self, data:str, raise_error:bool=True):
        output = self.DATA.get(data)
        if output == None and raise_error:
            raise ValueError(f"Data '{data}' not found.")
        return output
    
    #################### General Methods ####################
    
    # Display basic info
    def display_info(self):
        """Function to display basic information about the model."""
        subs_fields_to_zero = {f:0 for f in self.fields}
        print(f"========== Model Information ==========")
        print("Field definitions:")
        display(Math(sp.latex(self.field1_matrixsymbol) + "=" + sp.latex(self.field1) + r"\to" + sp.latex(self.field1.subs(subs_fields_to_zero)) ))
        display(Math(sp.latex(self.field2_matrixsymbol) + "=" + sp.latex(self.field2) + r"\to" + sp.latex(self.field2.subs(subs_fields_to_zero)) ))
        print("Tree-level potential:")
        if self.V0_display[1] == 0:
            display(Math("V_0" + "=" + sp.latex(self.V0_display[0])))
        else:
            display(Math("V_0" + "=" + sp.latex(self.V0_display[0]) + sp.latex(self.V0_display[1])))
        print("Counterterm potential:")
        if self.VCT_display[1] == 0:
            display(Math("V_{CT}" + "=" + sp.latex(self.VCT_display[0])))
        else:
            display(Math("V_{CT}" + "=" + sp.latex(self.VCT_display[0]) + sp.latex(self.VCT_display[1])))

        res = generate_masses_gaugebosons(self, show_procedure=True)
        res2 = generate_masses_fermions(self, type="II")[0]
        print("Top quark mass (type II):")
        display(Math("m_t" + "=" + sp.latex(res2)))
        # add info about symmetries, mass terms, CP etc later 
        
    def save(self):
        save_pkl(self, self.path_model)
        
    def delete(self):
        pass
    
    def load_data(self, filename:str):
        return load_data(filename=filename, path=self.path_data)
        
    
    def clear_data(self):
        pass
    
    #################### Validation Methods ####################
    
    def __validate_and_unpack_potential(self, potential_func, params):
        V = 1
        V_display = 1
        
        # check number of args
        sig = inspect.signature(potential_func)
        args = sig.parameters
        # Count parameters that don't have default values
        required = [p for p in args.values() if p.default == inspect.Parameter.empty]
        if len(required) not in (3, 5):
            raise ValueError(
                f"Invalid number of required potential function arguments: {len(required)}. "
                "Expected either 3 (x1, x2, params) or 5 (x1, x2, params, fields, bgfields)."
            )
            
        # Call the function
        try:
            if len(required) == 3:
                result = potential_func(self.field1, self.field2, params)
                result_display = potential_func(self.field1_matrixsymbol, self.field2_matrixsymbol, params)
            elif len(required) == 5:
                result = potential_func(self.field1, self.field2, params, self.fields, self.bgfields)
                result_display = potential_func(self.field1_matrixsymbol, self.field2_matrixsymbol, params, self.fields, self.bgfields)
        except ValueError as e:
            print(f"Likely error cause: The numper of parameters must be correct. \n {e}")
        except TypeError as e:
            print(f"Likely error cause: Can't add Matrix symbol with a scalar symbol. Return separatly as (matrixterm, scalarterm). \n {e}")
        except Exception as e: 
            print(f"Error in the potential function. \n {e}")
        

        # Unpack the result
        try:
            if isinstance(result, tuple):
                V_matrix, V_scalar = result
                V_matrix_display, V_scalar_display = result_display
            else:
                V_matrix = result
                V_scalar = 0
                V_matrix_display = result_display
                V_scalar_display = 0
        except Exception as e:
            print(f"Error when returning the value of the potential function. Use tuple(matrixobj, sclarobj) or matrixobj: {e}")
        
        # 
        V = V_matrix[0] + V_scalar
        V = subs_complex_expressions([V.expand()])[0]
        V_display = [V_matrix_display, V_scalar_display]
        
        return subs_complex_expressions([V]), V_display

    #################### Analytical Methods ####################
    
    # Generate the tree-level masses
    def generate_level0_masses(self, VEV:bool=False, solve_eigenvalues=True, apply_tadpole=True, show_procedure:bool=True, show_solution:bool=True) -> None:
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "generate_level0_masses")
        
        generate_level0_masses(self, VEV, apply_tadpole, solve_eigenvalues, show_procedure, show_solution)
        
        self.save()
        
    # Generate the expressions for the counterterms
    def solve_counterterms(self, extra_eqs=None, show_procedure:bool=True, show_solution:bool=True) -> None:
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "solve_counterterms")
        
        # Solve the counterterms
        solve_counterterms(self, extra_eqs=extra_eqs, show_procedure=show_procedure, show_solution=show_solution)
        
        self.save()
    
    #################### Numerical Methods ####################
    
    
    
    
    
    
    
    