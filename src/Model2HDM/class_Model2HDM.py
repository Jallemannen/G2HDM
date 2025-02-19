
# Default Packages
import os 
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
from src.utils import constants as const


class Model2HDM:
    
    def __init__(self, symbols:dict, potentials:dict, name:str="Model", evaluate_now:bool=False, **kwargs):
        
        # Paths for saving and loading data
        self.name = name
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.path_root = os.path.join(project_root, "Models", "Saved_models", self.name)
        self.path_modeldata = os.path.join(self.path_root, "model_data")
        self.path_model = os.path.join(self.path_modeldata, "model.pkl")
        self.path_data = os.path.join(self.path_root, "Data")
        self.path_plots = os.path.join(self.path_root, "Plots")
        
        # Create directories
        os.makedirs(os.path.join(self.path_root), exist_ok=True)
        os.makedirs(os.path.join(self.path_modeldata), exist_ok=True)
        os.makedirs(os.path.join(self.path_data), exist_ok=True)
        os.makedirs(os.path.join(self.path_plots), exist_ok=True)
        
        # Symbols
        self.SYMBOLS = symbols
        
        # Fields
        self.field1_matrixsymbol = kwargs["field1_matrixsymbol"]
        self.field2_matrixsymbol = kwargs["field2_matrixsymbol"]
        self.field1 = kwargs["field1"]
        self.field2 = kwargs["field2"]
        self.bgfield1 = kwargs["bgfield1"]
        self.bgfield2 = kwargs["bgfield2"]
        
        # Field sets
        self.fields = kwargs["fields"]
        self.massfields = kwargs["massfields"]
        self.bgfields = kwargs["bgfields"]
        self.VEVs = kwargs["VEVs"]
        
        # Parameters
        self.V0_params_symbols = kwargs["V0_params"]
        self.VCT_params_symbols = kwargs["VCT_params"]
        self.V0_params = subs_complex_params(self.V0_params_symbols)
        self.VCT_params = subs_complex_params(self.VCT_params_symbols)
        
        # Values
        self.VEV_values = None
        self.bgfield_values = None
        self.V0_params_values = None
        self.VCT_params_values = None
        
        # Potentials
        V0_func = potentials["V0"]
        VCT_func = potentials["VCT"]
        self.V0 = subs_complex_expressions([V0_func(self.field1, self.field2, self.V0_params_symbols).expand()])[0].expand()
        self.VCT = subs_complex_expressions([VCT_func(self.field1, self.field2, self.VCT_params_symbols , self.fields, self.bgfields).expand()])[0].expand()
        
        self.V0_display = V0_func(self.field1_matrixsymbol, self.field2_matrixsymbol, self.V0_params_symbols)
        self.VCT_display1, self.VCT_display2 = VCT_func(self.field1_matrixsymbol, self.field2_matrixsymbol, self.VCT_params_symbols, self.fields, self.bgfields)
        
        # Gauge bosons
        W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
        B = sp.Symbol("B", real=True)
        self.fields_gauge = [B, W1, W2, W3]
        G0, G1, G2, G3 = sp.symbols("G_0 G_1 G_2 G_3", real=True)
        self.massfields_gauge = [G0, G1, G2, G3]
        
        # Dict for diverse data
        self.DATA = {}
        
        for key, value in kwargs.items():
            self.DATA[key] = value
            
        if evaluate_now:
            pass
            
        # Save the model
        #self.save()
    
    #################### Setters ####################
                
    def assign_V0_params_values(self, params_values:dict):
        if self.V0_params_values != None:
            print("Warning: Overwriting existing parameter values.")
        if len(params_values) != len(self.V0_params):
            raise ValueError("Please assign values to all parameters.")
        self.V0_params_values = params_values
        self.save()
    
    def assign_bgfield_values(self, bgfield_values:dict):
        if self.bgfield_values != None:
            print("Warning: Overwriting existing background field values.")
        if len(bgfield_values) != len(self.fields):
            raise ValueError("Please assign values to all background fields.")
        self.bgfield_values = bgfield_values
        self.save()
        
    def assign_VEV_values(self, VEV_values:dict):
        if self.VEV_values != None:
            print("Warning: Overwriting existing VEV values.")
        if len(VEV_values) != len(self.VEVs):
            raise ValueError("Please assign values to all VEV fields.")
        self.VEV_values = VEV_values
        self.save()
        
    def assign_VCT_constraints(self, constraints:list):
        """Constraints as a list of eqs"""
        self.DATA["VCT_constraints"] = constraints
        self.save()
        
    def assign_param_search_inputs(self, params_ranges:list, params_free:list, params_relations:list):
        self.DATA["param_search_inputs"] = {
            "params_ranges":params_ranges,
            "params_free":params_free,
            "params_relations":params_relations
        }
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

        print(f"========== Model Information ==========")
        print("Field definitions:")
        display(Math(sp.latex(self.field1_matrixsymbol) + "=" + sp.latex(self.field1)))
        display(Math(sp.latex(self.field2_matrixsymbol) + "=" + sp.latex(self.field2)))
        print("Tree-level potential:")
        display(Math("V_0" + "=" + sp.latex(self.V0_display)))
        print("Counterterm potential:")
        display(Math("V_{CT}" + "=" + sp.latex(self.VCT_display1) + sp.latex(self.VCT_display2)))

        res = generate_masses_gaugebosons(self, show_procedure=True)
        res2 = generate_masses_fermions(self, type="II")[0]
        print("Top quark mass (type II):")
        display(Math("m_t" + "=" + sp.latex(res2)))
        # add info about symmetries, mass terms, etc later 
        
    def save(self):
        save_pkl(self, self.path_model)
        
    def delete(self):
        pass
    
    def load_data(self, filename:str):
        return load_data(filename=filename, path=self.path_data)
        
    
    def clear_data(self):
        pass

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
    
    
    
    
    
    
    
    