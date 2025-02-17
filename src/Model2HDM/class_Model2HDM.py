
# Default Packages
import os 
import numpy as np
import sympy as sp
from IPython.display import display, Math # type: ignore
import time

# Custom Packages
from .methods_Model2HDM import *
from .methods_parameterSearch import *
#from .constraints import *

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from src.utils import constants as const


class Model2HDM:
    
    def __init__(self, symbols:dict, potentials:dict, name:str="Model", **kwargs):
        
        # Paths for saving and loading data
        self.name = name
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
        self.path_root = os.path.join(project_root, "Models", "Saved models", self.name)
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
        
        # Dict for diverse data
        self.DATA = {}
        
        for key, value in kwargs.items():
            self.DATA[key] = value
            
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
    
    # Calculate the counterterm parameter values
    def calculate_counterterm_values(self, show_solution:bool=True) -> None:
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "calculate_counterterm_values")
        
        # Assertions
        #assert self.DATA["VCT_params_eqs_sol"] != None, "Please solve the counterterms first."
        #assert kwargs["VEV_values"] != None, "Please assign values to all VEV fields."
        #assert kwargs["bgfield_values"] != None, "Please assign values to all background fields."
        #assert kwargs["V0_params_values"] != None, "Please assign values to all parameters."
        
        # Calculate the counterterm values
        VCT_params_values = calculate_counterterm_values(self, show_solution=show_solution)
        
        # Save
        self.VCT_params_values = VCT_params_values
           
        self.save()
        
    #################### Generate datasets ####################
        
    def calculate_data2D_level0(self, free_bgfield, N, Xrange,
                              name:str="", save:bool=True,
                              calc_potential=True, calc_mass=True,):
        # params_values=None, bgfield_values=None, VEV_values=None,
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "calculate_data2D_level0")
        
        # Assertions
        """if self.V0_params_values == None:
            params_values = self.V0_params_values
        if bgfield_values == None:
            bgfield_values = self.bgfield_values
        if VEV_values == None:
            VEV_values = self.VEV_values
        if params_values == None or bgfield_values == None or VEV_values == None:
            raise ValueError("Please assign values to all parameters, background fields and VEVs.")"""
        
        # Assertions
        """if calc_potential:
            V0 = self.potential("V0")
        if calc_mass:
            M0 = self.masses("M0")
        if not calc_potential and not calc_mass:
            raise ValueError("Please select at least one of the following to generate data: calc_potential, calc_mass")
        """
        
        # Generate the data
        X, Y1, Y2 = calculate_data2D_level0(self, free_bgfield,
                          N=N, Xrange=Xrange,
                          calc_potential=calc_potential, calc_mass=calc_mass, sorting=True)

        if save:
            if name != "":
                name = name + "_"
            if calc_potential:
                save_data({"omega":X, "V":Y1}, f"{name}potential_level0_data", self.path_data)
            if calc_mass:
                save_data({"omega":X} | {f"m_{j+1}":Y2[j] for j in range(8)}, f"{name}mass_level0_data", self.path_data)
    
    # One-loop level potential and masses
    def calculate_data2D_level1(self, free_bgfield, N, Xrange, calc_potential=True, calc_mass=True, name:str="", save:bool=True):
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "calculate_data2D_level1")
        
        # Generate the data
        X, Y1, Y2 = calculate_data2D_level1(self, free_bgfield, N, Xrange=Xrange, calc_potential=calc_potential, calc_mass=calc_mass)
    
        if save:
            if name != "":
                name = name + "_"
            if calc_potential:
                save_data({"omega":X, "V":Y1}, f"{name}potential_level1_data", self.path_data)
            if calc_mass:
                save_data({"omega":X} | {f"m_{j+1}":Y2[j] for j in range(8)}, f"{name}mass_level1_data", self.path_data)
    
    # Tree-level potential and masses, with 2 free background fields
    def calculate_data3D_level0():
        pass
    
    def calculate_data3D_level1():
        pass
    
    def plot_data2D(self, name="", level=0, plot_potential=True, plot_masses=True, save_fig:bool=True,
                    Xrange=None, Yrange=None):
        
        reload_and_import("src.Model2HDM.methods_Model2HDM", "load_and_plot_data2D")
        
        load_and_plot_data2D(self, name, level, plot_potential, plot_masses, Xrange=Xrange, Yrange=Yrange)

    
    #################### Parameter search ####################

    def param_search(self, params_ranges:list=None, params_free:list=None, params_relations:list=None,
                 N_processes:int=1, runtime:int=None, iterations:int=None, filename="paramsearch", merge:bool=True)->None:
        
        reload_and_import("src.Model2HDM.methods_parameterSearch", "param_search")
        
        param_search(self, params_ranges, params_free, params_relations, N_processes, runtime, iterations, filename, merge)
    
    
    
    
    
    
    
    
    