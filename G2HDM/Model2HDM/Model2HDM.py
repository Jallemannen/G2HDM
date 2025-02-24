
# Default Packages
import os 
import inspect
import numpy as np
import sympy as sp
from IPython.display import display, Math # type: ignore
import time

# Custom Packages
from .methods_Model2HDM import *

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from ..utils import constants as const
from .standard_symbols import STANDARD_SYMBOLS
from .standard_potentials import potential_V0, potential_VCT_gen
from . import standard_fields


class Model2HDM:
    
    def __init__(self,
                 name:str="model",
                 fields:list[sp.Symbol]=None,
                 bgfields:list[sp.Symbol]=None,
                 VEVs:list[sp.Symbol]=None,
                 project_root:str=None,
                 evaluate_now:bool=False):
        
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
        self.SYMBOLS = {} #symbols
        self.SYMBOLS.update(STANDARD_SYMBOLS)
        p = STANDARD_SYMBOLS
        
        # Construct the fields
        if fields is None:
            fields = standard_fields.fields_gen
        if bgfields is None:
            bgfields = standard_fields.bgfields_gen
        if VEVs is None:
            VEVs = standard_fields.VEVs_gen
            
        self.fields = fields
        self.bgfields = bgfields
        self.VEVs = VEVs
        self.__construct_fields(bgfields=bgfields, VEVs=VEVs, fields=fields)   
        
        # Div fields
        self.field1_matrixsymbol = p["field1_matrixsymbol"]
        self.field2_matrixsymbol = p["field2_matrixsymbol"]
        
        self.massfields = [p["h1"], p["h2"], p["h3"], p["h4"], p["h5"], p["h6"], p["h7"], p["h8"]]
        
        # Potentials (initialize as standard potentials)
        

        """V0_parameters_keys = ["mu11", "mu22", "L1", "L2", "L3", "L4", "L12", "L5", "L6", "L7"]
        self.V0_params_complex = {}
        for key in V0_parameters_keys:
            if key in STANDARD_SYMBOLS:
                self.V0_params_complex[key] = STANDARD_SYMBOLS[key]"""
        
        
        self.V0_params_complex = [p["mu11"], p["mu22"], p["L1"], p["L2"], p["L3"], p["L4"], p["mu12"], p["L5"], p["L6"], p["L7"]]
        self.V0_params = subs_complex_params(self.V0_params_complex)
        self.V0 = None
        self.construct_V0(potential_V0, self.V0_params_complex, fields = self.fields, bgfields = self.bgfields)
        
        self.VCT_params_complex = [p["dmu11"], p["dmu22"], p["dL1"], p["dL2"], p["dL3"], p["dL4"], 
                                   p["dmu12"], p["dL5"], p["dL6"], p["dL7"],
                                   p["dT1"], p["dT2"], p["dTCP"], p["dTCB"],
                                   ]# p["dD13"]] , p["dD33"], p["dD14"], p["dD24"], p["dD67"], p["dD78"]]
        self.VCT_params = subs_complex_params(self.VCT_params_complex)
        self.VCT = None
        self.construct_VCT(potential_VCT_gen, self.VCT_params_complex, fields = self.fields, bgfields = self.bgfields)
        
        # Gauge bosons
        W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
        B = sp.Symbol("B", real=True)
        self.fields_gauge = [B, W1, W2, W3]
        G0, G1, G2, G3 = sp.symbols("G_0 G_1 G_2 G_3", real=True)
        self.massfields_gauge = [G0, G1, G2, G3]
        
        # Quarks
        ...
        
        # Leptons
        ...
        
        # Generica Data for storing eg analytical solutions
        self.DATA = {}
        
    #################### Constructors ####################
    
    def __construct_fields(self,
                         bgfields:list[sp.Symbol],
                         VEVs:list[sp.Symbol],
                         fields:list[sp.Symbol] = None):
        
        # Assertions
        if fields is None:
            fields = self.fields
        assert isinstance(fields, list), "The fields must be a list."
        assert len(fields) == 8, "The number of fields must be 8."
        assert isinstance(bgfields, list), "The bgfields must be a list."
        assert len(bgfields) == 8, "The number of bgfields must be 8."
        assert isinstance(VEVs, list), "The VEVs must be a list."
        assert len(VEVs) == 8, "The number of VEVs must be 8."
        
        f1, f2, f3, f4, f5, f6, f7, f8 = fields
        w1, w2, w3, w4, w5, w6, w7, w8 = bgfields
        
        self.bgfield1 = sp.simplify(sp.Matrix([w5+sp.I*w7, w1+sp.I*w3])/sp.sqrt(2))
        self.bgfield2 = sp.simplify(sp.Matrix([w6+sp.I*w8, w2+sp.I*w4])/sp.sqrt(2))
        self.field1 = sp.simplify(sp.Matrix([f5+sp.I*f7, f1+sp.I*f3])/sp.sqrt(2) + self.bgfield1)
        self.field2 = sp.simplify(sp.Matrix([f6+sp.I*f8, f2+sp.I*f4])/sp.sqrt(2) + self.bgfield2)
        
        self.fields = fields
        self.bgfields = bgfields
        self.VEVs = VEVs
        
        self.update_symbols({f"f{i+1}": self.fields[i] for i in range(8)})
        self.update_symbols({f"w{i+1}": self.bgfields[i] for i in range(8)})
        self.update_symbols({f"v{i+1}": self.VEVs[i] for i in range(8)})
        #print("OBS, Also need to update the potentials!")
                
    def construct_V0(self, func, parameters, **kwargs):
        """
        Constructs the tree-level potential V0.
        """
        self.V0, self.V0_display = self.__validate_and_unpack_potential(func, parameters, **kwargs)
        self.V0_params_complex = parameters
        self.V0_params = subs_complex_params(parameters)
        
    def construct_VCT(self, func, parameters, **kwargs):
        """
        Constructs the counterterm potential VCT.
        """
        self.VCT, self.VCT_display = self.__validate_and_unpack_potential(func, parameters, **kwargs)
        self.VCT_params_complex = parameters
        self.VCT_params = subs_complex_params(parameters)
    
    def update_symbols(self, symbols:dict):
        """
        Updates the symbols library.
        """
        self.SYMBOLS.update(symbols)
        
    # Validate and unpack potential into V and V_display
    def __validate_and_unpack_potential(self, potential_func, params, **kwargs):
        # Set default values if None
        """if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}

        # Validate types of args and kwargs
        if not isinstance(args, tuple):
            raise TypeError(f"Expected 'args' to be a tuple, got {type(args).__name__}")
        if not isinstance(kwargs, dict):
            raise TypeError(f"Expected 'kwargs' to be a dict, got {type(kwargs).__name__}")
        """
        # Inspect the function signature
        param_names = inspect.signature(potential_func).parameters
        bound_args = {}
        try:
            # Bind the provided arguments to the function's signature
            if "fields" in param_names and "fields" in kwargs:
                bound_args["fields"] = kwargs["fields"]
            if "bgfields" in param_names and "bgfields" in kwargs:
                bound_args["bgfields"] = kwargs["bgfields"]
        except TypeError as e:
            raise TypeError(f"Error binding arguments to {potential_func.__name__}: {e}")

        # Call the function with the bound arguments
        try:
            result = potential_func(self.field1, self.field2, params, **bound_args)
            result_display = potential_func(self.field1_matrixsymbol, self.field2_matrixsymbol, params, **bound_args)
        except TypeError as e:
            raise TypeError(f"TypeError in {potential_func.__name__}: {e}")
        except ValueError as e:
            raise ValueError(f"ValueError in {potential_func.__name__}: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in {potential_func.__name__}: {e}")

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
            raise ValueError(f"Error unpacking the result from {potential_func.__name__}: {e}")

        # Combine and process the potential
        try:
            V = V_matrix[0] + V_scalar 
            V = subs_complex_expressions([V.expand()])[0]
            V_display = [V_matrix_display, V_scalar_display]
        except Exception as e:
            raise ValueError(f"Error processing potential from {potential_func.__name__}: {e}")


        return V.expand(), V_display
    
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
    
    #################### Meta Methods ####################
    
    # Display basic info
    def display_info(self):
        """Function to display basic information about the model."""
        subs_fields_to_zero = {f:0 for f in self.fields}
        print(f"========== Model Information ==========")
        print("Fields:")
        display(self.fields)
        display(self.bgfields)
        display(self.VEVs)
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

        """res = generate_masses_gaugebosons(self, show_procedure=True)
        res2 = generate_masses_fermions(self, type="II")[0]
        print("Top quark mass (type II):")
        display(Math("m_t" + "=" + sp.latex(res2)))"""
        # add info about symmetries, mass terms, CP etc later 
        
    def save(self):
        save_pkl(self, self.path_model)
        
    def delete(self):
        pass
    
    def load_data(self, filename:str):
        return load_data(filename=filename, path=self.path_data)
        
    
    def clear_data(self):
        pass
    
    #################### Methods ####################
    
    def rotate(self, R:sp.Matrix) -> None:
        """
        Rotate the fields by the matrix R.
        """
        # Get the new symbols from R
        new_symbols = {}
        basis_params = []
        for symb in R.free_symbols:
            new_symbols[symb.name] = symb
            basis_params.append(symb)
        self.update_symbols(new_symbols)
        self.basis_params = basis_params
        
        # Rotate the fields
        vec = sp.Matrix([self.field1, self.field2])
        vec_rot = R * vec
        self.field1 = vec_rot[0]
        self.field2 = vec_rot[1]
        
        # Rotate the parameters
        ...
    
    #################### Analytical Methods ####################
    
    # Generate the tree-level masses
    def generate_level0_masses(self, VEV:bool=False, solve_eigenvalues=True, apply_tadpole=True, show_procedure:bool=True, show_solution:bool=True) -> None:
        
        #reload_and_import("G2HDM.Model2HDM.methods_Model2HDM", "generate_level0_masses")
        
        generate_level0_masses(self, VEV, apply_tadpole, solve_eigenvalues, show_procedure, show_solution)
        
        self.save()
        
    # Generate the expressions for the counterterms
    def solve_counterterms(self, extra_eqs=None, show_procedure:bool=True, show_solution:bool=True) -> None:
        
        #reload_and_import("G2HDM.Model2HDM.methods_Model2HDM", "solve_counterterms")
        
        # Solve the counterterms
        solve_counterterms(self, extra_eqs=extra_eqs, show_procedure=show_procedure, show_solution=show_solution)
        
        self.save()
        
    def generate_Veff(self):
        pass