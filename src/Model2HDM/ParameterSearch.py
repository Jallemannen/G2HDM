
# Packages
import numpy as np
import sympy as sp
import os
import time 
import random

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

#
from ..MultiProcessing.class_MultiProcessing import * 
from ..MultiProcessing.methods_MultiProcessing import * 

#
#from .class_Model2HDM import Model2HDM
from .methods_Model2HDM import *
from .constraints_ParameterSearch import * # type: ignore


#################### General Methods ####################

# child class?
class ParamSearch:
    # Add a folder inside the class
    # Plot points inside a bin (like histogram)
    # Include all 8 masses when saving the params
    def __init__(self, model):
        self.default_point = self.calculate_default_point()
    
    def __init__(self, model):
        pass

    def add_point(self, point):
        pass
    
    def get_point(self, index, include_V0_params = True, include_VCT_params = True, include_masses = True):
        pass
    
    def calculate_default_point(self):
        pass

#################### Iterators ####################

def iterator_param_search(kwargs):
    
    # unpacking
    model = kwargs["model"]
    param_ranges = kwargs["param_ranges"]
    params = kwargs["params"]
    params_free = kwargs["params_free"]
    masses = kwargs["masses"]
    
    # Choose random values for the free parameters within the ranges
    params_free_values = [0 for _ in range(len(param_ranges))]
    for i, param_range in enumerate(param_ranges):
        if isinstance(param_range, (int,float)):
            params_free_values[i] = param_range
        else:
            params_free_values[i] = random.uniform(param_range[0], param_range[1])
    
    # Subs
    subs_params = {symb:val for symb, val in zip(params_free, params_free_values)}
    
    # params (V0 + VEV)
    params_values = [0 for _ in range(len(params))] #[param.subs(subs_params) for param in params]
    for i, param in enumerate(params):
        if isinstance(param, (int,float)):
            params_values[i] = param
        else:
            params_values[i] = param.subs(subs_params)
    
    # VCT params
    VCT_params_values = calculate_counterterm_values(model, V0_params_values=params_values[0:14], VEV_values=params_values[14:], show_solution=False)
    
    # Effective masses at vev
    threshold = 1e-5
    masses_values = [0 for m in range(len(masses))]
    for i, m in enumerate(masses):
        m = m.subs(subs_params).evalf()
        if sp.Abs(sp.im(m)) < threshold*sp.Abs(sp.re(m)):
            m = sp.re(m)
        masses_values[i] = sp.sign(m)*sp.sqrt(sp.Abs(m))
    
    if not all([m >= 0 for m in masses_values]):
        return None
    
    # Return result
    result = [params_values, VCT_params_values, masses_values] 
    #print(masses_values)
    
    return result

#################### Data collection ####################


def param_search(model:object, params_ranges:list=None, params_free:list=None, params_relations:list=None,
                 N_processes:int=1, runtime:int=None, iterations:int=None, filename="unnamed", merge:bool=True) -> None:
    """
    Perform a parameter search for the free parameters of the model.

    Args:
        model (model2HDM): model2HDM object
        params_ranges (list): list of ranges or numbers for the free parameters. Example: [[0,1], 1, [-2,2]]
        params_free (list): list of sypy symbols representing the free parameters
        params_relations (list): relations between the free parameters and the model parameters (V0 + VEV). Can also be a number/constant.
        N_processes (int, optional): number of parallel processes. Defaults to 1.
        runtime (int, optional): set a maximum runtime in seconds. 
        iterations (int, optional): set a maximum number of iterations.
    """
    
    # Unpacking
    if params_ranges == None and params_free == None and params_relations == None:
        param_search_inputs = model.DATA.get("param_search_inputs", None)
        if param_search_inputs == None:
            raise Exception("param_search_inputs not found in model.DATA")
        else:
            params_ranges = param_search_inputs.get("params_ranges", None)
            params_free = param_search_inputs.get("params_free", None)
            params_relations = param_search_inputs.get("params_relations", None)
    if params_ranges == None or params_free == None or params_relations == None:
        raise Exception("params_ranges, params_free, params_relations not found")

    params_symbols = model.V0_params + [symb for symb in model.VEVs if symb != 0]

    # Assertions
    assert len(params_ranges) == len(params_free), "params_ranges and params_free must have the same length"
    assert len(params_relations) == len(params_symbols), "params_relations and params_symbols must have the same length"
    
    # Substituting to the free parameters
    subs_parameter_relations = {symb:expr for symb, expr in zip(params_symbols, params_relations)}
    
    params = params_relations
    masses = model.DATA.get("M0_eigenvalues_VEV", None) #could use M0 and diagonlize, but need to know the order then
    assert masses != None, "M0_eigenvalues_VEV not found in model.DATA"
    masses = [m.subs(subs_parameter_relations) for m in masses]
    
    # Printing
    N_free_params = sum([1 for el in params_ranges if not isinstance(el, int)])
    print("Free parameters: ", N_free_params)
    
    # Multiprocessing
    kwargs = {
        "param_ranges":params_ranges,
        "params_free":params_free,
        
        "params":params,
        "masses":masses,
        
        "model":model
    }
    
    # debug
    #results = iterator_param_search(kwargs)
    #display(results)

    # Start multiprocessing
    results = multiprocessing(iterator_param_search, kwargs, 
                              processes=N_processes, max_runtime=runtime, max_iterations=iterations)
    

    #display(results)
    
    # unpack and merge the data
    params_keys = [sp.latex(symb) for symb in params_symbols]
    VCT_params_keys = [sp.latex(symb) for symb in model.VCT_params]
    masses_keys = [f"m_{i+1}" for i in range(len(masses))]
    params_all_keys = params_keys + VCT_params_keys + masses_keys
    #model.DATA["paramsearch_params_all_keys"] = params_all_keys
    
    # Merge data
    params_values_merged = [[] for _ in range(len(params_all_keys))]
    for result in results:
        params_values = result[0] + result[1] + result[2]
        for j, param_value in enumerate(params_values):
            params_values_merged[j].append(param_value)
            
    # create data container
    results_final = {}
    for i, key in enumerate(params_all_keys):
        results_final[key] = params_values_merged[i]

    # Save data to file
    filename = filename + "_ps"
    save_data(results_final, filename, path=model.path_data, merge=merge, show_size=True)
    
    
    
    
def apply_constraints(datafile_name, constraints:list):
    """ 
    creates a new datafile with params that satisfy the constraints
    """
    if "positivity" in constraints:
        pass