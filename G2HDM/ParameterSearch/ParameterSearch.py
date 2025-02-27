
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
from ..Model2HDM.Model2HDM import Model2HDM
from ..Model2HDM.methods_Model2HDM import *
from ..ModelCalculators.Model2HDMCalculator import Model2HDMCalculator

from . parameter_constraints import *


#################### General Methods ####################

class ParamSearch:
    # Add a folder inside the class
    # Plot points inside a bin (like histogram)
    # Include all 8 masses when saving the params
    def __init__(self, model:Model2HDM, name:str="unnamed_ps", subs_VEV_values:dict=None):
        
        # Input
        self.model = model
        self.model_name = model.name
        self.name = name
        
        # Paths
        os.makedirs(os.path.join(model.path_root, "paramerer_search"), exist_ok=True)
        
        self.path = os.path.join(model.path_root, "paramerer_search", name)
        os.makedirs(self.path, exist_ok=True)
        
        self.path_data = os.path.join(self.path , "data")
        os.makedirs(self.path_data, exist_ok=True)
        
        self.path_plots = os.path.join(self.path , "figures")
        os.makedirs(self.path_plots, exist_ok=True)
        
        self.path_plk = os.path.join(self.path , "ParamSearch_data", "ps.pkl")
        os.makedirs(self.path_plk, exist_ok=True)
        
        # Default subs
        if subs_VEV_values == None:
            raise Exception("subs_VEV_values not found")
        self.subs_VEV_values = subs_VEV_values
        
        # define indexes
        self.V0_params_indexrange = [0,14]
        self.VCT_params_length = 14
        N = 14+self.VCT_params_length
        self.VCT_params_indexrange  = [14, N]
        self.masses_indexrange  = [N, N+8]
        
        # constraints
        self.constraints_have_been_applied = False
        self.applied_constraints = {
            "positive masses": True, # Default
            "renormalized": True, # Default
            "perturbative": None, # Check later
            "Tadpole conditions": None, # Check later
            "positivity": False,
            "unitarity": False,
            "oblique": False,
            
        }
        
    def save(self):
        save_pkl(self, self.path_plk)
        
    def assign_parameter_ranges(self, params_ranges:list, params_free:list, params_relations:list):

        self.params_ranges = params_ranges
        self.params_free = params_free
        self.params_relations = params_relations
        
        # Substitutions
        self.params_symbols = self.model.V0_params + [symb for symb in self.model.VEVs if symb != 0]
        self.subs_parameter_relations = {symb:expr for symb, expr in zip(self.params_symbols, self.params_relations)}
        self.params = params_relations
        
        # Assertions
        
        assert len(self.params_ranges) == len(self.params_free), "params_ranges and params_free must have the same length"
        assert len(self.params_relations) == len(self.params_symbols), "params_relations and params_symbols must have the same length"
    
    #################### Methods ####################

    def add_point(self, point):
        pass
    
    def get_point(self, index, output:str="all", filename:str=None)->list:
        if filename == None:
            filename = self.name
        data = load_data(filename, path=self.path_data)
        point = [data[key][index] for key in data.keys()]
        if output == "all":
            return point
        elif output == "V0_params":
            return point[self.V0_params_indexrange[0]:self.V0_params_indexrange[1]]
        elif output == "VCT_params":
            return point[self.VCT_params_indexrange[0]:self.VCT_params_indexrange[1]]
        elif output == "masses":
            return point[-8:] #[self.masses_indexrange[0]:self.masses_indexrange[1]]
        else:
            raise Exception("Invalid output. Choose 'all', 'V0_params', 'VCT_params' or 'masses'")
    
    def get_points(self, indexes:list, output:str="all", filename:str=None)->list[list]:
        if filename == None:
            filename = self.name
        data = load_data(filename, path=self.path_data)
        points = []
        for index in indexes:
            point = [data[key][index] for key in data.keys()]
            if output == "all":
                points.append(point)
            elif output == "V0_params":
                points.append(point[self.V0_params_indexrange[0]:self.V0_params_indexrange[1]])
            elif output == "VCT_params":
                points.append(point[self.VCT_params_indexrange[0]:self.VCT_params_indexrange[1]])
            elif output == "masses":
                points.append(point[-8:])
            else:
                raise Exception("Invalid output. Choose 'all', 'V0_params', 'VCT_params' or 'masses'")
        return points
    
    
    def points(self, filename:str=None)->int:
        """Returns the number of points in the datafile"""
        if filename == None:
            filename = self.name
        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        return dataframe.shape[0]
    
    def default_point(self, output:str="all"):
        """Default point for the parameter search. May be used as a reference point."""
        # Values
        v = 246

        # Tadpole conditions
        Z1, R_Z6, I_Z6 = 1, 1, 1
        Y11 = -v**2/2*Z1
        R_Y12 = v**2/2*R_Z6
        I_Y12 = v**2/2*I_Z6
        # Mass parameter reassign
        MHp, Z3 = 500, 4
        Y22 = MHp**2 - (v**2/2)*Z3

        # Default V0 params
        V0_params_values = [Y11, Y22,
                            Z1,1,Z3,4,
                            R_Y12,I_Y12,
                            -1,-1,R_Z6,I_Z6,-1,1]
        
        # Masses at VEV
        subs_V0_params_values = {symb:val for symb,val in zip(self.model.V0_params, V0_params_values)}
        masses = self.model.DATA.get("M0_eigenvalues_VEV", None) #could use M0 and diagonlize, but need to know the order then
        assert masses != None, "M0_eigenvalues_VEV not found in model.DATA"
        masses_values = [0 for _ in range(len(masses))] # neutral first
        for i in range(len(masses)):
            m = masses[i].subs(subs_V0_params_values | self.subs_VEV_values).evalf()
            masses_values[i] = sp.sign(sp.re(m))*sp.sqrt(sp.Abs(sp.re(m)))
        
        # Counterterm values
        VCT_params_values = Model2HDMCalculator(self.model, subs_V0_params_values, self.subs_VEV_values).counterterm_values()
        
        res = 0
        if output == "all":
            res = V0_params_values + VCT_params_values + masses_values
        elif output == "V0_params":
            res = V0_params_values
        elif output == "VCT_params":
            res = VCT_params_values
        elif output == "masses":
            res = masses_values
        else:
            raise Exception("Invalid output. Choose 'all', 'V0_params', 'VCT_params' or 'masses'")
        
        return res

    def show(self, filename:str=None):
        if filename == None:
            filename = self.name
        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        display(dataframe)
    
    def apply_constraints(self, new_name:str, constraints:list) -> object:
        """ 
        creates a new datafile and object with params that satisfy the constraints
        """
        new_dataframe = load_data(filename=self.name, path=self.path_data, as_dict=False)
        new_ps = ParamSearch(self.model, name=new_name, subs_VEV_values=self.subs_VEV_values)
        new_ps.constraints_have_been_applied = True
        if "positivity" in constraints:
            new_dataframe = constraint_positivity(self, new_dataframe)
            new_ps.applied_constraints["positivity"] = True
        if "unitarity" in constraints:
            new_dataframe = constraint_unitarity(self, new_dataframe)
            new_ps.applied_constraints["unitarity"] = True
        if "global minimum" in constraints:
            new_dataframe = constraint_global_minimum(self, new_dataframe)
            new_ps.applied_constraints["global minimum"] = True
        if "oblique" in constraints:
            new_dataframe = constraint_oblique(self, new_dataframe)
            new_ps.applied_constraints["oblique"] = True
            
        save_data(new_dataframe, filename=new_name, path=new_ps.path_data, merge=False, show_size=True)
            
        return new_ps
    
    #################### Parameter search #################### 
    
    def __unpack_data_multiprocessing(self, results):
        
        # unpack and merge the data
        params_keys = [sp.latex(symb) for symb in self.params_symbols]
        VCT_params_keys = [sp.latex(symb) for symb in self.model.VCT_params]
        masses_keys = [f"m_{i+1}" for i in range(8)] # 8 masses
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

        return results_final
    
    # Main parameter search method
    def ps(self, N_processes:int=1, runtime:int=None, iterations:int=None, filename=None, merge:bool=True, symbolic_masses:bool=True):
        
        # Constraints
        apply_constraints_post = False
        if self.constraints_have_been_applied:
            print("Constraints have been applied. The parameter search will be performed on the constrained data.")
            response = input("Do you want to continue (y/n)?")
            if response.lower() == "y":
                response2 = input("Do you want to apply the same constraint post search? (y/n)?")
                if response2.lower() == "y":
                    apply_constraints_post = True
            else:
                print("Aborting parameter search.")
                return
        
        # name 8
        if filename == None:
            filename = self.name
        
        
        
        # Printing
        N_free_params = sum([1 for el in self.params_ranges if not isinstance(el, int)])
        print("Free parameters: ", N_free_params)
        
        # Multiprocessing
        
        kwargs = {
            "param_ranges":self.params_ranges,
            "params_free":self.params_free,
            
            "params":self.params,
            
            "model":self.model,
            "subs_VEV_values":self.subs_VEV_values
        }
        
        if symbolic_masses:
            # Assign symbolic masses from model
            masses = self.model.DATA.get("M0_eigenvalues_VEV", None) #could use M0 and diagonlize, but need to know the order then
            assert masses != None, "M0_eigenvalues_VEV not found in model.DATA"
            masses = [m.subs(self.subs_parameter_relations) for m in masses]
            kwargs["masses"] = masses
            iterator = iterator_symbolic_ps
        else:
            iterator = iterator_ps
            kwargs["M0"] = self.model.getdata("M0", None)
        
        results = multiprocessing(iterator, kwargs, 
                              processes=N_processes, max_runtime=runtime, max_iterations=iterations)
    
        results_final = self.__unpack_data_multiprocessing(results)
        
        # Save data to file
        #filename = filename + "_ps"
        save_data(results_final, filename=self.name, path=self.path_data, merge=merge, show_size=True)
        
    
    #def ps(self):
        """unsorted_masses"""
        #pass
        
    
    #################### Plotting ####################
    
        

#################### Iterators ####################

def iterator_ps(kwargs):
    # unpacking
    model = kwargs["model"]
    param_ranges = kwargs["param_ranges"]
    params = kwargs["params"]
    params_free = kwargs["params_free"]
    subs_VEV_values = kwargs["subs_VEV_values"]
    
    #print("test")
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
    subs_V0_params_values = {symb:val for symb,val in zip(model.V0_params, params_values[0:14])}
    
    MDC = Model2HDMCalculator(model, subs_V0_params_values, subs_VEV_values)
    
    MDC.assign_counterterm_values(raise_warning=False)
    if MDC.is_renormalized == False:
        return None
    VCT_params_values = MDC.VCT_params_values #MDC.counterterm_values()
        
    # Effective masses at vev
    masses,_ = MDC.calculate_masses_higgs()
    #display(masses)
    masses_values = [0 for m in range(len(masses))]
    for i, m in enumerate(masses):
        if m < 0:
            sign = -1
        else:
            sign = 1
        masses_values[i] = sign*sp.sqrt(sp.Abs(m))
    
    #display(masses_values)
    
    # postive masses constraint
    if not all([m >= 0 for m in masses_values]):
        return None
    
    # Return result
    result = [params_values, VCT_params_values, masses_values] 
    
    return result


def iterator_symbolic_ps(kwargs):
    
    # unpacking
    model = kwargs["model"]
    param_ranges = kwargs["param_ranges"]
    params = kwargs["params"]
    params_free = kwargs["params_free"]
    masses = kwargs["masses"]
    subs_VEV_values = kwargs["subs_VEV_values"]
    
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
    subs_V0_params_values = {symb:val for symb,val in zip(model.V0_params, params_values[0:14])}
        
    VCT_params_values = Model2HDMCalculator(model, subs_V0_params_values, subs_VEV_values).counterterm_values()
        
    # Effective masses at vev
    threshold = 1e-5
    masses_values = [0 for m in range(len(masses))]
    for i, m in enumerate(masses):
        m = m.subs(subs_params).evalf()
        if sp.Abs(sp.im(m)) < threshold*sp.Abs(sp.re(m)):
            m = sp.re(m)
        if m < 0 or m.is_negative:
            sign = -1
        else:
            sign = 1
        masses_values[i] = sign*sp.sqrt(sp.Abs(m))
    
    if not all([m >= 0 for m in masses_values]):
        return None
    
    # Return result
    result = [params_values, VCT_params_values, masses_values] 
    #print(masses_values)
    
    return result
