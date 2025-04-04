
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
        os.makedirs(os.path.join(model.path_root, "parameter_search"), exist_ok=True)
        
        self.path = os.path.join(model.path_root, "parameter_search", name)
        os.makedirs(self.path, exist_ok=True)
        
        self.path_data = os.path.join(self.path , "data")
        os.makedirs(self.path_data, exist_ok=True)
        
        self.path_plots = os.path.join(self.path , "figures")
        os.makedirs(self.path_plots, exist_ok=True)
        
        self.path_plk = os.path.join(self.path , "ParamSearch_data", "ps.pkl")
        os.makedirs(os.path.join(self.path , "ParamSearch_data"), exist_ok=True)
        
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
            
        # modify paths for new ps
        new_ps.path = os.path.join(self.model.path_root, "parameter_search", new_name)
        os.makedirs(self.path, exist_ok=True)
        new_ps.path_data = os.path.join(new_ps.path, "data")
        os.makedirs(new_ps.path_data, exist_ok=True)
        new_ps.path_plots = os.path.join(new_ps.path, "figures")
        os.makedirs(new_ps.path_plots, exist_ok=True)
        os.makedirs(os.path.join(new_ps.path, "ParamSearch_data"), exist_ok=True)
        new_ps.path_plk = os.path.join(new_ps.path, "ParamSearch_data", "ps.pkl")
        
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

    # WIP
    def plot_hexbin_two_params(self, param1_index: int, param2_index: int, bins=30, filename: str = None, 
                            compare_ps_list=[], compare_ps_list_colors=["grey"], single_color=None, **kwargs):
        from matplotlib.colors import ListedColormap
        
        if filename is None:
            filename = self.name

        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        param1 = dataframe.keys()[param1_index]
        param2 = dataframe.keys()[param2_index]

        fig, ax = plt.subplots()

        # Apply user-defined tick settings
        if "set_xticks" in kwargs:
            ax.set_xticks(kwargs["set_xticks"])
        if "set_yticks" in kwargs:
            ax.set_yticks(kwargs["set_yticks"])

        # Compute axis limits to ensure alignment
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        extent = [x_min, x_max, y_min, y_max]

        # Plot background dataset(s) with a **masked single color**
        for i, compare_ps in enumerate(compare_ps_list):
            compare_dataframe = load_data(compare_ps.name, path=compare_ps.path_data, as_dict=False)

            # Create a masked colormap with one uniform color
            solid_color = compare_ps_list_colors[i]
            cmap_bg = plt.get_cmap(solid_color)
            new_colors_bg = cmap_bg(np.ones(cmap_bg.N))  # Force single color
            new_colors_bg[0] = [1, 1, 1, 0]  # Make zero-count tiles white/transparent
            masked_cmap_bg = ListedColormap(new_colors_bg)

            # Generate hexbin plot for the background (lower alpha for visibility)
            ax.hexbin(compare_dataframe[param1], compare_dataframe[param2], gridsize=bins, 
                    cmap=masked_cmap_bg, edgecolors='none', alpha=0.5, extent=extent)

        # Plot main dataset
        if single_color:
            # Use a single solid color with masked zero counts
            cmap_main = plt.get_cmap(single_color)
            new_colors_main = cmap_main(np.ones(cmap_main.N))  # Force single color
            new_colors_main[0] = [1, 1, 1, 0]  # Make zero-count tiles white/transparent
            masked_cmap_main = ListedColormap(new_colors_main)
            hb = ax.hexbin(dataframe[param1], dataframe[param2], gridsize=bins, cmap=masked_cmap_main, 
                        edgecolors='none', alpha=0.8, extent=extent)  # Increase alpha to stand out
        else:
            # Use `viridis` gradient colormap
            hb = ax.hexbin(dataframe[param1], dataframe[param2], gridsize=bins, cmap='viridis', 
                        edgecolors='none', extent=extent)

        # Extract hexagon bin counts for normalization
        counts = hb.get_array()
        max_count = counts.max()

        # Modify colormap to make zero-count tiles white
        cmap = plt.get_cmap('viridis')
        new_colors = cmap(np.linspace(0, 1, cmap.N))
        new_colors[0] = [1, 1, 1, 0]  # Set the lowest value to fully transparent (RGBA)
        new_cmap = ListedColormap(new_colors)

        # Replot with the adjusted colormap
        ax.clear()

        # Re-add background hexbin with **masked colormap**
        for i, compare_ps in enumerate(compare_ps_list):
            compare_dataframe = load_data(compare_ps.name, path=compare_ps.path_data, as_dict=False)
            ax.hexbin(compare_dataframe[param1], compare_dataframe[param2], gridsize=bins, 
                    cmap=masked_cmap_bg, edgecolors='none', alpha=0.5, extent=extent)

        # Re-add the main dataset
        if single_color:
            hb = ax.hexbin(dataframe[param1], dataframe[param2], gridsize=bins, cmap=masked_cmap_main, 
                        edgecolors='none', alpha=1.0, extent=extent)  # Keep it visible
        else:
            hb = ax.hexbin(dataframe[param1], dataframe[param2], gridsize=bins, cmap=new_cmap, 
                        edgecolors='none', extent=extent)

        # Add colorbar **only if using `viridis`**
        if not single_color:
            cbar = plt.colorbar(hb, label='Normalized Count')
            cbar.set_ticks([0, 0.5 * max_count, max_count])
            cbar.set_ticklabels(["0", "50%", "100%"])

        # Apply user-defined tick labels
        if "set_xticklabels" in kwargs:
            plt.xticks(kwargs["set_xticklabels"])
        if "set_yticklabels" in kwargs:
            plt.yticks(kwargs["set_yticklabels"])

        # Ensure hexagons are not distorted
        ax.set_aspect('equal')

        # Labels and title
        plt.xlabel(f"${param1}$")
        plt.ylabel(f"${param2}$")
        plt.title(f"${param1}$ vs ${param2}$")

        # Save and display
        plt.savefig(os.path.join(self.path_plots, f"histogram_{param1}_vs_{param2}.png"))
        plt.show()

        
    def plot_bin2D_two_params(self, param1_index: int, param2_index: int, bins=30, filename: str = None, 
                            compare_ps_list=[], compare_ps_list_colors=["grey"], single_color=False, **kwargs):
        from matplotlib.colors import ListedColormap
        
        #from matplotlib.colors import ListedColormap

        # Determine file name and load main dataset
        if filename is None:
            filename = self.name
        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        # Assume dataframe.keys() returns a list-like object
        params = list(dataframe.keys())
        param1 = params[param1_index]
        param2 = params[param2_index]
        
        # Extract main data values
        x = dataframe[param1]
        y = dataframe[param2]

        # Create figure and axis (using object-oriented API)
        fig, ax = plt.subplots()

        # Apply user-defined tick settings if provided
        if "set_xticks" in kwargs:
            ax.set_xticks(kwargs["set_xticks"])
        if "set_yticks" in kwargs:
            ax.set_yticks(kwargs["set_yticks"])
            
        y_min, y_max = ax.get_ylim()
        x_min, x_max = ax.get_xlim()
        extent = [x_min, x_max, y_min, y_max]
        
        # Define common bin edges using main dataset's min and max
        if isinstance(bins, int):
            #xedges = np.linspace(np.min(x), np.max(x), bins + 1)
            #yedges = np.linspace(np.min(y), np.max(y), bins + 1)
            xedges = np.linspace(x_min, x_max, bins + 1)
            yedges = np.linspace(y_min, y_max, bins + 1)
        else:
            # If bins is already a pair of arrays (or similar), unpack accordingly
            xedges, yedges = bins

        # Plot background dataset(s) with a masked single color, using common bin edges
        for i, compare_ps in enumerate(compare_ps_list):
            compare_dataframe = load_data(compare_ps.name, path=compare_ps.path_data, as_dict=False)
            # Use same parameter names as main data to ensure alignment
            x_bg = compare_dataframe[param1]
            y_bg = compare_dataframe[param2]
            
            # Compute 2D histogram using common bin edges
            counts_bg, _, _ = np.histogram2d(x_bg, y_bg, bins=[xedges, yedges])
            # Mask zero counts
            counts_bg = np.ma.masked_where(counts_bg == 0, counts_bg)
            
            # Create a colormap from the provided color string if needed
            color = compare_ps_list_colors[i] if i < len(compare_ps_list_colors) else "grey"
            cmap_bg = ListedColormap([color])
            
            img = ax.imshow(counts_bg.T, extent=extent, origin='lower', cmap=cmap_bg, alpha=1.0, aspect='auto')

        # Compute histogram for the main dataset using the same bin edges
        counts, _, _ = np.histogram2d(x, y, bins=[xedges, yedges])
        counts = np.ma.masked_where(counts == 0, counts)

        if single_color:
            # For a single solid color, create a ListedColormap
            if "color" in kwargs:
                color = kwargs["color"]
            else:
                color = "blue"
            cmap_main = ListedColormap([color])
        else:
            cmap_main = "viridis"
            
        img = ax.imshow(counts.T, extent=extent, origin='lower', cmap=cmap_main, aspect='auto')
        
        # Add a colorbar only when using viridis
        if not single_color:
            cbar = plt.colorbar(img, ax=ax, label='Normalized Count')
            cbar.set_ticks([0, 0.5 * counts.max(), counts.max()])
            cbar.set_ticklabels(["0", "0.5", "1"])

        # Apply user-defined tick labels if provided
        if "set_xticklabels" in kwargs:
            ax.set_xticklabels(kwargs["set_xticklabels"])
        if "set_yticklabels" in kwargs:
            ax.set_yticklabels(kwargs["set_yticklabels"])

        # Labels and title
        ax.set_xlabel(f"${param1}$")
        ax.set_ylabel(f"${param2}$")
        ax.set_title(f"${param1}$ vs ${param2}$")
        ax.set_aspect('equal')

        # Ensure that the directory exists before saving
        if not os.path.exists(self.path_plots):
            os.makedirs(self.path_plots)
        save_path = os.path.join(self.path_plots, f"histogram2D_{param1}_vs_{param2}.png")
        plt.savefig(save_path)
        plt.show()
        
        
            
    def plot_scatter_two_params(self, param1_index:int, param2_index:int, filename:str=None, **kwargs):
        plt.figure()
        if filename == None:
            filename = self.name
            
        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        param1 = dataframe.keys()[param1_index]
        param2 = dataframe.keys()[param2_index]
        plt.scatter(dataframe[param1], dataframe[param2], marker=".")
        plt.xlabel(f"${param1}$")
        plt.ylabel(f"${param2}$")
        plt.title(f"${param1}$ vs ${param2}$")
        plt.savefig(os.path.join(self.path_plots, f"scatter_{param1}_vs_{param2}.png"))
        plt.show()
        
    def plot_params_effective(self, bins=30, filename:str=None):
        plt.figure()
        if filename == None:
            filename = self.name
            
        dataframe = load_data(filename, path=self.path_data, as_dict=False)
        params = dataframe.keys()[0:14]
        params_CT = dataframe.keys()[14:28]
        
        for i in range(14):
            plt.hist(dataframe[params[i]], bins=bins, alpha=0.5, label=params[i])
            plt.hist(dataframe[params_CT[i]], bins=bins, alpha=0.5, label=params_CT[i])
            plt.hist(dataframe[params[i]], bins=bins, alpha=0.5, label=params[i])

        plt.legend()
        plt.xlabel("Parameter values")
        plt.ylabel("Frequency")
        plt.title("Parameter values")
        plt.savefig(os.path.join(self.path_plots, f"histogram_params.png"))
        plt.show()
    

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
