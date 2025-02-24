import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import multiprocessing
import warnings

from .Model2HDMCalculator import Model2HDMCalculator


class Model2HDMPlotter:
    
    def __init__(self, MC_list:list, **kwargs):
        self.MC_list = MC_list
        
        self.v = 246
        # Default plot configuration
        default_config = {
            "figsize": (10, 6),
            "line_style": "-",
            "line_width": 2,
            "color": None,  # Automatically assigns colors if None
            "grid": True,
            "xlabel": "X",
            "ylabel": "Potential V(X)",
            "title": "Potential Plot",
            "alpha_0": 1.0,
            "alpha_1": 1.0,
            "line_style_0": "-",
            "line_style_1": "-"
        }
        
        # Override defaults with user-specified options
        self.plot_config = {**default_config, **kwargs.get("plot_config", {})}
        
    def set_plot_config(self, **kwargs):
        self.plot_config.update(kwargs)
        
    ############### Helper functions ###############
        
    def _get_subplot_grid(self, num_plots):
        if num_plots == 1:
            return (1, 1)
        elif num_plots == 2:
            return (1, 2)
        elif num_plots in [3, 4]:
            return (2, 2)
        elif num_plots in [5, 6]:
            return (2, 3)
        else:
            return (3, 3)
    
    def _plot_single_potential_from_data(self, data, ax, levels, alphas, index=None):
        """Plot potential using pre-computed data on a given axis."""
        if 0 in levels and "V0" in data:
            X, V0 = data["X"], data["V0"]
            ax.plot(X/self.v, V0/self.v**4,
                    linestyle=self.plot_config["line_style_0"],
                    linewidth=self.plot_config["line_width"],
                    color='blue',  # or use self.plot_config["color"] if defined
                    alpha=alphas[0],
                    label="Level 0")
        if 1 in levels and "V1" in data:
            X, V1 = data["X"], data["V1"]
            ax.plot(X/self.v, V1/self.v**4,
                    linestyle=self.plot_config["line_style_1"],
                    linewidth=self.plot_config["line_width"],
                    color='red',  # or use self.plot_config["color"] if defined
                    alpha=alphas[1],
                    label="Level 1")
                    
        ax.set_xlabel(self.plot_config["xlabel"])
        ax.set_ylabel(self.plot_config["ylabel"])
        if index is not None:
            ax.set_title(f"{self.plot_config['title']} {index}")
        else:
            ax.set_title(self.plot_config["title"])
        ax.legend()
        ax.grid(self.plot_config["grid"])
        return ax

    def _calculate_potential_data(self, args):
        """Helper function to compute potential data for one model.
           This function is designed to be used with multiprocessing."""
        N, free_bgfield, Xrange, MC, levels = args
        data = {}
        # It is assumed that both methods return the same X array.
        if 0 in levels:
            X, V0 = MC.calculate_Vdata2D_level0(N, free_bgfield, Xrange)
            data["X"] = X
            data["V0"] = V0
        if 1 in levels:
            X, V1 = MC.calculate_Vdata2D_level1(N, free_bgfield, Xrange)
            # If X wasn’t computed above, assign it here.
            if "X" not in data:
                data["X"] = X
            data["V1"] = V1
        return data


    ###############################################

    def calculate_plot_potential(self, N, free_bgfield, Xrange, levels):
        """Compute potential data for all models using multiprocessing."""
        
        args_list = [(N, free_bgfield, Xrange, MC, levels) for MC in self.MC_list]
        
        # Check if models are renormalized
        for i, MC in enumerate(self.MC_list):
            if MC.is_renormalized == None:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    #warnings.simplefilter("always")
                    warnings.filterwarnings("ignore", category=UserWarning)
                    MC.assign_counterterm_values()
                #MC.assign_counterterm_values()
            if MC.is_renormalized == False:
                print(f"\nWarning: model with parameter {i+1} is not renormalized. Results may be incorrect.\n")
        
        with multiprocessing.Pool(processes=min(len(self.MC_list), multiprocessing.cpu_count())) as pool:
            results = pool.map(self._calculate_potential_data, args_list)
        return results
        
    def plot_potential(self, N, free_bgfield, Xrange, levels=[0, 1], multiprocessing_enabled=False):
        if not isinstance(levels, list) or any(l not in [0, 1] for l in levels):
            raise ValueError("levels must be a list containing 0, 1, or both.")

        num_plots = len(self.MC_list)
        if num_plots == 0:
            raise ValueError("No models provided for plotting.")
        
        rows, cols = self._get_subplot_grid(num_plots)
        fig, axes = plt.subplots(rows, cols, figsize=self.plot_config["figsize"])
        axes = np.array(axes).flatten()  # flatten for easy indexing
        
        # Set alpha values
        alpha_0 = self.plot_config["alpha_0"] if (0 in levels and 1 in levels) else 1.0
        alpha_1 = self.plot_config["alpha_1"]
        alphas = [alpha_0, alpha_1]
        
        if multiprocessing_enabled:
            # Compute potential data in parallel.
            potential_results = self.calculate_plot_potential(N, free_bgfield, Xrange, levels)
        else:
            # Compute potential data sequentially.
            potential_results = self._calculate_potential_data(N, free_bgfield, Xrange, levels)
        
        # Plot in the main process.
        for i, data in enumerate(potential_results):
            self._plot_single_potential_from_data(data, axes[i], levels, alphas, index=i+1)
        
        plt.tight_layout()
        plt.show()
    

            
        
            
        