import os
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
import multiprocessing
#import warnings

from ..utils.methods_data import save_data, load_data
from .Model2HDMCalculator import Model2HDMCalculator


class Model2HDMPlotter:
    def __init__(self, MC_list: list, **kwargs):
        self.MC_list = MC_list

        # Create separate directories for data and figures.
        self.path_data = os.path.join(MC_list[0].model.path_root, "plots_data")
        os.makedirs(self.path_data, exist_ok=True)

        self.path_plots = os.path.join(MC_list[0].model.path_root, "plots_figures")
        os.makedirs(self.path_plots, exist_ok=True)

        self.v = 246

        # Default configuration for potential and masses.
        # Note: Both a top-level "line_width" and individual line_width settings
        # in potential_config and masses_config have been added.
        self.default_config = {
            "figsize": (20, 20),
            "line_width": 2,
            "grid": True,
            "dpi": 500,
            "fontsize": 16,
            "title_fontsize": 20,
            "xlabel_fontsize": 18,
            "ylabel_fontsize": 18,
            "tick_labelsize": 18,
            "legend": True,
            "legend_loc": "upper left",
            "legend_fontsize": 10,
            # Potential-specific configuration
            "potential_config": {
                "xlabel": r"$\omega$",
                "ylabel": r"Potential $V(\omega)$",
                "title": "Potential Plot",
                "line_width": 1.5,  # Use this line width for potential plots.
                "color_0": "blue",
                "color_1": "red",
                "alpha_0": 1.0,
                "alpha_1": 1.0,
                "legend_0": True,
                "legend_1": True,
                "line_style_0": "-",
                "line_style_1": "-",
                "potential_fig_filename": "potential_plot.png",
                "Xrange": None,
                "Yrange": None,
                "separated": False
            },
            # Masses-specific configuration
            "masses_config": {
                "xlabel": r"$\omega$",
                "ylabel": "Masses",
                "title": "Mass Plot",
                "line_width": 1.5,  # Use this line width for mass plots.
                "color_0": ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'],
                "color_1": ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray'],
                "alpha_0": 0.4,
                "alpha_1": 1.0,
                "legend_0": False,
                "legend_1": True,
                "line_style_mass_0": ["-", "-", "-", "-", "--", "--", "--", "--"],
                "line_style_mass_1": ["-", "-", "-", "-", "--", "--", "--", "--"],
                "mass_fig_filename": "mass_plot.png",
                "Xrange": None,
                "Yrange": None,
                "separated": False
            }
        }

        # Merge in any user-specified configuration.
        self.plot_config = self.default_config.copy()
        if "plot_config" in kwargs:
            for key, value in kwargs["plot_config"].items():
                self.plot_config[key] = value

    # ===================== Configuration Functions =====================

    def set_plot_config(self, **kwargs):
        self.plot_config.update(kwargs)

    def set_plot_config_potential(self, **kwargs):
        self.plot_config["potential_config"].update(kwargs)

    def set_plot_config_masses(self, **kwargs):
        self.plot_config["masses_config"].update(kwargs)

    # ===================== Helper Functions =====================

    def _get_subplot_grid(self, num_plots):
        if num_plots == 1:
            return (1, 1)
        elif num_plots == 2:
            return (1, 2)
        elif num_plots in [3, 4]:
            return (2, 2)
        elif num_plots in [5, 6]:
            return (2, 3)
        elif num_plots in [8]:
            return (4, 2)
        else:
            return (3, 3)

    def _model_filename(self, base_filename, model_index, level=None):
        """
        Generate a filename for a specific model's data.
        If level is provided, include it in the filename.
        """
        name, ext = os.path.splitext(base_filename)
        if level is not None:
            return f"{name}_model_{model_index}_level_{level}{ext}"
        else:
            return f"{name}_model_{model_index}{ext}"

    # ----------------- Data Calculation Functions -----------------

    def _calculate_potential_data(self, args):
        """
        Compute potential data for one model.
        args: (N, free_bgfield, Xrange, MC, levels)
        Returns a dictionary possibly containing both "V0" and "V1".
        """
        N, free_bgfield, Xrange, MC, levels = args
        data = {}
        if 0 in levels:
            X, V0 = MC.calculate_Vdata2D_level0(N, free_bgfield, Xrange)
            data["X"] = X
            data["V0"] = V0
        if 1 in levels:
            X, V1 = MC.calculate_Vdata2D_level1(N, free_bgfield, Xrange)
            if "X" not in data:
                data["X"] = X
            data["V1"] = V1
        return data

    def _calculate_masses_data(self, args):
        """
        Compute mass data for one model for a single level.
        Expects: (N, Xrange, free_bgfield, MC, level)
        Returns a dictionary with keys "X", "Y1", "Y2", ..., "Y8".
        """
        N, Xrange, free_bgfield, MC, level = args
        if level == 0:
            X, Y = MC.calculate_massdata2D_level0(N, Xrange, free_bgfield, sorting=True)
        elif level == 1:
            X, Y = MC.calculate_massdata2D_level1(N, Xrange, free_bgfield, sorting=True)
        else:
            raise ValueError("Level must be 0 or 1 for mass data.")
        result = {"X": np.ravel(X)}
        for i, Y_i in enumerate(Y):
            result[f"Y{i+1}"] = np.ravel(Y_i)
        return result

    # ===================== Potential Data Processing =====================

    def calculate_plot_potential(self, N, free_bgfield, Xrange, levels, base_filename="potential_data",
                                   overwrite=False, multiprocessing_enabled=False):
        """
        Compute potential data for all models and for each requested level.
        Data files are stored under self.path_data.
        Returns a dictionary mapping keys "model_{i}_level_{l}" to their data.
        If multiprocessing_enabled is True, computation is done in parallel.
        """
        results_dict = {}
        indices_to_compute = []

        # Check which models need computation.
        for i, MC in enumerate(self.MC_list):
            for level in levels:
                model_filename = self._model_filename(base_filename, i+1, level)
                file_path = os.path.join(self.path_data, model_filename)
                if overwrite or not os.path.exists(file_path):
                    indices_to_compute.append(i)
                    break

        if indices_to_compute:
            args_to_compute = [(N, free_bgfield, Xrange, self.MC_list[i], levels)
                               for i in indices_to_compute]
            # Print whether we're computing in parallel or serially.
            if multiprocessing_enabled:
                print("Computing potential data in parallel...")
                with multiprocessing.Pool(processes=min(len(args_to_compute), multiprocessing.cpu_count())) as pool:
                    computed_results = pool.map(self._calculate_potential_data, args_to_compute)
            else:
                print("Computing potential data serially...")
                computed_results = [self._calculate_potential_data(args) for args in args_to_compute]
            # Save computed results to disk.
            for idx, data in zip(indices_to_compute, computed_results):
                for level in levels:
                    model_filename = self._model_filename(base_filename, idx+1, level)
                    level_data = {"X": data["X"]}
                    if level == 0 and "V0" in data:
                        level_data["V0"] = data["V0"]
                    elif level == 1 and "V1" in data:
                        level_data["V1"] = data["V1"]
                    save_data(level_data, model_filename, self.path_data, merge=False, show_size=False)
                    results_dict[f"model_{idx+1}_level_{level}"] = level_data

        # Load data for models that did not need recomputation.
        for i in range(len(self.MC_list)):
            for level in levels:
                key = f"model_{i+1}_level_{level}"
                if key not in results_dict:
                    model_filename = self._model_filename(base_filename, i+1, level)
                    results_dict[key] = load_data(model_filename, self.path_data)

        return results_dict

    def plot_potential(self, N, free_bgfield, Xrange, levels=[0, 1],
                       multiprocessing_enabled=False, overwrite=False, base_filename="potential_data",
                       save=True):
        """
        Plot the potential for each model on a single subplot.
        Both level 0 and level 1 data (if available) are overlaid.
        Only the leftmost subplot displays the y-axis label.
        """
        # Validate inputs.
        if not isinstance(levels, list) or any(l not in [0, 1] for l in levels):
            raise ValueError("levels must be a list containing 0, 1, or both.")
        if not self.MC_list:
            raise ValueError("No models provided for plotting.")

        base_filename = base_filename if base_filename.endswith(".csv") else base_filename + ".csv"
        potential_results = self.calculate_plot_potential(N, free_bgfield, Xrange, levels,
                                                           base_filename=base_filename,
                                                           overwrite=overwrite,
                                                           multiprocessing_enabled=multiprocessing_enabled)

        # Determine global X limits.
        all_X = []
        for key, data in potential_results.items():
            all_X.extend(np.array(data["X"]).tolist())
        global_x_min = min(all_X)
        global_x_max = max(all_X)

        num_models = len(self.MC_list)
        rows, cols = self._get_subplot_grid(num_models)
        fig, axes = plt.subplots(rows, cols, figsize=self.plot_config["figsize"])
        axes = np.array(axes).flatten()

        plt.subplots_adjust(wspace=0, hspace=0)
        pot_cfg = self.plot_config["potential_config"]

        # Plot each model's potential.
        for i in range(num_models):
            ax = axes[i]
            key0 = f"model_{i+1}_level_0"
            if key0 in potential_results and "V0" in potential_results[key0]:
                data0 = potential_results[key0]
                X0 = np.array(data0["X"])
                ax.plot(X0/self.v, np.array(data0["V0"])/self.v**4,
                        linestyle=pot_cfg["line_style_0"],
                        linewidth=pot_cfg["line_width"],
                        color=pot_cfg["color_0"],
                        alpha=pot_cfg["alpha_0"],
                        label="Level 0")
            key1 = f"model_{i+1}_level_1"
            if key1 in potential_results and "V1" in potential_results[key1]:
                data1 = potential_results[key1]
                X1 = np.array(data1["X"])
                ax.plot(X1/self.v, np.array(data1["V1"])/self.v**4,
                        linestyle=pot_cfg["line_style_1"],
                        linewidth=pot_cfg["line_width"],
                        color=pot_cfg["color_1"],
                        alpha=pot_cfg["alpha_1"],
                        label="Level 1")
            ax.set_xlabel(pot_cfg["xlabel"], fontsize=self.plot_config["xlabel_fontsize"])
            # Only the leftmost subplots show the y-axis label.
            if i % cols == 0:
                ax.set_ylabel(pot_cfg["ylabel"], fontsize=self.plot_config["ylabel_fontsize"])
            ax.set_title(f"{pot_cfg['title']} {i+1}", fontsize=self.plot_config["title_fontsize"])
            if self.plot_config["legend"]:
                ax.legend(loc=self.plot_config["legend_loc"], fontsize=self.plot_config["legend_fontsize"])
            ax.grid(self.plot_config["grid"])
            ax.set_xlim(global_x_min/self.v, global_x_max/self.v)

        if save:
            print("Saving potential plot...")
            fig.savefig(os.path.join(self.path_plots, pot_cfg["potential_fig_filename"]),
                        dpi=self.plot_config["dpi"])
        plt.show()

    # ===================== Mass Data Processing =====================

    def calculate_plot_masses(self, N, Xrange, free_bgfield, levels, base_filename="mass_data", overwrite=False,
                              multiprocessing_enabled=False):
        """
        Compute mass data for all models and for each requested level.
        Data files are stored under self.path_data.
        Returns a dictionary mapping keys "model_{i}_level_{l}" to their mass data.
        If multiprocessing_enabled is True, uses parallel computation.
        """
        results_dict = {}
        indices_to_compute = []
        args_to_compute = []
        for i, MC in enumerate(self.MC_list):
            for level in levels:
                model_filename = self._model_filename(base_filename, i+1, level)
                file_path = os.path.join(self.path_data, model_filename)
                if overwrite or not os.path.exists(file_path):
                    indices_to_compute.append((i, level))
                    args_to_compute.append((N, Xrange, free_bgfield, MC, level))
                else:
                    results_dict[f"model_{i+1}_level_{level}"] = load_data(model_filename, self.path_data)

        if args_to_compute:
            if multiprocessing_enabled:
                print("Computing mass data in parallel...")
                with multiprocessing.Pool(processes=min(len(args_to_compute), multiprocessing.cpu_count())) as pool:
                    computed_results = pool.map(self._calculate_masses_data, args_to_compute)
            else:
                print("Computing mass data serially...")
                computed_results = [self._calculate_masses_data(args) for args in args_to_compute]
            for idx, (i, level) in enumerate(indices_to_compute):
                model_filename = self._model_filename(base_filename, i+1, level)
                data = computed_results[idx]
                save_data(data, model_filename, self.path_data, merge=False, show_size=False)
                results_dict[f"model_{i+1}_level_{level}"] = data
        return results_dict

    def plot_masses(self, N, Xrange, free_bgfield, levels=[0, 1],
                    multiprocessing_enabled=False, overwrite=False, base_filename="mass_data",
                    save=True):
        """
        Plot the mass data for each model on a single subplot.
        Both level 0 and level 1 data (if available) are overlaid.
        Each mass eigenvalue is labeled as "m_i" if the corresponding legend flag is True.
        Only the leftmost subplot displays the y-axis label.
        """
        if not isinstance(levels, list) or any(l not in [0, 1] for l in levels):
            raise ValueError("levels must be a list containing 0, 1, or both.")
        if not self.MC_list:
            raise ValueError("No models provided for plotting.")

        base_filename = base_filename if base_filename.endswith(".csv") else base_filename + ".csv"
        mass_results = self.calculate_plot_masses(N, Xrange, free_bgfield, levels,
                                                  base_filename=base_filename, overwrite=overwrite,
                                                  multiprocessing_enabled=multiprocessing_enabled)

        # Determine global X limits.
        all_X = []
        for key, data in mass_results.items():
            all_X.extend(np.array(data["X"]).tolist())
        global_x_min = min(all_X)
        global_x_max = max(all_X)

        num_models = len(self.MC_list)
        rows, cols = self._get_subplot_grid(num_models)
        fig, axes = plt.subplots(rows, cols, figsize=self.plot_config["figsize"], sharey=True)
        axes = np.array(axes).flatten()
        plt.subplots_adjust(wspace=0, hspace=0)
        mass_cfg = self.plot_config["masses_config"]

        for i in range(num_models):
            ax = axes[i]
            colors = mass_cfg["color_0"]
            line_styles = mass_cfg["line_style_mass_0"]
            # Plot level 0 data if available.
            key0 = f"model_{i+1}_level_0"
            if key0 in mass_results:
                data0 = mass_results[key0]
                X0 = np.array(data0["X"])
                for j in range(8):
                    Yj = np.array(data0[f"Y{j+1}"])
                    label0 = f"m_{j+1}" if mass_cfg.get("legend_0", False) else None
                    ax.plot(X0/self.v, Yj,
                            linestyle=line_styles[j],
                            linewidth=mass_cfg["line_width"],
                            color=colors[j],
                            alpha=mass_cfg["alpha_0"],
                            label=label0)
            # Plot level 1 data if available.
            key1 = f"model_{i+1}_level_1"
            if key1 in mass_results:
                data1 = mass_results[key1]
                X1 = np.array(data1["X"])
                for j in range(8):
                    Yj = np.array(data1[f"Y{j+1}"])
                    label1 = f"m_{j+1}" if mass_cfg.get("legend_1", False) else None
                    ax.plot(X1/self.v, Yj,
                            linestyle=line_styles[j],
                            linewidth=mass_cfg["line_width"],
                            color=colors[j],
                            alpha=mass_cfg["alpha_1"],
                            label=label1)
            ax.set_xlabel(mass_cfg["xlabel"], fontsize=self.plot_config["xlabel_fontsize"])
            # Only set y-label for leftmost subplots.
            if i % cols == 0:
                ax.set_ylabel(mass_cfg["ylabel"], fontsize=self.plot_config["ylabel_fontsize"])
            ax.set_title(f"{mass_cfg['title']} {i+1}", fontsize=self.plot_config["title_fontsize"])
            if self.plot_config["legend"]:
                ax.legend(loc=self.plot_config["legend_loc"], fontsize=self.plot_config["legend_fontsize"])
            ax.grid(self.plot_config["grid"])
            ax.set_xlim(global_x_min/self.v, global_x_max/self.v)

        if save:
            print("Saving mass plot...")
            fig.savefig(os.path.join(self.path_plots, mass_cfg["mass_fig_filename"]),
                        dpi=self.plot_config["dpi"])
        plt.show()
