
# Packages
import numpy as np
import sympy as sp
import os
import time 
import pandas as pd

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *
from ..utils import constants as const

#
from ..ModelCalculators.Model2HDMCalculator import Model2HDMCalculator


#################### Constraints ####################

# Positivity constraint
def constraint_positivity(ps:object, dataframe) -> dict:
    

    # Loop over points and check the constraint condition
    remove_points_index = []
    j=0
    print("Checking positivity constraint...")
    for i, row in dataframe.iterrows():
        j += 1
        print("Progress: ", j, "/", len(dataframe), end="\r")
        point = row.tolist()
        l1 = point[2]
        l2 = point[3]
        l3 = point[4]
        l4 = point[5]
        rl5 = point[8]
        il5 = point[9]
        l5 = rl5 + 1j*il5
        #print(l1, l2, l3, l4, l5)
        if not l1 > 0 or not l2 > 0:
            remove_points_index.append(i)
        elif not l3 + np.sqrt(l1*l2) > 0:
            remove_points_index.append(i)
        elif not l3 + l4 - sp.Abs(l5) + np.sqrt(l1*l2) > 0:
            remove_points_index.append(i)
    
    print()
    #print(remove_points_index)
    # Remove points
    new_data = {}
    for key in dataframe.keys():
        new_data[key] = dataframe[key].drop(remove_points_index)
        
    return pd.DataFrame(new_data)

# Unitarity constraint
def constraint_unitarity(ps:object, dataframe) -> dict:

    # Loop over points and check the constraint condition
    remove_points_index = []
    j=0
    print("Checking unitarity constraint...")
    for i, row in dataframe.iterrows():
        j += 1
        print("Progress: ", j, "/", len(dataframe), end="\r")
        point = row.tolist()
        l1 = point[2]
        l2 = point[3]
        l3 = point[4]
        l4 = point[5]
        rl5 = point[8]
        il5 = point[9]
        l5 = rl5 + 1j*il5
        rl6 = point[10]
        il6 = point[11]
        l6 = rl6 + 1j*il6
        rl7 = point[12]
        il7 = point[13]
        l7 = rl7 + 1j*il7
        coeff = 1/(8*np.pi)
        S00 = coeff * np.array([[3*l1, 2*l3+l4, 3*l6, 3*np.conjugate(l6)], 
                                [2*l3+l4, 3*l2, 3*l7, 3*np.conjugate(l7)], 
                                [3*np.conjugate(l6), 3*np.conjugate(l7), l3+2*l4, 3*np.conjugate(l5)], 
                                [3*l6, 3*l7, 3*l5, l3+2*l4]])
        S01 = coeff * np.array([[l1, l4, l6, np.conjugate(l6)], 
                                [l4, l2, l7, np.conjugate(l7)],
                                [np.conjugate(l6), np.conjugate(l7), l3, np.conjugate(l5)], 
                                [l6, l7, l5, l3]])
        S20 = coeff * np.array([[(l3-l4)]])
        S21 = coeff * np.array([[l1, l5, np.sqrt(2)*l6],
                                [np.conjugate(l5), l2, np.sqrt(2)*np.conjugate(l7)],
                                [np.conjugate(l6), np.sqrt(2)*l7, l3+l4]])
        
        # Eigenvalues
        eigvals_S00 = np.linalg.eigvals(S00)
        eigvals_S01 = np.linalg.eigvals(S01)
        eigvals_S20 = np.linalg.eigvals(S20)
        eigvals_S21 = np.linalg.eigvals(S21)
        
        #print([np.abs(res) for res in eigvals_S00])
        #print([np.abs(res) for res in eigvals_S01])
        #print("test", [np.abs(res) for res in eigvals_S20])
        #print([np.abs(res) for res in eigvals_S21])

        # check |eval|<=1
        #kolla ivanov o ginzgurg
        cond1 = np.all(np.abs(eigvals_S00) <= 2)
        cond2 = np.all(np.abs(eigvals_S01) <= 2)
        cond3 = np.all(np.abs(eigvals_S20) <= 2)
        cond4 = np.all(np.abs(eigvals_S21) <= 2)
        
        if not cond1 or not cond2 or not cond3 or not cond4:
            remove_points_index.append(i)

    print()
    
    # Remove points
    new_data = {}
    for key in dataframe.keys():
        new_data[key] = dataframe[key].drop(remove_points_index)
        
    return pd.DataFrame(new_data)

# Global minimum constraint
def constraint_global_minimum(ps:object, dataframe) -> dict:

    # Veff(v) < Veff(v=0)
    
    VEVs_indexrange = [14,14]
    # Loop over points and check the constraint condition
    remove_points_index = []
    subs_fields_to_zero = {symb:0 for symb in ps.model.fields}
    length = dataframe.shape[0]
    j=0
    print("Checking global minimum constraint...")
    for i, row in dataframe.iterrows():
        j += 1
        print("Progress: ", j, "/", length, " | ", end="\r")
        point = row.tolist()
        #print(point)
        VEVs = point[14], # check index range later
        
        subs_VEV_values = {symb:val for symb, val in zip(ps.model.VEVs, VEVs)}
        V0_param_values = point[0:14]
        MC = Model2HDMCalculator(ps.model, V0_params_values=V0_param_values, subs_VEV_values=subs_VEV_values)
        
        subs_omega_values_0 = {symb:val for symb, val in zip(ps.model.bgfields, [0 for _ in range(len(VEVs))]) } 
        subs_omega_values_v = {symb:val for symb, val in zip(ps.model.bgfields, VEVs)}
        MC.assign_counterterm_values()
        Vcw_0 = MC.calculate_VCW_point(subs_bgfield_values=subs_omega_values_0)
        Vcw_v = MC.calculate_VCW_point(subs_bgfield_values=subs_omega_values_v)
        V0_0 = MC.V0_simplified.subs(subs_omega_values_0)
        V0_v = MC.V0_simplified.subs(subs_omega_values_v)
        VCT_0 = MC.VCT_simplified.subs(subs_omega_values_0)
        VCT_v = MC.VCT_simplified.subs(subs_omega_values_v)
        Veff_0 = V0_0 + VCT_0 + Vcw_0
        Veff_v = V0_v + VCT_v + Vcw_v
        Veff_0 = Veff_0.subs(subs_fields_to_zero).evalf()
        Veff_v = Veff_v.subs(subs_fields_to_zero).evalf()
        
        
        cond = Veff_v < Veff_0
        if not cond:
            remove_points_index.append(i)
    
    print()
        
    # Remove points

    new_data = {}
    for key in dataframe.keys():
        new_data[key] = dataframe[key].drop(remove_points_index)
    
    return pd.DataFrame(new_data)
        

def constraint_oblique(params) -> bool:
    """
    Oblique parameters constraints (constraints on the hi masses)
    """
    pass
