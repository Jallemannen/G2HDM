
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

from . import parameter_constraints as constraints

# ===================== General =====================

def load_ps(model:str, ps:str, path:str=None, project_root = None):
    """Need to be in Saved folder. Returns the loaded model, given its folder, if it exists, else it returns None"""
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), ''))
    if path == None:
        path = os.path.join(project_root, "saved_models", f"{model}", "parameter_search", f"{ps}", "ParamSearch_data", "ps.pkl")

    return load_pkl(path)    

# ===================== Plotting =====================

def plot_ps_points(ps:object, data_name:str, colors:list, style="bin"):

    if style == "bin":
        pass
    elif style == "scatter":
        pass


def plot_ps_histogram(ps:object, data_name:str, colors:list):
    pass

    
def plot_compare_ps_data(ps_list:list, data_name:str, colors:list):
    pass


def get_ps_values2D(ps:object, *args):
    """args as indicies we want to extract from the data"""
    L = []
    return L

