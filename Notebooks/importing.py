# This file is used to import the project root directory to the python path

# Importing the necessary libraries
import sys
import os
import importlib
import inspect
from IPython import get_ipython

#################### Methods ####################

def declare_project_root():
    project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    return project_root

def in_jupyter():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True
        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False
        else:
            # Other type (possibly unknown)
            return False
    except NameError:
        # Probably standard Python interpreter
        return False

#################### Runs once on import ####################    

# Declare the project root directory    
project_root = declare_project_root()

#if in_jupyter():
#    print("Running in Jupyter Notebook/Lab")
#else:
#    print("Running in a terminal or non-IPython environment")


#print(project_root, sys.path)
import src.utils.constants as const

# Default packages
import sys
import os
import sympy as sp 
import numpy as np  
import matplotlib.pyplot as plt  
from IPython.display import display, Math  
import time

# Importing the MultiProcessing modules
#from src.MultiProcessing.class_MultiProcessing import MultiprocessingManager
#from src.MultiProcessing.methods_MultiProcessing import *

# Importing the Model2HDM modules
from src.Model2HDM.class_Model2HDM import Model2HDM
from src.Model2HDM.methods_Model2HDM import * 
from src.Model2HDM.ParameterSearch import * 
from src.Model2HDM.constraints_ParameterSearch import * 

from src.Model2HDM.ModelDataCalculator import ModelDataCalculator

# Importing the Utils modules
from src.utils.methods_math import *
from src.utils.methods_data import *
from src.utils.methods_general import *
