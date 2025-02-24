
# Packages
import numpy as np
import sympy as sp
import os
import time 

# Custom Packages, Utils
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *
from ..utils import constants as const

#
from ..MultiProcessing.class_MultiProcessing import * 
from ..MultiProcessing.methods_MultiProcessing import * 


#################### Constraints ####################

def constraint_positivity(params) -> bool:
    pass

def constraint_oblique(params) -> bool:
    """
    Oblique parameters constraints (constraints on the hi masses)
    """
    pass
