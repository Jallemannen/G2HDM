# Runs once when the parent folder is imported

# Packages
from .Model2HDM import *
from .ModelCalculators import *
from .MultiProcessing import *
from .utils import *
from .ParameterSearch import *

# List of packages to include when doing a "*" import. 
__all__ = ["Model2HDM", "ModelCalculators", "utils", "MultiProcessing", "ParameterSearch"]

# Add all the methods from the packages to the __all__ list
from .Model2HDM import __all__ as methods_all
__all__.extend(methods_all)

from .ModelCalculators import __all__ as methods_all
__all__.extend(methods_all)

from .MultiProcessing import __all__ as methods_all
__all__.extend(methods_all)

from .utils import __all__ as methods_all
__all__.extend(methods_all)

from .ParameterSearch import __all__ as methods_all
__all__.extend(methods_all)



"""__all__ = ["Model2HDM", "methods_Model2HDM", 
           "methods_math", "methods_data", "methods_general",
           "ParameterSearch", "constraints_ParameterSearch",
           "class_MultiProcessing", "methods_MultiProcessing"]"""

# Print the packages that are imported
"""print("Using default packages, including:")
for name in __all__:
    if name != __all__[-1]:
        print(name, end=", " )
    else:
        print(name)"""