
from .ParameterSearch import ParamSearch
from .parameter_constraints import *

__all__ = ["ParamSearch",
            "parameter_constraints"
           ]

import pkgutil
import inspect
import sys

# Get the current package name
package_name = __name__

# Automatically import all submodules
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, package_name + "."):
    __import__(module_name)

# Automatically add all functions and classes from submodules to the package namespace
#__all__ = []  # Collects everything for import *

for module_name in list(sys.modules):
    if module_name.startswith(package_name):  # Only scan submodules of Model2HDM
        module = sys.modules[module_name]

        for name, obj in inspect.getmembers(module):
            if inspect.isfunction(obj) or inspect.isclass(obj):  # Only include functions/classes
                globals()[name] = obj  # Inject into the current namespace
                __all__.append(name)  # Add to __all__
