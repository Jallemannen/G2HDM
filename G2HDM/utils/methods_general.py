import sys
import time
import sympy as sp
from IPython.display import display, clear_output, Math
import shutil

#################### Printing Methods ####################

def is_jupyter():
    """Detect if running inside Jupyter Notebook."""
    try:
        return 'zmqshell' in str(sys.modules['IPython'].get_ipython())
    except:
        return False

def latex_to_sympy(latex_expr):
    """Convert LaTeX math string to a SymPy expression."""
    try:
        return sp.sympify(latex_expr, evaluate=False)  # Prevents unwanted simplifications
    except Exception:
        return latex_expr  # If conversion fails, return the raw string

def smart_print(*args, show=True):
    """Smart print function with optional display control."""
    if not show:
        return  # If show=False, skip printing

    for arg in args:
        if isinstance(arg, Math):
            # Extract LaTeX string from Math object
            latex_expr = arg._repr_latex_()[2:-2]  # Remove LaTeX delimiters
            
            if is_jupyter():
                display(arg)  # Jupyter: Use Math rendering
            else:
                sympy_expr = latex_to_sympy(latex_expr)  # Convert LaTeX to SymPy
                print("Math Expression (Console):")
                sp.pprint(sympy_expr, use_unicode=True)  # Console: Pretty print with Unicode
        else:
            if is_jupyter():
                display(arg)  # Jupyter: Use display()
            else:
                print(arg)  # Console: Use standard print

def refresh_text(*args):
    """
    Refresh multiple lines of text **in-place** without clearing other output.
    
    - In Jupyter: Uses IPython widgets to update displayed text.
    - In Console: Moves the cursor up and overwrites previous lines.
    
    Each argument is printed on its own line.
    """
    if is_jupyter():
        # In Jupyter, use clear_output but retain previous output
        clear_output(wait=True)
        for line in args:
            print(line)
    else:
        # Get terminal width to pad lines correctly
        columns = shutil.get_terminal_size((80, 20)).columns
        
        # Move cursor up for each previous line, then overwrite
        for i in range(len(args)):
            sys.stdout.write("\033[F")  # Move cursor up one line

        for line in args:
            sys.stdout.write("\r" + line.ljust(columns) + "\n")  # Overwrite the line

        sys.stdout.flush()

def sec_to_hms(secs:float)->str:
    h = int(secs // 3600)
    m = int((secs % 3600) // 60)
    s = int(secs % 60)
    return f"{h}h:{m:02d}m:{s:02d}s"

#################### General Methods ####################

def bytes_suffix(bytes:int)->str:
    """Converts bytes to a string with an appropriate size suffix."""
    if bytes < 1024:
        return f"{bytes} B"
    elif bytes < 1024**2:
        return f"{bytes/1024:.2f} KB"
    elif bytes < 1024**3:
        return f"{bytes/1024**2:.2f} MB"
    elif bytes < 1024**4:
        return f"{bytes/1024**3:.2f} GB"
    else:
        return f"{bytes/1024**4:.2f} TB"

def cprint(*args, **kwargs):
    show = kwargs.pop("show", True)
    if show:
        print(*args, **kwargs)
    
def cdisplay(*args, **kwargs):
    show = kwargs.pop("show", True)
    if show:
        display(*args, **kwargs)
    else:
        pass
    
def present(*args, func=display, **kwargs):
    show = kwargs.pop("show", True)
    if show:
        func(*args, **kwargs)
    else:
        pass
    


# Subs complex variables to re and im parts in a list of variables
def subs_complex_params(params:list, include_zeros:bool=True):
    """Input a list of sympy variables and returns a list of variables with real and imaginary parts."""
    variables = []
    for symb in params:
        if symb != 0:
            if symb.is_real:
                variables.append(symb)
            
            # add a way to give the same ans when inputs are in re and Im form. 
                """elif symb == sp.im(symb):
                    print("hit")
                    variables.append(symb)"""
            else:
                variables.append(sp.re(symb))
                variables.append(sp.im(symb))
        elif include_zeros == True:
            variables.append(symb)
    return variables

# Subs complex variables to re and im parts in a list of expressions
def subs_complex_expressions(expr_list:list):
    """Substitutes complex variables to real and imaginary parts."""
    new_expr_list = []
    for expr in expr_list:
        for symb in expr.free_symbols:
            if symb != 0:
                if symb.is_real != True:
                    expr = expr.subs({symb.conjugate(): sp.re(symb) - sp.I*sp.im(symb)})
                    expr = expr.subs({symb: sp.re(symb) + sp.I*sp.im(symb)})
        new_expr_list.append(expr)
    return new_expr_list

# convert list of eqs to dict
def eqs_to_dict(eqs:list):
    """Converts a list of sympy equations to a dictionary."""
    eq_dict = {}
    for eq in eqs:
        eq_dict[eq.lhs] = eq.rhs
    return eq_dict

# Generate matrix
def generate_matrix(n, symb, symmetric=False):
    """
    Generates an nxn matrix with symbolic variables for each element.

    Args:
        n: The size of the matrix.

    Returns:
        A sympy Matrix object with symbolic variables.
    """
    symbols = []
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            symbols.append(sp.Symbol(f"{symb}_{{{i}{j}}}", real="true"))

    matrix_elements = [symbols[i:i+n] for i in range(0, len(symbols), n)]
    matrix = sp.Matrix(matrix_elements)
  
    if symmetric:
        for i in range(n):
            for j in range(i+1, n):
                matrix[j, i] = matrix[i, j]
        
    return matrix

# Generate vector
def generate_vector(n, symb):
  """
  Generates an n-length vector with symbolic variables.

  Args:
    n: The length of the vector.

  Returns:
    A sympy Matrix object representing the vector.
  """
  symbols = []
  for i in range(1, n + 1):
    symbols.append(sp.Symbol(f"{symb}_{{{i}}}", real="true"))

  vector = sp.Matrix(symbols)
  return vector




