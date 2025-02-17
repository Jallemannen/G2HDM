
import sympy as sp
from IPython.display import display 

#################### General Methods ####################

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




