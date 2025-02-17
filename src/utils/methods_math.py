# Mathematical functions

# Imports
import sympy as sp  
import numpy as np  
from IPython.display import display, Math

from sympy.physics.quantum.dagger import Dagger   

#################### Math functions ####################

def dagger(M):
    return sp.conjugate(sp.transpose(M))

def hessian(potential, variables):
    """
    Calculates the Hessian matrix of a given potential with respect to a list of variables.

    Args:
    potential: The sympy expression representing the potential.
    variables: A list of sympy symbols representing the variables.

    Returns:
    A sympy Matrix object representing the Hessian matrix.
    """
    n = len(variables)
    hessian = np.zeros((n, n), dtype=object)
    for i in range(n):
        for j in range(n):
            hessian[i, j] = sp.diff(potential, variables[i], variables[j])
    return sp.Matrix(hessian)

#################### Methods ####################

def simplify_each_term(expr):
    if isinstance(expr, sp.Add): 
        return sp.Add(*[sp.simplify(term) for term in expr.args])
    return sp.simplify(expr)


def compute_unique_permutations(N, r):
    from itertools import product, combinations_with_replacement
    result = []
    # Generate unique combinations (with replacement)
    NN = [n for n in range(0,N)]
    NNr = [NN for n in range(r)]
    unique_combinations = list(combinations_with_replacement(NN,r))
    all_combinations = list(product(*NNr))

    unique_combinations_permuations = [0 for n in range(len(unique_combinations))]
    for i, ucomb in enumerate(unique_combinations):
        for comb in all_combinations:
            if set(comb) == set(ucomb):
                unique_combinations_permuations[i] += 1
    
    return unique_combinations, unique_combinations_permuations

def diagonalize_numerical_matrix(matrix, sorting=False):
    """
    Diagonalizes a matrix using SymPy.

    Args:
        matrix: A SymPy Matrix.

    Returns:
        A tuple containing the diagonal matrix and the transformation matrix.
    """

    #use_numpy = False
    #use_sympy = True
    threshold = 1e-10
    
    # check if sympy matrix and convert to numpy
    if isinstance(matrix, sp.Matrix):
       #check if matrix is numerical
        if all(entry.is_number for entry in matrix):
            matrix = np.array(matrix).astype(np.float64)
        else:
            raise ValueError("Matrix is not numerical.")
    elif isinstance(matrix, list):
        pass
    
    
    if not isinstance(matrix, np.ndarray):
        raise ValueError("Matrix is not a valid type or could not be converted to a numpy array.")
    
    block_diagonal = False    
    # check is matrix is 4x4 block diagonal
    if np.allclose(matrix[0:4, 4:8], 0, atol=threshold):
        block_diagonal = True   
        m_top = matrix[0:4, 0:4]
        m_bottom = matrix[4:8, 4:8]
        # Compute eigenvalues and eigenvectors
        eigenvalues_top, R_top = np.linalg.eigh(m_top) 
        eigenvalues_bottom, R_bottom = np.linalg.eigh(m_bottom)
        eigenvalues = np.concatenate((eigenvalues_top, eigenvalues_bottom))
        R = np.block([[R_top, np.zeros((4,4))], [np.zeros((4,4)), R_bottom]])
    
    else:
        # Compute eigenvalues and eigenvectors
        eigenvalues, P = np.linalg.eigh(matrix)
        R = P
    
    R_inv = np.linalg.inv(R)

    # Create the diagonal matrix
    D = np.diag(eigenvalues)

    # Verify: M should be equal to R^(-1) * D * R

    D_reconstructed = (R_inv @ matrix @ R)
    D_reconstructed = np.where(np.abs(D_reconstructed) < threshold, 0, D_reconstructed)
        
    #print("D_reconstructed:")
    #display(D_reconstructed)
        
    D = D_reconstructed 
    
    eigenvectors = R
    
    if sorting: # Add eigenvecotor sorting later!
        if block_diagonal:
            indicies_top = np.argsort(eigenvalues[0:4])
            indicies_bottom = np.argsort(eigenvalues[4:8])
            eigenvalues = np.array([eigenvalues_top[i] for i in indicies_top] + [eigenvalues_bottom[i] for i in indicies_bottom])
        else:
            indicies = np.argsort(eigenvalues)
            eigenvalues = np.array([eigenvalues[i] for i in indicies])

    return eigenvalues, eigenvectors, R


def get_independent_equations_indices(equations, symbols, include_constant=False):

    independent_indices = []
    dependent_indices = []
    independent_matrix = None  # This will hold the matrix (as a Sympy Matrix) of accepted rows

    for idx, eq in enumerate(equations):
        # Convert the equation to standard form: expr == 0 by subtracting rhs from lhs.
        expr = sp.expand(eq.lhs - eq.rhs)
        
        # Build the row vector using the coefficients for each symbol.
        row_coeffs = [sp.simplify(expr.coeff(s)) for s in symbols]
        
        if include_constant:
            # The constant term is what remains when all symbols are set to 0.
            constant = sp.simplify(expr.subs({s: 0 for s in symbols}))
            row_coeffs.append(constant)
        
        # Convert the coefficient list to a 1xN Sympy Matrix.
        row_matrix = sp.Matrix([row_coeffs])
        
        if independent_matrix is None:
            # If no rows have been accepted yet, accept this one by default.
            independent_matrix = row_matrix
            independent_indices.append(idx)
        else:
            # Stack the new row with the previously accepted rows.
            candidate_matrix = sp.Matrix.vstack(independent_matrix, row_matrix)
            # If the rank increases, then the new row is linearly independent.
            if candidate_matrix.rank() > independent_matrix.rank():
                independent_matrix = candidate_matrix
                independent_indices.append(idx)
    
    return independent_indices


