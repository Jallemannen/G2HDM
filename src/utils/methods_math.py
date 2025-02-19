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

# May remove later?
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

# Solve a linear system of equations
def linear_solve(eq_list, var_list, include_trivial_solutions=False):
    """
    Solves a list of linear equations.
    """
    try:
        sol = sp.linsolve(eq_list, var_list)
    except Exception as e:
        print("Error raised when solving the linear equations")
        print(e)
        return None
    if len(sol) == 0:
        return None
    
    sol_eqs = [sp.Eq(var, val) for var, val in zip(var_list, sol.args[0])]
    
    if not include_trivial_solutions:
        sol_eqs = [eq for eq in sol_eqs if eq != True]
    else:
        sol_eqs = [sp.Eq(var, val, evaluate=False) for var, val in zip(var_list, sol.args[0])]
    
    return sol_eqs


# ADD: Include all masses! (eigenvalues)
def diagonalize_numerical_matrix(matrix, sorting=False):
    """
    Diagonalizes a numerical matrix using numpy. 
    Assumes symmetric matrix.

    Args:
        matrix: A SymPy Matrix.

    Returns:
        A tuple containing the diagonal matrix and the transformation matrix.
    """
    
    threshold = 1e-10

    # If matrix is a sympy Matrix, check that all entries are numbers, then convert.
    if isinstance(matrix, sp.Matrix):
        if all(entry.is_number for entry in matrix):
            # Convert using list-of-lists to get proper numerical (float) values.
            matrix = np.array(matrix.tolist(), dtype=np.float64)
        else:
            raise ValueError("Matrix is not numerical.")
    elif isinstance(matrix, list):
        matrix = np.array(matrix, dtype=np.float64)
    elif not isinstance(matrix, np.ndarray):
        raise ValueError("Matrix is not a valid type or could not be converted to a numpy array.")

    # Ensure matrix is square.
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")

    N = matrix.shape[0]
    block_diagonal = False

    # Check for block-diagonality only if N is even.
    if N % 2 == 0:
        n = N // 2
        # Check that both off-diagonal blocks are near zero.
        if (np.allclose(matrix[0:n, n:N], 0, atol=threshold) and 
            np.allclose(matrix[n:N, 0:n], 0, atol=threshold)):
            block_diagonal = True

    if block_diagonal:
        n = N // 2
        m_top = matrix[0:n, 0:n]
        m_bottom = matrix[n:N, n:N]

        eigenvalues_top, R_top = np.linalg.eigh(m_top)
        eigenvalues_bottom, R_bottom = np.linalg.eigh(m_bottom)

        # Optionally sort each block (np.linalg.eigh already returns sorted eigenvalues;
        # but if you need additional control, you can sort here)
        if sorting:
            top_indices = np.argsort(eigenvalues_top)
            bottom_indices = np.argsort(eigenvalues_bottom)
            eigenvalues_top = eigenvalues_top[top_indices]
            eigenvalues_bottom = eigenvalues_bottom[bottom_indices]
            R_top = R_top[:, top_indices]
            R_bottom = R_bottom[:, bottom_indices]

        eigenvalues = np.concatenate((eigenvalues_top, eigenvalues_bottom))
        # Build block-diagonal eigenvector matrix.
        R = np.block([[R_top, np.zeros((n, n))],
                      [np.zeros((n, n)), R_bottom]])
    else:
        # For a general symmetric matrix, np.linalg.eigh returns sorted eigenvalues.
        eigenvalues, R = np.linalg.eigh(matrix)

    # Since R is orthonormal, we can verify the diagonalization via:    D_reconstructed = R.T @ matrix @ R
    """D_reconstructed = R.T @ matrix @ R
    D_reconstructed[np.abs(D_reconstructed) < threshold] = 0
    D = D_reconstructed"""

    eigenvectors = R

    return eigenvalues, eigenvectors, R
    
    
    
    
    
    
    
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
    
    
    # check is matrix is 4x4 block diagonal
    N = int(matrix.shape[0])
    n = int(N/2)
    block_diagonal = False
    if np.allclose(matrix[0:n, n:N], 0, atol=threshold):
        block_diagonal = True
        m_top = matrix[0:n, 0:n]
        m_bottom = matrix[n:N, n:N]
        # Compute eigenvalues and eigenvectors
        eigenvalues_top, R_top = np.linalg.eigh(m_top) 
        eigenvalues_bottom, R_bottom = np.linalg.eigh(m_bottom)
        eigenvalues = np.concatenate((eigenvalues_top, eigenvalues_bottom))
        R = np.block([[R_top, np.zeros((n,n))], [np.zeros((n,n)), R_bottom]])
    
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
            indicies_top = np.argsort(eigenvalues[0:n])
            indicies_bottom = np.argsort(eigenvalues[n:N])
            eigenvalues = np.array([eigenvalues_top[i] for i in indicies_top] + [eigenvalues_bottom[i] for i in indicies_bottom])
        else:
            indicies = np.argsort(eigenvalues)
            eigenvalues = np.array([eigenvalues[i] for i in indicies])

    return eigenvalues, eigenvectors, R"""


def sort_eigenvalues(evals_list, evecs_list=None):
    
    from scipy.optimize import linear_sum_assignment
    num_steps = len(evals_list)
    
    # Case 1: No eigenvector data provided. Sort each step independently.
    if evecs_list is None:
        sorted_evals = np.array([np.sort(evals) for evals in evals_list])
        return sorted_evals

    # Case 2: Use eigenvectors to match branches across parameter steps.
    sorted_evals = []
    sorted_evecs = []
    
    # For the first parameter value, use the given order as baseline.
    sorted_evals.append(evals_list[0])
    sorted_evecs.append(evecs_list[0])
    prev_evecs = evecs_list[0]  # expected shape: (n, n) with eigenvectors as columns

    for i in range(1, num_steps):
        current_evals = evals_list[i]
        current_evecs = evecs_list[i]
        
        # Compute the cost matrix based on the absolute overlap between eigenvectors.
        # A higher overlap means a lower cost (hence the minus sign).
        cost = -np.abs(prev_evecs.conj().T @ current_evecs)
        
        # Solve the assignment problem to match eigenvector branches.
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Reorder the current eigenvalues and eigenvectors to match the previous order.
        current_evals_sorted = current_evals[col_ind]
        current_evecs_sorted = current_evecs[:, col_ind]
        
        sorted_evals.append(current_evals_sorted)
        sorted_evecs.append(current_evecs_sorted)
        
        # Update previous eigenvectors for the next iteration.
        prev_evecs = current_evecs_sorted

    sorted_evals = np.array(sorted_evals)
    
    return sorted_evals, sorted_evecs
    
    
    


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


