# Methods for the model class

# Calculate = Numerical solution
# Generate / Solve = Analytical solution

# Imports
import os
import sympy as sp  
import numpy as np  
import matplotlib.pyplot as plt  
import time as time
from IPython.display import display, Math  

# Custom imports
#from .class_Model2HDM import Model2HDM
from .potentials import *
from ..utils.methods_math import *
from ..utils.methods_data import *
from ..utils.methods_general import *

# constants
from src.utils import constants as const

#################### General Methods ####################

# Load model
def load_model(name:str, path:str=None, project_root = None):
    """Need to be in Saved folder. Returns the loaded model, given its folder, if it exists, else it returns None"""
    if project_root is None:
        project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
    if path == None:
        path = os.path.join(project_root, "Models", "Saved models", f"{name}", "model_data", "model.pkl")

    return load_pkl(path)    

# Find parameter relations between two models (WIP)
def generate_parameter_relations(model1, model2, R,
                                           display_solution=True, extra_subs={}):
    
    if display_solution:
            print("========== Parameter Relations ==========")

    def find_parameter_eqs(model1, model2, R, extra_subs={}):
        x1 = model1.field1 
        x2 = model1.field2
        params = model1.params("V0", complex=False)
        params_op = model2.params("V0", complex=False)
        potential = model1.potential("V0_display")
        
        # define subs
        #subs1 = {dagger(x1_symb)*x1_symb:1, dagger(x2_symb)*x2_symb:1, dagger(x1_symb)*x2_symb:1, dagger(x2_symb)*x1_symb:1}
        subs1 = {dagger(x1)*x1:1, dagger(x2)*x2:1, dagger(x1)*x2:1, dagger(x2)*x1:1}
        #subs1 = {dagger(y1)*y1:1, dagger(y2)*y2:1, dagger(y1)*y2:1, dagger(y2)*y1:1}
        
        # Define function to separate the terms and parameters from the potential
        def separate_potential_terms_and_params(pot, params):
            #subs1 = {dagger(x1)*x1:1, dagger(x2)*x2:1, dagger(x1)*x2:1, dagger(x2)*x1:1} #x1_symb
            # Create list of terms we want to factor out
            terms = pot.eval(x1, x2).as_ordered_terms()
            new_terms = []
            for term in terms:
                symbols = term.free_symbols
                for symb in symbols:
                    if symb in params:
                        new_terms.append(term.subs(subs1)) #, sp.Rational(1,2):1})) #coeff*
            #print("new_terms")
            #display(*new_terms)
        
            # Collect all the parameters corresponing the the factored terms (ordered, and conjugated)
            params_symbols = []
            for i, term in enumerate(terms):
                param = term.subs({new_terms[i]:1})
                params_symbols.append(param.expand())
                
                
            ### sort out prefactors (1/2 and -1)
            terms_prefactors = []
            terms_sorted = []
            for term in new_terms:
                #subs1 = {dagger(x1)*x1:1, dagger(x2)*x2:1, dagger(x1)*x2:1, dagger(x2)*x1:1} #x1_symb
                coeff = term.subs(subs1).as_coeff_Mul()[0]
                terms_prefactors.append(coeff)
                terms_sorted.append(term/coeff)
                
            #print("params_symbols")
            #display(*params_symbols)
            #display(*terms_sorted, *params_symbols, *terms_prefactors) 
            return terms_sorted, params_symbols, terms_prefactors
        
        ### Separeate the terms and parameters from the potential
        params_symbols, field_terms, terms_prefactors = separate_potential_terms_and_params(potential, params)

        for i,field_term in enumerate(field_terms):
            field_terms[i] = field_term.expand()
            
        #print("new_terms", using_higgs_basis)
        #display(*field_terms)
        #display(*params_symbols)
        
        if len(field_terms) != len(params_symbols):
            print("field_terms")
            display(*field_terms)
            print("params_symbols")
            display(*params_symbols)
            raise Exception("The number of terms and parameters are not equal.")
        
        ### Insert the new basis into the potential
        a1, a2 = sp.symbols("a1 a2")
        x = R * sp.Matrix([a1,a2])
        x = x.subs({a1: x1, a2: x2})
        V0 = potential.eval(x[0], x[1], params=params_op).expand() #expand_power_exp=True, expand_mul=True
        #V0_sorted = sp.collect(V0.expand()/2, new_terms)*2
        #V0_sorted = custom_collect(V0.expand()/2, new_terms)*2
        V0_sorted = V0 # Make sure the potential is sorted out correctly
        
        ## Use this to debug (checks if the transform is correct)
        #display(V0_sorted.subs({model.omega2:0, model.omega_CP:0, model.omega:model.omega1}).expand())
        #display(V0_sorted.subs({model.omega1:0, model.omega_CP:0, model.omega:model.omega2}).expand())
        
        ## Debug
        #display(x[0],x[1])
        #display(V0_sorted)
        #print("field_terms")
        #display(*field_terms)
        
        ### sort out the new params
        new_params = []
        V0_rest = V0_sorted
        for field_term, coeff in zip(field_terms, terms_prefactors): # list of the field terms in the potential
            new_param_term = 0
            for term in V0_sorted.as_ordered_terms(): # list of the terms in the potential
                
                
                factors = field_term.as_ordered_factors()
                factors_terms = term.as_ordered_factors()
                if len(factors) == 4:
                    param_term1 = term.subs({field_term:1})
                    # check revese of (xixj)*(xixj) 
                    field_term_reverse = factors[2]*factors[3] * factors[0]*factors[1]
                    param_term2 = term.subs({field_term_reverse:1})
                elif len(factors) == 2 and len(factors_terms) == 3:
                    param_term1 = term.subs({field_term:1})
                    param_term2 = 0
                else:
                    param_term1 = 0
                    param_term2 = 0
                    
                if param_term1 != 0 and param_term1 != term: 
                    ## Debug sorting
                    #print("debug")
                    #display(field_term) #, dagger(field_term).doit())
                    #display(coeff)
                    #display(term)
                    
                    new_param_term += param_term1
                    V0_rest = V0_rest - term
        
                elif param_term2 != 0 and param_term2 != term: #and param_term2 == param_term2.subs(subs1) and param_term2 != term.subs({sp.Mul(dagger(field_term).doit(),dagger(field_term).doit(),evaluate=True):1}):
                    ## Debug sorting
                    #print("reversal")
                    #display(field_term, field_term_reverse)
                    #display(term)
                    
                    new_param_term += param_term2
                    V0_rest = V0_rest - term
         
            if new_param_term != 0:
                new_params.append(new_param_term/coeff )#/coeff #.subs({x1_symb:0, x2_symb:0})
            else:
                raise Exception("No parameter found for term: {}".format(field_term))  
        
        # Check so that all terms has been sorted out
        try: 
            V0_rest[0].evalf() == 0
        except:
            print("WARNING: Not all terms were sorted out.")
            display(Math("V_{rest} = " + sp.latex(V0_rest))) 
            #raise Exception("Not all terms were sorted out.")
        
        ### Create eqs and Sort by re and im parts
        equations = [] 
        for symb, param in zip(params_symbols, new_params):
            param_exp = subs_complex_expressions([param])[0].expand(complex=True)
            if symb.is_real:
                #equations.append(sp.Eq(symb, param))
                #equations.append(sp.Eq(sp.re(symb), sp.re(param) )) #.expand(complex=True)
                equations.append(sp.Eq(symb, param_exp)) # subs_complex_expressions([param])[0])) #.expand(complex=True)])[0]))#.expand(complex=True))) #(sp.re(param) + sp.I*sp.im(param)) )) # sp.re(param) + sp.im(param)
            else:
                if list(symb.free_symbols)[0].conjugate() != symb:
                    equations.append(sp.Eq(sp.re(symb), sp.re(param_exp))) #.expand(complex=True))))#)) #.expand(complex=True)))
                    equations.append(sp.Eq(sp.im(symb), sp.im(param_exp))) #.expand(complex=True))))#)) #.expand(complex=True)))

        return equations

    # Run the function
    eqs_lambda_to_Z = find_parameter_eqs(False, extra_subs)
    if display_solution:
        print("Rotation matrix:")
        display(R)
        print("lambda in terms of Z:")
        display(*eqs_lambda_to_Z)
     
    eqs_Z_to_lambda = find_parameter_eqs(True, extra_subs)
    if display_solution:
        print("--------------------------------------------")
        print("Inverse rotation matrix:")
        display(R.inv())
        print("Z in terms of lambda:")
        display(*eqs_Z_to_lambda)

    return eqs_lambda_to_Z, eqs_Z_to_lambda

#################### Solvers ####################
# show = "solution", "procedure", "none"

def _generate_tadpole_equations(potential, fields, subs_dict, remove_trivial_eqs=True)->list:
    """
    Generates a list of tadpole equations.
    """
    tadpole_eqs = []
    for field in fields:
        diff = potential.expand().diff(field)
        diff = diff.subs(subs_dict)
        tadpole_eqs.append(sp.Eq(0, diff, evaluate=False))
        
    if remove_trivial_eqs:
        tadpole_eqs = [eq for eq in tadpole_eqs if eq.rhs != 0]
    
    return tadpole_eqs

def _linear_solve(eq_list, var_list, include_trivial_solutions=False):
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

# Generate analytical solution of the tree-level masses
def generate_level0_masses(model:object, VEV:bool=False, apply_tadpole:bool=True, solve_eigenvalues:bool=True, 
                           show_procedure:bool=True, show_solution:bool=True):
    
    #V0, params, fields, bgfields, VEVfields, 
    #VEV=False, apply_tadpole=True, solve_eigenvalues=True,
    #show_procedure=False, show_solution=False
    
    # Unpacking
    V0 = model.V0
    params = model.V0_params
    fields = model.fields
    bgfields = model.bgfields
    VEVs = model.VEVs
    
    if show_procedure:
        show_solution = True
    elif not show_solution:
        show_procedure = False
    
    # Asserions
    assert V0 is not None, "The potential has not been assigned"
    
    # Define subsitutions
    fields_to_zero_dict = {field:0 for field in fields}
    bgfields_to_VEV_dict = {bg:VEV for bg, VEV in zip(bgfields, VEVs)}
    
    # Potential at VEV
    V0_VEV = V0.subs(bgfields_to_VEV_dict)
    
    if VEV:
        V0 = V0_VEV
    
    # Generate tadpole eqs
    tadpole_eqs = _generate_tadpole_equations(V0_VEV, fields, fields_to_zero_dict, False)
    
    # Solving tadpole eqs
    sol_tadpole_eqs = _linear_solve(tadpole_eqs, params)
    if apply_tadpole:
        sol_tadpole_dict = eqs_to_dict(sol_tadpole_eqs)
    else:
        sol_tadpole_dict = {}
    
    cprint("Tadpole equations:", show=show_procedure)
    for eq, field in zip(tadpole_eqs, fields):
        latex_string = r"\frac{\partial V_0}{\partial " + sp.latex(field) + "} = " + sp.latex(eq)
        cdisplay(Math(latex_string), show=show_procedure)
    
    cprint("Solutions to the tadpole equations:", show=show_procedure)
    cdisplay(*sol_tadpole_eqs, show=show_procedure)
    
    # Generate hessian
    M0 = sp.hessian(V0, fields).subs(fields_to_zero_dict)
    M0 = M0.subs(sol_tadpole_dict)
    
    cprint("Mass matrix:", show=show_procedure)
    cdisplay(M0, show=show_procedure)
    
    # Solve for the eigenvalues
    M_neutral = None
    M_charged = None
    M0_charged_eigenvalues = None
    M0_neutral_eigenvalues = None
    M0_eigenvalues = None
    
    if solve_eigenvalues:
        
        if M0[:4, 4:].is_zero_matrix:
    
            M_neutral = M0[:4, :4] 
            M_charged = M0[4:, 4:] 
            # Extract eigenvalues
            cprint("Charged mass eigenvalues:", show=show_solution)
            M0_charged_eigenvalues = [val for val in list(M_charged.eigenvals())]
            
            for i, val in enumerate(M0_charged_eigenvalues):
                cdisplay(Math(f"m_{i+1} = " + sp.latex(val)), show=show_solution)
            
            cprint("Neutral mass eigenvalues:", show=show_solution)
            M0_neutral_eigenvalues = [val for val in list(M_neutral.eigenvals())]  
            
            for i, val in enumerate(M0_neutral_eigenvalues):
                cdisplay(Math(f"m_{i+1} = " + sp.latex(val)), show=show_solution)
            
            # change order, neutral first.
            M0_eigenvalues = M0_charged_eigenvalues + M0_neutral_eigenvalues
            
        else:
        
            cprint("Mass eigenvalues:", show=show_solution)
            M0_eigenvalues = [val for val in list(M0.eigenvals())]
            
            for i, val in enumerate(M0_eigenvalues):
                cdisplay(Math(f"m_{i+1} = " + sp.latex(val)), show=show_solution)
            
        
    # Save the results
    if VEV:
        model.DATA["V0_tadpole_eqs"] = tadpole_eqs
        model.DATA["V0_tadpole_eqs_solutions"] = sol_tadpole_eqs
        model.DATA["M0_VEV"] = M0
        model.DATA["M_charged_VEV"] = M_charged
        model.DATA["M_neutral_VEV"] = M_neutral
        model.DATA["M0_charged_eigenvalues_VEV"] = M0_charged_eigenvalues
        model.DATA["M0_neutral_eigenvalues_VEV"] = M0_neutral_eigenvalues
        model.DATA["M0_eigenvalues_VEV"] = M0_eigenvalues
    
    else:
        model.DATA["V0_tadpole_eqs"] = tadpole_eqs
        model.DATA["V0_tadpole_eqs_solutions"] = sol_tadpole_eqs
        model.DATA["M0"] = M0
        model.DATA["M_charged"] = M_charged
        model.DATA["M_neutral"] = M_neutral
        model.DATA["M0_charged_eigenvalues"] = M0_charged_eigenvalues
        model.DATA["M0_neutral_eigenvalues"] = M0_neutral_eigenvalues
        model.DATA["M0_eigenvalues"] = M0_eigenvalues
    
    # Return the results
    #return tadpole_eqs, sol_tadpole_eqs, M0, M_charged, M_neutral, M0_charged_eigenvalues, M0_neutral_eigenvalues

# Solve tree-level masses
def solve_counterterms(model, extra_eqs=None, show_procedure=True, show_solution=True):
    
    # Unpacking
    VCT_params = model.VCT_params
    fields = model.fields
    bgfields = model.bgfields
    VEVs = model.VEVs
    VCT = model.VCT
    
    if show_procedure:
        show_solution = True
    elif not show_solution:
        show_procedure = False
    
    # Asserions
    assert VCT is not None, "The potential has not been assigned"

    # Define subsitutions
    subs_fields_to_zero = {field:0 for field in fields}
    subs_bgfields_to_VEV = {bg:VEV for bg, VEV in zip(bgfields, VEVs) if bg != 0}

    # Generate mass matrix and tadpole eqs
    Ni = generate_vector(8, "N")
    Hij = generate_matrix(8, "H", symmetric=True)
    
    MCT = sp.hessian(VCT, fields).expand().subs(subs_fields_to_zero)
    MCT_VEV = MCT.subs(subs_bgfields_to_VEV)
    tadpole_eqs = _generate_tadpole_equations(VCT.subs(subs_bgfields_to_VEV), fields, subs_fields_to_zero, False)
    
    # Construct matrix equation
    M_lhs = -sp.diag(sp.diag(*Ni), Hij)
    M_rhs = sp.diag(*[eq.rhs for eq in tadpole_eqs], MCT_VEV)
    equations = [sp.Eq(M_lhs[i, j], M_rhs[i, j]) for i in range(M_lhs.rows) for j in range(M_lhs.cols) if i <= j]
    non_trivial_eqs = [eq for eq in equations if isinstance(eq, sp.Equality) and not (eq.rhs == 0 or eq.lhs == 0)]
    equations = non_trivial_eqs
    
    indicies = get_independent_equations_indices(equations, VCT_params)
    equations_indep = [equations[i] for i in indicies]
    
    n_params = len(VCT_params)
    textwidth = 50

    if show_procedure:
        print(" Initial conditions ".center(textwidth, "="))
        print("Number of variables:", n_params)
        print("Number of total eqs:", len(non_trivial_eqs))
        print("Number of linearly independent equations:", len(equations_indep))
        if extra_eqs is not None:
            print("Number of extra constraints:", len(extra_eqs))
    
        print(" Equations ".center(textwidth, "="))
        display(*equations)
        print(" Linearly independent equations ".center(textwidth, "="))
        display(*equations_indep)
        if extra_eqs is not None:
            print("\n", " Extra constraints ".center(textwidth, "="))
            display(*extra_eqs)
        

    # Solve the equations
    if extra_eqs is not None:
        equations = equations + extra_eqs

    NH_variables = [list(eq.lhs.free_symbols)[0] for eq in equations]
    solution = _linear_solve(equations, VCT_params + NH_variables, include_trivial_solutions=True)

    if solution is None:
        raise Exception("No solution found")
    
    # Sort the solution
    solution_eqs = [eq for eq in solution[0:n_params] if eq.evalf() != True]
    undetermined_eqs = [eq for eq in solution[0:n_params] if eq.evalf() == True]
    consistency_eqs = [eq for eq in solution[n_params:] if eq.evalf() != True]
    
    cprint(" Solution ".center(textwidth, "="), show=show_solution)
    cdisplay(*solution_eqs, show=show_solution)
    cprint(" Undetermined equations ".center(textwidth, "="), show=show_solution and len(undetermined_eqs) > 0)
    cprint("Undetermined parameters will be set to zero", show=show_solution and len(undetermined_eqs) > 0)
    cdisplay(*undetermined_eqs, show=show_solution and len(undetermined_eqs) > 0)
    cprint(" Consistency equations ".center(textwidth, "="), show=show_solution)
    consistency_eqs = [eq for eq in solution[n_params:] if eq.evalf() != True]
    
    cdisplay(*consistency_eqs, show=show_solution)
    
    # Check if the solution works
    cprint("========== Counterterm matrix ==========".center(textwidth, "="), show=show_procedure)
    cprint("Please check if the solution is correct", show=show_procedure)
    subs_solution = {eq.lhs: eq.rhs for eq in solution_eqs} | {eq.lhs:0 for eq in undetermined_eqs}
    MCT_sol = sp.simplify(MCT_VEV.subs(subs_solution).expand())
    cdisplay(MCT_VEV, MCT_sol, show=show_procedure)
    #check against MCW at vev
    
    # Return the solution
    equations = solution[0:n_params]
    result_eqs = [sp.Eq(eq.lhs,eq.rhs.subs({eq.lhs:0 for eq in undetermined_eqs})) for eq in equations]
    
    model.DATA["VCT_params_solution"] = result_eqs
    model.DATA["MCT"] = MCT
    
    #return result_eqs, MCT
    # Assign the VCT parameter values

#################### Numerical calculations ####################

# Calculate tree-level masses
def _calculate_level0_masses(M0_numerical, sorting=True):
    masses, field_states, R = diagonalize_numerical_matrix(M0_numerical, sorting=sorting)
    return masses

# Calc N_i and Hij from roman 2016 paper
def calculate_CW_potential_derivatives_numerical(M0_numerical, V0_fdep, fields, fields_mass,
                                                 show_procedure=False):
    """_summary_

    Args:
        M0_numerical: _description_
        V0: Should be evaluated for all values except for the fields
        fields: _description_
        fields_mass: _description_
        show_procedure: _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    v = const.v
    mu = v
    
    #M0_neutal = M0[:4, :4]
    #M0_charged = M0[4:, 4:]
    #R_n = compute_diagonalization_matrix(M0_neutal)
    #R_c = compute_diagonalization_matrix(M0_charged)
    
    #, params_values_subs, bgfields_values_subs, VEV_values_subs
    #M0 = M0.subs(params_values_subs)
    #M0 = M0.subs(VEV_values_subs)
    #M0 = M0.subs(bgfields_values_subs)
    
    masses, field_states, R = diagonalize_numerical_matrix(M0_numerical)
    
    # Check for imaginary masses
    for mass in masses:
        if isinstance(mass, complex):
            if mass.imag < mass.real*1e-5:
                mass = mass.real
            print("Imaginary mass found!:", mass)
                

    ### Calculate mass eigenstates in terms of the fields
    fields_states_mass = sp.Matrix(R) * sp.Matrix(fields_mass)
    fields_states = sp.Matrix(R).inv() * sp.Matrix(fields)
    fields_states_mass_subs = {f:fm for f,fm in zip(fields, fields_states_mass)}
    fields_mass_to_zero = {f:0 for f in fields_mass}
    fields_to_zero = {f:0 for f in fields_states}
    
    # Potential in terms of mass eigenstates
    V0_mass = V0_fdep.subs(fields_states_mass_subs).expand()

    #for key, value in fields_states_mass_subs.items():
    #    display(Math(f"{key} = {value}"))
    
    #display(*fields_states_mass_subs.items())
    
    ### Calculate the trilinear and quartic couplings
    from itertools import permutations
    L3 = np.zeros((8,8,8))
    L4 = np.zeros((8,8,8,8))
    
    cprint("Calculating the trilinear couplings...", show = show_procedure)
    combinations3, S3 = compute_unique_permutations(8,3)
    
    n=0
    for i,j,k in combinations3:
        fieldterm = fields_mass[i]*fields_mass[j]*fields_mass[k]
        term = V0_mass.coeff(fieldterm).expand().subs(fields_mass_to_zero)
        for a,b,c in permutations([i,j,k]):
            L3[a,b,c] = sp.re(term)
        n+=1
        if n % 32 == 0:
            cprint(n, f"/ {len(combinations3)} ", end="\r", show = show_procedure)
    
    cprint("\nCalculating the quartic couplings...", show = show_procedure)
    combinations4, S4 = compute_unique_permutations(8,4)
    for i,j,k,l in combinations4:
        fieldterm = fields_mass[i]*fields_mass[j]*fields_mass[k]*fields_mass[l]
        term = V0_mass.coeff(fieldterm).expand().subs(fields_mass_to_zero)
        for a,b,c,d in permutations([i,j,k,l]):
            L4[a,b,c,d] = term
        n+=1
        if n % 32 == 0:
            cprint(n, f"/ {len(combinations4)} ", end="\r", show = show_procedure)
    
    ### Calculate the derivatives
    NCW = np.zeros(8)
    MCW = np.zeros((8,8))
    
    kT = 3/2
    sT = 0 # for all scalar particles
    coeff = (-1)**(2*sT)*(1+2*sT) / 2
    coeff = 1/2
    epsilon = 1/(4*np.pi)**2
    
    # log term 1, with regulator
    def g(m):
        is_negative = False
        sign = 1
        if m < 0:
            is_negative = True
            m = -m
            sign = -1
            
        if m==0: #.is_zero:
            result = 0
        else:
            result =  m * (sp.log(m/mu**2)-kT+1/2)
            
        if is_negative:
            return -result
        else:
            return result
    
    # log term 2, with regulator
    def f(m1, m2):
        m1_is_negative = False
        m2_is_negative = False
        sign1 = 1
        sign2 = 1
        if m1 < 0:
            m1_is_negative = True
            m1 = -m1
            sign1 = -1
        if m2 < 0:
            m2_is_negative = True
            m2 = -m2
            sign2 = -1
        
        # calc
        if m1 == 0 and m2 == 0:
            return 1
        if m1 != 0:
            log1 = sign1*sp.log(m1/mu**2)
            log2 = 0
            
        if sp.Abs(m1-m2)>1e-5: # add threshold
            if m2 != 0:
                log2 = sign2*sp.log(m2/mu**2)
            if m1 == 0:
                log1 = 0
            if m2 == 0:
                log2 = 0        
        else: 
            return 1 + sign1*sp.log(m1/mu**2)
            
        return (m1*log1 - m2*log2)/(m1-m2)
            
        """elif sp.Eq(m1,m2) or m2 == 0: #m2.is_zero: # IR divergency with regulator
            if m1_is_negative:
                return 1 - sp.log(m1/mu**2)
            else:
                return 1 + sp.log(m1/mu**2)
        elif m1 == 0:  # m1.is_zero: 
            if m2_is_negative:
                return 1 - sp.log(m2/mu**2)
            else:
                return 1 + sp.log(m2/mu**2)"""
        #return (m2*sp.log(m2/mu**2))/(-m2)
        """else:
            term1 = m1*sp.log(m1/mu**2) / (m1-m2)
            term2 = m2*sp.log(m2/mu**2) / (m2-m1)
            if m1_is_negative:
                term1 = -term1
            if m2_is_negative:
                term2 = -term2
            return term1 + term2"""
            #return (m1*sp.log(m1/mu**2)-m2*sp.log(m2/mu**2))/(m1-m2)

    # Calculate the Ni derivatives
    cprint(r"Calculating the $N_i$ derivatives...", show = show_procedure)
    n=0
    
    for i in range(8):
        for j in range(8):
            for a in range(8):
                NCW[i] += epsilon * coeff * R[i,j] * L3[a][a][j] * g(masses[a]) #M[a] * (sp.log(M[a]/mu**2)-kT+1/2)
                n += 1
                if n % 64 == 0:
                    cprint(n, "/ 512 ", end="\r", show = show_procedure)
      

    cprint(r"\nCalculating the $H_{ij}$ derivatives...", show = show_procedure)
    n=0
    
    Terms1 = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            term = 0
            for c in range(8):
                term += L4[c][c][i][j] * g(masses[c]) #M[c] * (sp.log(M[c]/mu**2)-kT+1/2)
                n += 1
                if n % 64 == 0:
                    cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            Terms1[i][j] = term
        
    Terms2 = [[0 for _ in range(8)] for _ in range(8)]
    for i in range(8):
        for j in range(8):
            term = 0
            for a in range(8):
                for b in range(8):
                    F = f(masses[a],masses[b])
                    term += L3[a][b][i] * L3[a][b][j] * (F-kT+1/2) 
                    n += 1
                    if n % 64 == 0:
                        cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            Terms2[i][j] = term 
           
    for k in range(8):
        for l in range(8):
            term = 0
            for i in range(8):
                for j in range(8):
                    term += coeff * epsilon * R[k,i] * R[l,j]  * (Terms2[i][j]+Terms1[i][j])
                    n += 1
                    if n % 64 == 0:
                        cprint(n, "/ 8704 ", end="\r", show = show_procedure)
            MCW[k][l] = term



    return sp.Matrix(NCW), sp.Matrix(MCW)

# Calculate and assign the VCT parameter values
def calculate_counterterm_values(model, V0_params_values=None, VEV_values=None, show_solution=False):
    #VCT_params, VCT_params_sol, VEV, VEV_values, M0, V0, fields, fields_mass, show_solution=False, 
    """ 
    V0: Should be evaluated for all values except for the fields
    """
    ##### SETUP #####
    # Unpacking
    VCT_params_eqs_sol = model.DATA.get("VCT_params_solution",None)
    assert VCT_params_eqs_sol is not None, "The analytical counterterm solutions has not been assigned"
    V0_params = model.V0_params
    VCT_params = model.VCT_params
    VEV = model.VEVs
    bgfields = model.bgfields
    fields = model.fields
    massfields = model.massfields
    V0 = model.V0
    M0 = model.DATA.get("M0", None)
    assert M0 is not None, "The M0 mass matrix has not been assigned"
    
    # param values
    if V0_params_values == None:
        V0_params_values = model.V0_params_values
    if VEV_values == None:
        VEV_values = model.VEV_values
    
    # Substitutions
    subs_VEV = {symb:val for symb,val in zip(VEV, VEV_values) if symb != 0}
    subs_bgfields_to_VEV = {symb:val for symb,val in zip(bgfields, VEV) if symb != 0}
    subs_fields_to_zero = {field:0 for field in fields}
    subs_params_values = {param:value for param, value in zip(V0_params, V0_params_values)}
    
    # Calculate M0 and V0
    M0_numerical_VEV = M0.subs(subs_fields_to_zero).subs(subs_bgfields_to_VEV).subs(subs_VEV | subs_params_values).evalf()
    V0_fdep_VEV = V0.subs(subs_bgfields_to_VEV).subs(subs_VEV | subs_params_values).evalf()
    
    # Calculate the derivatives
    NCW, MCW = calculate_VCW_derivatives(model, VEV_values[0], M0_numerical_VEV, V0_fdep_VEV)
    
    ##### CALCULATIONS #####
    # OBS! Also need to consider extra constraints values later
    #subs_VCT_params_values = {symb:value for symb, value in zip(VCT_params, VCT_params_sol)}
    
    # create Ni and Hij subs dict
    Ni = generate_vector(8, "N")
    Hij = generate_matrix(8, "H", symmetric=True)
    
    equations_Ni = [sp.Eq(Ni[i], NCW[i]) for i in range(8)]
    equations_Hij = [sp.Eq(Hij[i,j], MCW[i,j]) for i in range(8) for j in range(8) if i <= j]
    
    subs_Ni = eqs_to_dict(equations_Ni)
    subs_Hij = eqs_to_dict(equations_Hij)
    
    # Assign the values
    VCT_params_values = [0 for i in range(len(VCT_params))]
    subs_VEV = {symb:val for symb, val in zip(VEV,VEV_values)}
    for i, eq in enumerate(VCT_params_eqs_sol):
        VCT_params_values[i] = eq.rhs.subs(subs_VEV).subs(subs_Ni | subs_Hij).evalf()
                
    # Display the results
    if show_solution:
        print("========== Counterterm values ==========".center(50, "="))
        for symb, value in zip(VCT_params, VCT_params_values):
            display(Math(sp.latex(symb) + "=" + sp.latex(value)))
        
    return VCT_params_values

# Calc the effective masses
def _calculate_level1_masses(model, omega, M0_numerical, V0_fdep, MCT_numerical, sorting=True):
    
    _, MCW = calculate_VCW_derivatives(model, omega, M0_numerical, V0_fdep)
    
    Meff = M0_numerical + MCT_numerical + MCW
    
    masses, field_states, R = diagonalize_numerical_matrix(Meff, sorting=sorting)
    
    return masses 

#################### Potentials ####################

def generate_masses_gaugebosons_OLD(model, T=0, show_procedure=False, show_solution=False):
    
    # unpacking
    bgfield1 = model.bgfield1
    bgfield2 = model.bgfield2 
    
    # Symbols
    W1, W2, W3 = sp.symbols("W1 W2 W3", real=True)
    B = sp.Symbol("B", real=True)
    g1 = sp.Symbol("g1", real=True)
    g2 = sp.Symbol("g2", real=True)
    Wp = sp.symbols("W^+", real=True)
    Wm = sp.symbols("W^-", real=True)
    Z = sp.symbols("Z", real=True) 
    A = sp.symbols("A", real=True)
    
    # Gauge boson substitutions
    #WW = g2**2 * (W1 + sp.I*W2) * (W1 - sp.I*W2)
    WP = (W1 + sp.I*W2)
    WM = (W1 - sp.I*W2)
    ZZ = (g1*B - g2*W3)**2 
    AA = (g1*B + g2*W3)**2
    
    subs_gauge = {WP: Wp, 
                  WM:Wm, 
                  ZZ: Z**2* sp.sqrt(g1**2 + g2**2), 
                  AA: A**2* sp.sqrt(g1**2 + g2**2)}

    # Calculation

    D = sp.Matrix([[g1*B+g2*W3, g2*(W1 - sp.I*W2)],
                    [g2*(W1 + sp.I*W2), g1*B-g2*W3]])
        
    result1 = dagger(D*bgfield1) * (D*bgfield1)
    new_result1 = sp.Rational(1,2) * result1[0].subs(subs_gauge)
    result2 = dagger(D*bgfield2) * (D*bgfield2)
    new_result2 = sp.Rational(1,2) * result2[0].subs(subs_gauge)
    
    result = new_result1 + new_result2
    
    # masses
    mWp_sq = result.diff(Wp, Wp)/2
    mWm_sq = result.diff(Wm, Wm)/2
    mWpm_sq = result.diff(Wp, Wm)
    mZ_sq = result.diff(Z, Z)/2
    mA_sq = result.diff(A, A)/2
    
    # Display
    if show_procedure or show_solution:
        print(" Kinetic term ".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{kin}} =" + sp.latex(result)))
        
        print("========== Gauge boson masses ==========".center(50, "="))
        present(Math("m_{W^+}^2" + "=" + sp.latex(mWp_sq)))
        present(Math("m_{W^-}^2" + "=" + sp.latex(mWm_sq)))
        present(Math("m_{W_\pm}^2" + "=" + sp.latex(mWpm_sq)))
        present(Math("m_{Z}^2" + "=" + sp.latex(mZ_sq)))
        present(Math("m_{A}^2" + "=" + sp.latex(mA_sq)))
        print("Mass ratio:")
        present(Math(r"\cos^2(\theta_W) = \frac{m^2_{W_\pm}}{m^2_{Z}}" + "=" + sp.latex(mWpm_sq/mZ_sq)))
    
    
    """if T==0:
        m_W = const.g1/sp.sqrt(2) * (h1_bg.T * h1_bg)
        m_Z = sp.sqrt(const.g1**2 + const.g2**2)/sp.sqrt(2) * (h1_bg.T * h1_bg)
        m_A = 0
        return m_W, m_Z, m_A
    else:
        return 0,0,0"""
        
    return [mWpm_sq, mZ_sq, mA_sq]
    
def generate_masses_fermions_OLD(model, T=0, type="I", only_top=True):
    
    # coupling constants
    yt = sp.Symbol("y_t", real=True)
    yb = sp.Symbol("y_b", real=True)

    bgfield1 = model.bgfield1
    bgfield2 = model.bgfield2

    bgfield1_c = sp.Matrix([bgfield1[1], bgfield1[0]])
    bgfield2_c = sp.Matrix([bgfield2[1], bgfield2[0]])

    # Symbols
    tL = sp.symbols("t_L", real=True)
    bL = sp.symbols("b_L", real=True)
    tR = sp.symbols("t_R", real=True)
    bR = sp.symbols("b_R", real=True)
    QLt = sp.Matrix([[tL], [bL]])  
    
    if T==0:
        if only_top:
            QLt = sp.Matrix([[tL], [0]])  
            
            Lt1 = yb * dagger(QLt) * bgfield1 * bR + yt * dagger(QLt) * bgfield1_c * tR
            Lt2 = yb * dagger(QLt) * bgfield2 * bR + yt * dagger(QLt) * bgfield2_c * tR
            if type == "I":
                L_Yuk = Lt1[0]
            if type == "II":
                L_Yuk = Lt1[0] + Lt2[0]
            mt_sq = L_Yuk.diff(tR, tL)
            
            return [mt_sq]
        
    return 0

# Generate the analytical solution of the Coleman-Weinberg potential
def generate_CW_potential_OLD(model):
    
    
    # Coleman weinberg potential function
    def V_CW(masses, spins, dofs, mu, scheme_constants):

        V = 0
        for mass, spin, dof, C in zip(masses, spins, dofs, scheme_constants):
            
            mass_squared = mass
            display(mass_squared)
            # Handle zero and negative masses
            if mass_squared.is_zero:
                log_term = 0
            else:
                #log_term = sp.sign(mass_squared)*sp.log(sp.Abs(mass_squared) / mu**2)
                log_term = sp.log(mass_squared / mu**2)

            factor = (-1) ** (2 * spin) * dof / (64 * sp.pi**2)

            # Add this particle's contribution to the effective potential
            V += factor * mass_squared**2 * (log_term - C)
            display(V)

        return V
    #
    gauge_masses = generate_masses_gaugebosons(model)
    fermion_masses = generate_masses_fermions(model, type="I", only_top=True)
    
    display(gauge_masses, fermion_masses)
    
    # Constants
    v = const.v
    mu = v
    
    # Unpacking
    masses_higgs = model.DATA["M0_eigenvalues"]
    
    # mass eigenvalues
    # OBS, we are using transverse and longitudinal gauge bosons (same as regular gauge bosons for T=0)
    #if T==0:
    #m_W, m_Z, m_Gamma = gauge_masses
    #m_WT, m_WL, m_ZT, m_ZL, m_GammaT, m_GammaL = model.gauge_eigenvalues_higgs
    #m_h1, m_h2, m_h3, m_G0, m_Hp, m_Gp, m_Hm, m_Gm = higgs_masses
    #m_t = fermion_masses[0]
    masses_gauge = gauge_masses #[m_W, m_Z, m_Gamma]
    masses_fermions = fermion_masses #[0] #[m_t]
    masses = masses_higgs + masses_gauge + masses_fermions
    
    spin_higgs, spin_fermions = 0, 1/2
    spins_higgs = [0 for _ in masses_higgs]
    #spins_gauge = [1,1,1,1,0,0]
    spins_gauge = [1,1,0]
    spins_fermions = [1/2 for _ in masses_fermions]
    spins = spins_higgs + spins_gauge + spins_fermions

    dof_higgs = 1
    dofs_higgs = [1 for _ in masses_higgs] 
    #dofs_gauge = [4,2,2,1,2,1]
    dofs_gauge = [4,2,1]
    dofs_fermions = [12]
    dofs = dofs_higgs + dofs_gauge + dofs_fermions
    
    scheme_constants_higgs, scheme_constants_gauge, scheme_constants_fermions = 3/2, 3/2, 5/6
    scheme_constants_higgs = [3/2 for _ in masses_higgs]
    scheme_constants_gauge = [3/2 for _ in masses_gauge]
    scheme_constants_fermions = [5/6 for _ in masses_fermions]
    scheme_constants = scheme_constants_higgs + scheme_constants_gauge + scheme_constants_fermions

    # Generate the potential
    V_cw_expr = V_CW(masses, spins, dofs, mu, scheme_constants)
    
    return V_cw_expr

def _generate_T_potental_OLD():
    pass

# Generate effective potential (with thermal corrections)
def _generate_effective_potential_OLD(T=0):
    pass

#################### Datasets ####################
# Add confirmation when overwriting data, or add option to overwrite

# Calculate and store data level 0 (potential and masses)
def calculate_data2D_level0(model, free_bgfield,
                          N, Xrange,
                          calc_potential=True, calc_mass=True, sorting=True):
    # unpacking
    V0_params = model.V0_params
    V0_params_values = model.V0_params_values
    bgfields = model.bgfields
    bgfield_values = model.bgfield_values
    VEV = model.VEVs
    VEV_values = model.VEV_values
    fields = model.fields
    M0 = model.DATA["M0"]
    V0 = model.V0
    
    if not calc_potential and not calc_mass:
        raise ValueError("Need to choose at least one calculation to perform.")
    
    subs_VEV = {symb:val for symb,val in zip(VEV, VEV_values)}
    subs_bgfields = {symb:val for symb,val in zip(bgfields, bgfield_values) if symb != free_bgfield}

    X = np.linspace(Xrange[0], Xrange[1], N)
    Y1 = None
    Y2 = None
    
    if calc_potential:
        subs_fields_to_zero = {field:0 for field in fields}
        subs_params_values = {param:value for param, value in zip(V0_params, V0_params_values)}
        
        Y1 = [0 for i in range(N)]
        V0 = V0.subs(subs_fields_to_zero).subs(subs_bgfields | subs_VEV | subs_params_values)
        for i,x in enumerate(X):
            V0_numerical = V0.subs({free_bgfield: x})
            Y1[i] = V0_numerical

    if calc_mass:
        Y2 = [0 for i in range(N)]
        M0 = M0.subs(subs_bgfields | subs_VEV | subs_params_values)
        for i,x in enumerate(X):
            M0_numerical = M0.subs({free_bgfield: x})
            M0_numerical = np.array(M0_numerical).astype(np.float64)
            masses = _calculate_level0_masses(M0_numerical, sorting=sorting)
            masses_new = np.zeros(8)
            for j, m in enumerate(masses):
                masses_new[j] = np.sign(m)*np.sqrt(np.abs(m))
            Y2[i] = masses_new
        Y2 = np.transpose(Y2)

    return X, Y1, Y2

# Calculate and store data level 1 (potential and masses)
def calculate_data2D_level1(model:object, free_bgfield, N:int, Xrange:list, calc_potential=True, calc_mass=True):
    
    ##### SETUP #####
    # Unpacking
    V0_params = model.V0_params
    V0_params_values = model.V0_params_values
    VCT_params = model.VCT_params
    VCT_params_values = model.VCT_params_values
    bgfields = model.bgfields
    bgfield_values = model.bgfield_values
    VEV = model.VEVs
    VEV_values = model.VEV_values
    fields = model.fields
    massfields = model.massfields
    M0 = model.DATA["M0"]
    V0 = model.V0
    MCT = model.DATA["MCT"]
    
    # Substitutions
    subs_fields_to_zero = {field:0 for field in fields}
    subs_VEV = {symb:val for symb,val in zip(VEV, VEV_values)}
    subs_bgfields = {symb:val for symb,val in zip(bgfields, bgfield_values) if symb != free_bgfield}
    subs_V0_params_values = {param:value for param, value in zip(V0_params, V0_params_values)}
    subs_VCT_params_values = {symb:val for symb,val in zip(VCT_params, VCT_params_values)}
    subs_dict = subs_VEV | subs_bgfields | subs_V0_params_values | subs_VCT_params_values
    
    #display(subs_bgfields)
    
    # X-axis
    X = np.linspace(Xrange[0], Xrange[1], N)
    Y1 = None
    Y2 = None
    
    if calc_potential:
        pass
    
    if calc_mass:
        M0_numerical = M0.subs(subs_fields_to_zero).subs(subs_dict)
        MCT_numerical = MCT.subs(subs_fields_to_zero).subs(subs_dict)
        V0_fdep = V0.subs(subs_dict)
        
        Y2 = [0 for i in range(N)]
        # add time estimate
        v = const.v
        for i,x in enumerate(X):
            print(f"Progress: {i+1}/{N} at x={x:0.2f}={x/v:0.2f}v", end="\r")
            M0_numerical_x = M0_numerical.subs({free_bgfield:x}).evalf()
            MCT_numerical_x = MCT_numerical.subs({free_bgfield:x}).evalf()
            V0_fdep_x = V0_fdep.subs({free_bgfield:x}).evalf()
            
            masses = _calculate_level1_masses(model, x, M0_numerical_x, V0_fdep_x, MCT_numerical_x, sorting=True)
            masses_new = np.zeros(8)
            for j, m in enumerate(masses):
                masses_new[j] = np.sign(m)*np.sqrt(np.abs(m))
            Y2[i] = masses_new
        Y2 = np.transpose(Y2)
    
    return X, Y1, Y2         
    
#################### Plotting ####################

# Load and plot data level 0 (potential and masses)
def load_and_plot_data2D(model:object, name="", level=0, plot_potential=True, plot_masses=True, save_fig:bool=True, 
                         Xrange=None, Yrange=None):
    """Loads and plots data for the model"""
    # add custom legend names

    v = const.v
    levelstr = "level" + str(level)
    if name != "":
        name = name + "_"

    if plot_potential:
        data_potential = load_data(f"{name}potential_{levelstr}_data", model.path_data)
    if plot_masses:
        data_mass = load_data(f"{name}mass_{levelstr}_data", model.path_data)
    
    if name != "":
        name = name + "_"
    
    if plot_potential:
        # Plot potential
        plt.figure()
        plt.plot(np.array(data_potential["omega"])/v, np.array(data_potential["V"])/v**4)
        plt.xlabel("omega")
        plt.ylabel("V")
        plt.title("Potential")
        plt.grid()
        if save_fig:
            plt.savefig(os.path.join(model.path_plots, f"{name}{levelstr}_potential.png"))
        plt.show()
    
    if plot_masses:
        # Plot masses
        plt.figure(figsize=(10,6), dpi=500)
        
        Xlim = [np.array(data_mass["omega"][0])/v, np.array(data_mass["omega"][-1])/v]
        Ylim = [1.1*min([min(data_mass[f"m_{i+1}"]) for i in range(8)]), 1.0*max([max(data_mass[f"m_{i+1}"]) for i in range(8)])]
        plt.xlim(Xlim[0], Xlim[1])
        plt.ylim(Ylim[0], Ylim[1])
        if Xrange is not None:
            plt.xlim(Xrange[0], Xrange[1])
        if Yrange is not None:
            plt.ylim(Yrange[0], Yrange[1])
        
        for i in range(8):
            linestyle = "-"
            linewidth = 1.5
            if i > 3:
                linestyle = "--"
                linewidth = 2.5
            plt.plot(np.array(data_mass["omega"])/v, data_mass[f"m_{i+1}"], label=f"m_{i+1}", linestyle=linestyle, linewidth=linewidth)
        plt.xlabel("omega")
        plt.ylabel("m")
        plt.title("Masses")

        
        plt.legend()
        plt.grid()
        
        if save_fig:
            plt.savefig(os.path.join(model.path_plots, f"{name}{levelstr}_masses.png"))
        plt.show()

