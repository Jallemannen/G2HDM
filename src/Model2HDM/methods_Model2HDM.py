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
        path = os.path.join(project_root, "Models", "Saved_models", f"{name}", "model_data", "model.pkl")

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

def rotate_basis(model, R):
    pass

#################### Solvers ####################
# show = "solution", "procedure", "none" ?

def generate_tadpole_equations(potential, fields, subs_dict, remove_trivial_eqs=True)->list:
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
    tadpole_eqs = generate_tadpole_equations(V0_VEV, fields, fields_to_zero_dict, False)
    
    # Solving tadpole eqs
    sol_tadpole_eqs = linear_solve(tadpole_eqs, params)
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
            #M0_charged_eigenvalues = [val for val in list(M_charged.eigenvals())]
            M0_charged_eigenvalues = [eigen for eigen, multiplicity in M_charged.eigenvals().items()
                                        for _ in range(multiplicity)] 
            
            for i, val in enumerate(M0_charged_eigenvalues):
                cdisplay(Math(f"m_{i+1} = " + sp.latex(val)), show=show_solution)
            
            cprint("Neutral mass eigenvalues:", show=show_solution)
            #M0_neutral_eigenvalues = [val for val in list(M_neutral.eigenvals())]
            M0_neutral_eigenvalues = [eigen for eigen, multiplicity in M_neutral.eigenvals().items()
                                        for _ in range(multiplicity)] 
            
            for i, val in enumerate(M0_neutral_eigenvalues):
                cdisplay(Math(f"m_{i+1} = " + sp.latex(val)), show=show_solution)
            
            # neutral first
            M0_eigenvalues = M0_neutral_eigenvalues + M0_charged_eigenvalues 
            
        else:
        
            cprint("Mass eigenvalues:", show=show_solution)
            #M0_eigenvalues = [val for val in list(M0.eigenvals())]
            M0_eigenvalues = [eigen for eigen, multiplicity in M0.eigenvals().items()
                                for _ in range(multiplicity)] 
            
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
    tadpole_eqs = generate_tadpole_equations(VCT.subs(subs_bgfields_to_VEV), fields, subs_fields_to_zero, False)
    
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
    solution = linear_solve(equations, VCT_params + NH_variables, include_trivial_solutions=True)

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

#################### Potentials ####################

def generate_kinetic_term_gauge(model, using_SM_fields=False):
    # unpacking
    field1 = model.field1
    field2 = model.field2 
    
    # Symbols
    W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
    W0, Wp, Wm = sp.symbols("W_0 W^+ W^-", real=True)
    B = sp.Symbol("B", real=True)
    g1 = sp.Symbol("g_1", real=True)
    g2 = sp.Symbol("g_2", real=True)
    
    # mass states
    Wp = sp.symbols("W^+", real=True)
    Wm = sp.symbols("W^-", real=True)
    Z = sp.symbols("Z^0", real=True) 
    A = sp.symbols("A", real=True)
    
    # Gauge boson substitutions
    if using_SM_fields:
        WP =  g2*Wp #g2*(-W1 + sp.I*W2)
        WM =  g2*Wm #g2*(-W1 - sp.I*W2)
        Z0 = Z * (g1**2 + g2**2)**sp.Rational(1,2) #(-g1*B + g2*W3) * (g1**2 + g2**2)**sp.Rational(1,2)
        A0 = A * (g1**2 + g2**2)**sp.Rational(1,2) #(g1*B + g2*W3) * (g1**2 + g2**2)**sp.Rational(1,2)

        D = sp.I*sp.Rational(1,2)*sp.Matrix([[A0, -WP],
                                            [-WM, -Z0]])
        D_dagger = -sp.I*sp.Rational(1,2)*sp.Matrix([[A0, -WP],
                                                    [-WM, -Z0]]) # transpose + complex conjugate Wm->Wp
    else:
        D = sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, g2*(W1 - sp.I*W2)],
                    [g2*(W1 + sp.I*W2), g1*B-g2*W3]])
        D_dagger = dagger(D)
        
        #D = sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, -sp.sqrt(2)*Wp],
        #                                    [-sp.sqrt(2)*Wm, g1*B-g2*W0,]])
        #D_dagger = -sp.I*sp.Rational(1,2)*sp.Matrix([[g1*B+g2*W3, -sp.sqrt(2)*Wp],
        #                                    [-sp.sqrt(2)*Wm, g1*B-g2*W0,]])
        
    result1 = (dagger(field1)*D_dagger) * (D*field1)
    
    result2 = (dagger(field2)*D_dagger) * (D*field2)
    
    result = result1[0] + result2[0]
    
    return result

def generate_yukawa_term_fermions(model, type="I", only_top=True):
    
    # coupling constants
    yt = sp.Symbol("y_t", real=True, positive=True)
    yb = sp.Symbol("y_b", real=True, positive=True)

    field1 = model.field1
    field2 = model.field2

    field1_c = sp.Matrix([field1[1], field1[0]])
    field2_c = sp.Matrix([field2[1], field2[0]])

    # Symbols
    tL = sp.symbols("t_L")
    bL = sp.symbols("b_L")
    tR = sp.symbols("t_R")
    bR = sp.symbols("b_R")
    ctL = sp.Symbol(r"\bar{t}_L")
    cbL = sp.Symbol(r"\bar{b}_L")
    ctR = sp.Symbol(r"\bar{t}_R")
    cbR = sp.Symbol(r"\bar{b}_R")
    QLt_dagger = sp.Matrix([[ctL], [cbL]]).T
    

    if only_top:
        QLt_dagger = sp.Matrix([[ctL], [0]]).T
    
    L_Yuk = 0
    Lt1 = yb * QLt_dagger * field1 * bR + yt * QLt_dagger * field1_c * tR
    Lt2 = yb * QLt_dagger * field2 * bR + yt * QLt_dagger * field2_c * tR
        
    if type == "I":
        L_Yuk += Lt1[0]
    if type == "II":
        L_Yuk += Lt1[0] + Lt2[0]
        
    return L_Yuk


def generate_masses_gaugebosons(model, show_procedure=False, show_solution=False):
    
    L_kin = generate_kinetic_term_gauge(model, using_SM_fields=False)
    subs_fields = {f:0 for f in model.fields}
    
    W1, W2, W3 = sp.symbols("W_1 W_2 W_3", real=True)
    #W0, Wp, Wm = sp.symbols("W_0 W^+ W^-", real=True)
    B = sp.Symbol("B", real=True)
    gaugefields = [B, W1, W2, W3]
    #gaugefields = [B, W0, Wp, Wm]
    #fields_gauge = [B, W0, Wp/sp.sqrt(2), Wm/sp.sqrt(2)]
    
    H = sp.hessian(L_kin.subs(subs_fields).expand(), gaugefields)
    #display(H)
    eigen_data = H.eigenvects()
    eigenvalues = [ev[0] for ev in eigen_data] #*sp.Rational(1,2) 
    eigenvectors = [ev[2] for ev in eigen_data]
    #display(eigenvalues, eigenvectors)
    #display(eigenvectors)

    evals = []
    evects = []
    states = []
    for eval, evec in zip(eigenvalues, eigenvectors):
        for vec in evec:
            evals.append(eval)
            evects.append(vec)
            states.append((sp.Matrix(vec).T * sp.Matrix(gaugefields))[0])
            
    R = sp.Matrix.hstack(*evects)
    masses = evals
    # is eval is neg --> set neg eigenstate instead
    
    # Filter negative masses 
    """masses1 = masses.copy()
    masses2 = masses.copy()
    states_sorted = []
    masses_sorted = []
    for i, mass1 in enumerate(masses1):
        term_mass = 0
        term_state = 0
        for j, mass2 in enumerate(masses2):
            if mass2.coeff(mass1) == -1:
                #assuming two states are mixed:
                term_mass = sp.Abs(mass2)*2
                term_state = (states[i]*-states[j]/2).expand()
        masses_sorted.append(term_mass)    
        states_sorted.append(term_state)   
    """          

    if show_procedure or show_solution:
        print(" Kinetic term ".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{kin}} =" + sp.latex(L_kin.subs(subs_fields))))
        
        print("========== Gauge boson masses ==========".center(50, "="))
        for i in range(int(len(masses))):
            present(Math("m^2" + "=" + sp.latex(masses[i])))
            present(Math(sp.latex(states[i]**2)))
        
        
        """# Mult fields if they have the same eigenvalue. 
        print("========== Gauge boson masses ==========".center(50, "="))
        present(Math("m_{W_\pm}^2" + "=" + sp.latex(masses[1]+masses[2])))
        present(Math("m_{Z}^2" + "=" + sp.latex(masses[3])))
        present(Math("m_{A}^2" + "=" + sp.latex(masses[0])))
        print("Mass ratio:")
        present(Math(r"\cos^2(\theta_W) = \frac{m^2_{W_\pm}}{m^2_{Z}}" + "=" + sp.latex((masses[1]+masses[2])/masses[3])))
        """
        
        model.DATA["M_gauge"] = H
        model.DATA["M_gauge_eigenvalues"] = masses
        
    return masses, states, R 
    
def generate_masses_fermions(model, type="I", only_top=True, show_procedure=False, show_solution=False): 
    # coupling constants

    subs_fields = {f:0 for f in model.fields}
    L_yuk = generate_yukawa_term_fermions(model, type="I", only_top=only_top).subs(subs_fields)
    display(L_yuk)
    # Symbols
    tL = sp.symbols("t_L")
    bL = sp.symbols("b_L")
    tR = sp.symbols("t_R")
    bR = sp.symbols("b_R")
    ctL = sp.Symbol(r"\bar{t}_L")
    cbL = sp.Symbol(r"\bar{b}_L")
    ctR = sp.Symbol(r"\bar{t}_R")
    cbR = sp.Symbol(r"\bar{b}_R")
    
    if only_top:
        fields = sp.Matrix([tL, tR, bL, bR])
        fields_conjugated = sp.Matrix([ctL, ctR, cbL, cbR])
        H = sp.Matrix([[0 for _ in fields] for _ in fields])
        
        #display(L_yuk.diff(ctL, tR), L_yuk.diff(ctL, tR))
        for i in range(len(fields)):
            #display(L_yuk.diff(fields_conjugated[i]), fields_conjugated[i])
            for j in range(len(fields)):
                H[i,j] += L_yuk.diff(fields[i], fields_conjugated[j])/2
                H[i,j] += L_yuk.diff(fields_conjugated[i], fields[j])/2
        #display(H)
        eigen_data = H.eigenvects()
        eigenvalues = [ev[0]*sp.Rational(1,2) for ev in eigen_data]
        eigenvectors = [ev[2] for ev in eigen_data]
        
    masses = []
    states = []
    states_conjugated = []
    for eval, evec in zip(eigenvalues, eigenvectors):
        for vec in evec:
            masses.append(eval)
            states.append(vec.T * sp.Matrix(fields))
            states_conjugated.append(vec.T * sp.Matrix(fields_conjugated))
            
    if show_procedure or show_solution:
        print("========== Yukawa term ==========".center(50, "="))
        display(Math(r"\mathcal{L}_{\text{Yuk}} =" + sp.latex(L_yuk.subs(subs_fields))))
        # Mult fields if they have the same eigenvalue. 
        for i in range(int(len(masses))):
            print("========== Fermion masses ==========".center(50, "="))
            present(Math("m^2" + "=" + sp.latex(masses[i])))
            state = (states_conjugated[i]*states[i]).expand()
            display(Math(sp.latex(state)))
        
    return masses, states

# Generate the analytical solution of the Coleman-Weinberg potential (not optimal..., try numerical instead)
def generate_VCW(model, only_top=True):
    
    
    # Constants
    v = const.v
    mu = v
    
    # Coleman weinberg potential function
    def V_CW(mass, dof, C):

        mass_squared = mass
        display(mass_squared)
        # Handle zero and negative masses
        if mass_squared.is_zero:
            log_term = 0
        else:
            #log_term = sp.sign(mass_squared)*sp.log(sp.Abs(mass_squared) / mu**2)
            log_term = sp.log(mass_squared / mu**2)

        factor = dof / (64 * sp.pi**2)

        # Add this particle's contribution to the effective potential
        return factor * mass_squared**2 * (log_term - C)
    
    
    #
    masses_gauge, _ = generate_masses_gaugebosons(model)
    masses_fermions, _ = generate_masses_fermions(model, type="I", only_top=only_top)
    
    
    #display(masses_gauge, masses_fermions)
    
    
    
    # Unpacking
    masses_higgs = model.DATA["M0_eigenvalues"]
    
    # mass eigenvalues
    masses = masses_higgs + masses_gauge + masses_fermions
    
    # Spins
    spins_higgs = [0 for _ in masses_higgs]
    spins_gauge = [1 for _ in masses_gauge]
    spins_fermions = [1/2 for _ in masses_fermions]

    # Dof
    dofs_higgs = [1 for _ in masses_higgs] 
    dofs_gauge = []
    for mass in masses_gauge:
        if mass.is_zero:
            dofs_gauge.append(2)
        else:
            dofs_gauge.append(3)
            
    dofs_fermions = [12]
    
    # Scheme constants
    scheme_constants_higgs = [3/2 for _ in masses_higgs]
    scheme_constants_gauge = [3/2 for _ in masses_gauge]
    scheme_constants_fermions = [5/6 for _ in masses_fermions]

    # Generate the potential
    Masses = masses_higgs + masses_gauge #+ masses_fermions
    Dofs = dofs_higgs + dofs_gauge #+ dofs_fermions
    Scheme_constants = scheme_constants_higgs + scheme_constants_gauge #+ scheme_constants_fermions
    V_cw_expr = 0
    for masses, dofs, scheme_constants in zip(Masses, Dofs, Scheme_constants):
        V_cw_expr += V_CW(masses, dofs, scheme_constants)
    
    return V_cw_expr
