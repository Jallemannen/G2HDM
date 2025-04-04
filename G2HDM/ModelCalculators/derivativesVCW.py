import sympy as sp
import symengine as se
import numpy as np


# Class for calculating the Coleman-Weinberg potential derivatives
class CWPotentialDerivativeCalculator:
    """ Calculates the Coleman-Weinberg potential derivatives for a given model 
    and a specific background field value, given at the initialization of the object."""
    
    # Constants
    threshold = 1e-10  
    epsilon = 1/(4*np.pi)**2
    
    def __init__(self, MDC:object, model:object, subs_bgfield_values:dict, regulator=246**2, scale=246): #
        
        self.model = model
        self.MDC = MDC
        
        # Constants
        self.mu = scale
        
        # Substitutions
        self.subs_bgfield_values = subs_bgfield_values
        
        # Masses
        self.masses_higgs, self.R_higgs = MDC.calculate_masses_higgs(subs_bgfield_values) #diagonalize_numerical_matrix(M0_numerical, sorting=True)
        self.masses_gauge, self.R_gauge = MDC.calculate_masses_gauge(subs_bgfield_values)
        #self.masses_fermions, self.R_fermions = MDC.calculate_masses_fermions(subs_bgfield_values)

        ####################################
        # TEST!!!
        #self.masses_higgs = [mass for i, mass in enumerate(self.masses_higgs) if i in [1,2,3,6,7]]
        """
        new_masses_higgs = []
        for i, mass in enumerate(self.masses_higgs):
            if i in [0,4,5,6,7] : #[0,4,5]: #[1,2,3,6,7] # [0,4,5,6,7] works
                new_masses_higgs.append(mass)
            else:
                new_masses_higgs.append(0)
        self.masses_higgs = new_masses_higgs
        """
        #self.masses_higgs[0] = self.masses_higgs[0] * 0.5 #2/3
        #self.masses_higgs[1] = self.masses_higgs[1] * 0.5 #2/3
        #self.masses_higgs[2] = self.masses_higgs[2] * 0.5 #2/3
        #self.masses_higgs[3] = self.masses_higgs[3] * 0.5 #2/3
        #self.masses_higgs = [mass*0.7 for mass in self.masses_higgs]
        ####################################

        # Add a regulator to the masses
        #display(self.masses_higgs)
        self.regulator = regulator
        self.masses_higgs = self.masses_higgs #+ self.regulator
        #display(self.masses_higgs)

        # Fields
        #self.massStates_higgs = sp.Matrix(self.R_higgs) * sp.Matrix(model.massfields)
        #self.massStates_gauge = sp.Matrix(self.R_gauge) * sp.Matrix(model.massfields_gauge)
        self.massStates_higgs = self.R_higgs @ model.massfields
        self.massStates_gauge = self.R_gauge @ model.massfields_gauge
        #self.massStates_fermions = sp.Matrix(self.R_fermions) * sp.Matrix(model.massfields_fermions)

        #print("Mass states:")
        #display(sp.Matrix(self.massStates_higgs))
        
        # Remove small coefficients
        def remove_small_coefficients(vec):
            updated_vec = []
            for term in vec:
                # Extract the terms
                coefficient = term.as_coefficients_dict()

                # Replace small coefficients with zero
                new_element = 0
                for symb,coeff in coefficient.items():
                    if sp.Abs(coeff) > 1e-10:
                        new_element += coeff*symb

                updated_vec.append(new_element)
            return np.array(updated_vec)
        
        #self.massStates_higgs = remove_small_coefficients(self.massStates_higgs)
        
        #print("Mass states:")
        ##display(sp.Matrix(remove_small_coefficients(self.massStates_higgs)))
        #display(sp.Matrix(remove_small_coefficients(self.R_higgs.T @ model.massfields)).subs({sp.Symbol(f"h_{i}",real=True):sp.Symbol(f"\phi_{i}",real=True) for i in range(8)}))
        
        # Potentials (This is a time sink)
        subs_massStates_higgs = {field:state for field,state in zip(model.fields, self.massStates_higgs)}
        subs_massStates_gauge = {field:state for field,state in zip(model.fields_gauge, self.massStates_gauge)}
        #self.V0 = MDC.V0_simplified.subs(subs_bgfield_values | subs_massStates_higgs).expand() #).evalf().subs(
        #self.L_kin = MDC.L_kin_simplified.subs(subs_bgfield_values | subs_massStates_gauge).expand()
        #self.L_yuk = MDC.L_yuk_simplified.subs(subs_bgfield_values).evalf().subs({field:state for field,state in zip(model.massfields_fermions, self.massStates_fermions)})
        #self.V0 = MDC.V0_simplified.xreplace(subs_bgfield_values | subs_massStates_higgs).expand() 
        
        #self.V0 = MDC.V0_simplified.xreplace(subs_bgfield_values) # ca 0.2s
        #self.V0 = custom_fast_expand_ultra(self.V0.xreplace(subs_massStates_higgs), model.massfields) # This takes the logest (ca 0.8-1s)
        
        #t0 = time.time()
        V0_se = se.sympify(MDC.V0_simplified).xreplace(subs_bgfield_values) 
        V0_se = V0_se.subs(subs_massStates_higgs).expand()
        self.V0 = se.sympify(sp.sympify(V0_se).xreplace({sp.I:0})) #V0_se #sp.sympify(V0_se)
        #self.V0 = V0_se
        #print("Time for V0: ", time.time()-t0)
        #display(self.V0)
        #display(V0_se.as_coefficients_dict())
        
        L_kin_se = se.sympify(MDC.L_kin_simplified).xreplace(subs_bgfield_values | subs_massStates_higgs | subs_massStates_gauge)
        L_kin_se = L_kin_se.expand()#.evalf()
        self.L_kin = se.sympify(sp.sympify(L_kin_se).xreplace({sp.I:0})) #L_kin_se.subs(se.I, 0)  #se.Reals(L_kin_se) #se.sympify(sp.re(sp.sympify(L_kin_se))) #L_kin_se.xreplace({se.I: 0}) #removes complex and simplifies # #L_kin_se #sp.sympify(L_kin_se)
        #display(MDC.L_kin_simplified.expand())
        #print(se.sympify(MDC.L_kin_simplified))
        #display(subs_bgfield_values | subs_massStates_higgs | subs_massStates_gauge)
        #print(self.L_kin.xreplace(se.I, 0) )
        #display(sp.sympify(L_kin_se))
        #self.L_kin = MDC.L_kin_simplified.xreplace(subs_bgfield_values | subs_massStates_gauge).expand()
        
        # Couplings
        self.L3_higgs = self.calculate_couplings3(self.V0, model.massfields, model.massfields)
        self.L3_gauge = self.calculate_couplings3(self.L_kin, model.massfields_gauge, model.massfields)
        #self.L3_fermions = self.calculate_couplings3(self.L_yuk, model.fields_fermions, model.massfields)
        
        #display(self.L3_gauge )
        
        self.L4_higgs = self.calculate_couplings4(self.V0, model.massfields, model.massfields) # ca 0.15s
        self.L4_gauge = self.calculate_couplings4(self.L_kin, model.massfields_gauge, model.massfields)
        #self.L4_fermions = self.calculate_couplings4()
        
        #print("Debug")
        #display(self.L4_higgs[0,0,0])
        #display(sp.Matrix(self.R_higgs))
        #display(self.masses_higgs)

    def first_derivative(self):
        # First derivative terms
        NCW_higgs = self.Ni(L3=self.L3_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        NCW_gauge = self.Ni(L3=self.L3_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        NCW_fermions = np.zeros(8) #Ni(L3, masses, kT, coeff)
        NCW = NCW_higgs + NCW_gauge + NCW_fermions
        NCW =  self.epsilon * self.R_higgs @ NCW # numpy matrix mul (j to i)  #sp.Matrix(NCW)
        #NCW = self.epsilon * NCW
        NCW = np.where(np.abs(NCW) < self.threshold, 0, NCW)
        return NCW #sp.Matrix(NCW) 
    
    def second_derivative(self):
        MCW_higgs = self.Hij(L3=self.L3_higgs, L4=self.L4_higgs, masses=self.masses_higgs, kT=3/2, coeff=1/2)
        MCW_gauge = self.Hij(L3=self.L3_gauge, L4=self.L4_gauge, masses=self.masses_gauge, kT=3/2, coeff=3/2)
        MCW_fermions = np.zeros((8,8)) # Hij(L3, L4, masses, kT, coeff)
        MCW = MCW_higgs + MCW_gauge + MCW_fermions
        #display(sp.Matrix(MCW_higgs), sp.Matrix(MCW_gauge))
        MCW = self.epsilon * self.R_higgs @ (MCW+MCW.T)/2 @ self.R_higgs.T
        #MCW = self.epsilon * (MCW+MCW.T)/2 
        MCW = np.where(np.abs(MCW) < self.threshold, 0, MCW)
        

        return MCW #sp.Matrix(MCW)
        
    ####### Helper functions #######

    # log term
    def f1(self, msq, mu):
        sign = 1
        if msq < 0:
            msq = -msq
            sign = -1
            return 0 #TEST
            
        if msq==0: #.is_zero:
            result = 0
        else:
            # Not IR div since this will be mult by the mass later
            # Add threshold here?
            # Apply IR regulator 
            msq += self.regulator
            result = sign*np.log(msq/mu**2) #m * (sp.log(m/mu**2)-kT+1/2)
        return result

    # log term 2, with regulator f2
    def f2(self, m1sq, m2sq, mu):
        #return 1
        sign1 = 1
        sign2 = 1
        if m1sq < 0:
            m1sq = -m1sq
            sign1 = -1
            return 0 #TEST
        if m2sq < 0:
            m2sq = -m2sq
            sign2 = -1
            return 0 #TEST
        
        # Apply IR regulator to all masses
        m1sq_R = m1sq + self.regulator
        m2sq_R = m2sq + self.regulator
        
        # calc
        log1 = 0
        log2 = 0
        """if m1sq == 0 and m2sq == 0:
            return 1
        if np.abs(m1sq-m2sq)<1e-5:
            return 1
        if m1sq != 0:
            log1 = sign1*np.log(m1sq_R/mu**2)
        if m2sq != 0: #and sp.Abs(m1sq-m2sq)>1e-5:
            log2 = sign2*np.log(m2sq_R/mu**2)
        else:
            return 1 + log1
            
        return (m1sq*log1 - m2sq*log2)/(m1sq-m2sq)
        """
        
        if m1sq == 0 and m2sq == 0:
            return 1
        if m1sq != 0:
            log1 = sign1*sp.log(m1sq_R/mu**2)
            log2 = 0
        
        if sp.Abs(m1sq-m2sq)>1e-5: # add threshold
            log2 = 0  
            if m2sq != 0:
                log2 = sign2*sp.log(m2sq_R/mu**2)
            if m1sq == 0:
                return log2
            elif m2sq == 0:
                return log1
            else:
                #print(log1, m2sq/m1sq, (m1sq*sp.Abs(log1) - m2sq*sp.Abs(log2))/(m1sq-m2sq))
                return (m1sq*sp.Abs(log1) - m2sq*sp.Abs(log2))/(m1sq-m2sq) #log1/(1-m2sq/m1sq)
        else: 
            return 1 + log1
    
    # Calculate trilinear couplings (Optimized)
    # Move to separate file?
    #@jit
    def calculate_couplings3(self, V, fields1, fields2):
        n1 = len(fields1)
        n2 = len(fields2)
        fields1 = [se.sympify(f) for f in fields1]
        fields2 = [se.sympify(f) for f in fields2]
        #L3 = np.empty((n1, n1, n2), dtype=object)
        L3 = np.zeros((n1, n1, n2), dtype=np.float64)
        fields_to_zero = {f: 0 for f in (fields1 + fields2)}
        # A cache to avoid recomputing the same fieldterm.
        cache = {}
        # Create a Poly object over all the fields.
        ##polyV = V.as_poly() #sp.Poly(V, list(set(fields1 + fields2)))
        coeff_dict = V.as_coefficients_dict()
        
        
        # Loop only over i <= j assuming symmetry in the first two indices.
        for i in range(n1):
            for j in range(i, n1):
                prod1 = fields1[i] * fields1[j]  # common product
                for k in range(n2):
                    fieldterm = prod1 * fields2[k]
                    if fieldterm not in cache:
                        # Expand and substitute only once per unique fieldterm.
                        ##term = V.coeff(fieldterm).subs(fields_to_zero) #Costly part
                        ##coeff = polyV.coeff_monomial(fieldterm)
                        ##term = coeff.subs(fields_to_zero)
                        term = coeff_dict.get(fieldterm, 0)
                        cache[fieldterm] = np.float64(term) #sp.re(term)
                    value = cache[fieldterm]
                    L3[i, j, k] = value
                    if i != j:
                        # Use symmetry to assign the swapped indices.
                        L3[j, i, k] = value
        return L3
    
    # calculate quartic couplings (Optimized)
    #@jit
    def calculate_couplings4(self, V, fields1, fields2):
        n1 = len(fields1)
        n2 = len(fields2)
        fields1 = [se.sympify(f) for f in fields1]
        fields2 = [se.sympify(f) for f in fields2]
        #L4 = np.empty((n1, n1, n2, n2), dtype=object)
        L4 = np.zeros((n1, n1, n2, n2), dtype=np.float64)
        fields_to_zero = {f: 0 for f in (fields1 + fields2)}
        # A cache to avoid recomputing the same fieldterm.
        cache = {}
        # Create a Poly object over all the fields.
        ##polyV = V.as_poly() #sp.Poly(V, list(set(fields1 + fields2)))
        coeff_dict = V.as_coefficients_dict()
        
        # Loop over indices in the first group assuming symmetry: i <= j.
        for i in range(n1):
            for j in range(i, n1):
                prod1 = fields1[i] * fields1[j]
                # Loop over indices in the second group assuming symmetry: k <= l.
                for k in range(n2):
                    for l in range(k, n2):
                        fieldterm = prod1 * fields2[k] * fields2[l]
                        if fieldterm not in cache:
                            ##term = V.coeff(fieldterm).subs(fields_to_zero) #Costly part
                            ##coeff = polyV.coeff_monomial(fieldterm)
                            ##term = coeff.subs(fields_to_zero)
                            term = coeff_dict.get(fieldterm, 0)
                            cache[fieldterm] = np.float64(term)
                        value = cache[fieldterm]
                        L4[i, j, k, l] = value
                        # Now fill in all the symmetric entries.
                        if i != j:
                            L4[j, i, k, l] = value
                        if k != l:
                            L4[i, j, l, k] = value
                        if i != j and k != l:
                            L4[j, i, l, k] = value
        return L4
    
    #@jit
    def Ni(self, L3, masses, kT, coeff):
        """
        Compute the quantity Ncw[j] = sum_a L3[a,a,j] * masses[a] * (f1(masses[a], mu) - kT + 0.5)
        and return coeff * Ncw.
        
        L3 is assumed to be a NumPy array of shape (n, n, 8), and masses is a list or array.
        """
        masses = np.array(masses)  # ensure it's a NumPy array
        n = len(masses)
        N = 8
        
        # Precompute f1 for each mass
        f1_vals = np.array([self.f1(m, self.mu) for m in masses])
        
        # Extract the diagonal elements L3[a,a,:] for each a (resulting in an array of shape (n, N))
        # One could also use np.einsum('aaj->aj', L3) if L3 is a full array.
        L3_diag = np.array([L3[a, a, :] for a in range(n)])
        # Compute contributions: for each a, shape (n, N)
        contributions = L3_diag * masses[:, None] * (f1_vals - kT + 0.5)[:, None]
        
        # Sum over a to get an array of shape (N,)
        Ncw = contributions.sum(axis=0)
        return coeff * Ncw

    #@jit
    def Hij(self, L3, L4, masses, kT, coeff):
        """
        Compute a symmetric matrix Mcw[i,j] from the L3 and L4 couplings.
        
        The first term is:
        sum_{a,b} L3[a,b,i] * L3[b,a,j] * (f2(masses[a], masses[b], mu) - kT + 0.5)
        The second term (only if masses[a] != 0) is:
        sum_{a} L4[a,a,i,j] * masses[a] * (f1(masses[a], mu) - kT + 0.5)
        
        L3 is assumed to have shape (n, n, 8) and L4 shape (n, n, 8, 8).
        """
        masses = np.array(masses)
        n = len(masses)
        #N = 8
        
        # -----------------------------
        # First term: use np.einsum for double sum over a and b.
        # Precompute the f2 matrix F2[a,b] = self.f2(masses[a], masses[b], mu)
        F2 = np.empty((n, n))
        for a in range(n):
            for b in range(n):
                F2[a, b] = self.f2(masses[a], masses[b], self.mu)
        factor = F2 - kT + 0.5
        
        # L3 has shape (n, n, N). We need L3[a,b,i] and L3[b,a,j].
        # Use swapaxes to get L3_swapped[b,a,j] = L3[a,b,j].
        L3_swapped = np.swapaxes(L3, 0, 1)
        
        # Compute the double sum over a and b:
        # term1[i,j] = sum_{a,b} L3[a,b,i] * L3[b,a,j] * factor[a,b]
        term1 = np.einsum('abi,abj,ab->ij', L3, L3_swapped, factor)
        
        # -----------------------------
        # Second term: contributions from L4
        # Precompute f1 for each mass, and only add if the mass is nonzero.
        # (You might want to decide what to do when mass==0; here we simply set f1 to 0.)
        f1_vals = np.array([self.f1(m, self.mu) if m != 0 else 0 for m in masses])
        
        # Extract the diagonal of L4: L4_diag[a,:,:] = L4[a,a,:,:] (shape: (n, N, N))
        L4_diag = np.array([L4[a, a, :, :] for a in range(n)])
        
        # term2[i,j] = sum_{a} L4[a,a,i,j] * masses[a] * (f1_vals[a] - kT + 0.5)
        term2 = np.einsum('aij,a->ij', L4_diag, masses * (f1_vals - kT + 0.5))
        
        # -----------------------------
        # Combine both contributions
        Mcw = term1 + term2
        
        
        return coeff * Mcw

    ####### "Raw" unoptimized helper functions #######

    # log term 2, IR divergent
    def f2_IRDIV(self, m1sq, m2sq, mu):
        
        sign1 = 1
        sign2 = 1
        if m1sq < 0:
            m1sq = -m1sq
            sign1 = -1
        if m2sq < 0:
            m2sq = -m2sq
            sign2 = -1
        
        # calc
        log1 = 0
        log2 = 0
        if m1sq == 0 and m2sq == 0:
            return 1
        if m1sq != 0:
            log1 = sign1*sp.log(m1sq/mu**2)
            log2 = 0
            
        if sp.Abs(m1sq-m2sq)>1e-5: # add threshold
            log2 = 0  
            if m2sq != 0:
                log2 = sign2*sp.log(m2sq/mu**2)
            if m1sq == 0:
                return log2
            if m2sq == 0:
                return log1
            else:
                return (m1sq*log1 - m2sq*log2)/(m1sq-m2sq) 
        else: 
            return 1 + log1
