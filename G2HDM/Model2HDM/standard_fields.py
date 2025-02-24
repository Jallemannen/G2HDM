
import sympy as sp

fields_gen = [sp.Symbol("\phi_1", real="true"), 
                sp.Symbol("\phi_2", real="true"), 
                sp.Symbol("a_1", real="true"), 
                sp.Symbol("a_2", real="true"), 
                sp.Symbol("\mathcal{R}(\phi_1^+)", real="true"), 
                sp.Symbol("\mathcal{R}(\phi_2^+)", real="true"), 
                sp.Symbol("\mathcal{I}(\phi_1^+)", real="true"), 
                sp.Symbol("\mathcal{I}(\phi_2^+)", real="true")]

bgfields_gen = [sp.Symbol("\omega_1", real="true"),
                sp.Symbol("\omega_2", real="true"),
                0,
                sp.Symbol("\omega_{CP}", real="true"),
                0,
                sp.Symbol("\omega_{CB}", real="true"),
                0,
                0]

VEVs_gen = [sp.Symbol("v_1", real="true"), 
            sp.Symbol("v_2", real="true"),
            0,
            sp.Symbol("v_{CP}", real="true"),
            0,
            sp.Symbol("v_{CB}", real="true"),
            0,
            0]
            