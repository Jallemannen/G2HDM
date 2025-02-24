
import sympy as sp

STANDARD_SYMBOLS = {}

STANDARD_SYMBOLS["h1"] = sp.Symbol("h_1", real="true")
STANDARD_SYMBOLS["h2"] = sp.Symbol("h_2", real="true")
STANDARD_SYMBOLS["h3"] = sp.Symbol("h_3", real="true")
STANDARD_SYMBOLS["h4"] = sp.Symbol("h_4", real="true")
STANDARD_SYMBOLS["h5"] = sp.Symbol("h_5", real="true")
STANDARD_SYMBOLS["h6"] = sp.Symbol("h_6", real="true")
STANDARD_SYMBOLS["h7"] = sp.Symbol("h_7", real="true")
STANDARD_SYMBOLS["h8"] = sp.Symbol("h_8", real="true")

STANDARD_SYMBOLS["Phi1"] = sp.Symbol("\Phi_1")
STANDARD_SYMBOLS["Phi2"] = sp.Symbol("\Phi_2")
STANDARD_SYMBOLS["field1_matrixsymbol"] = sp.MatrixSymbol("\Phi_1",2,1)
STANDARD_SYMBOLS["field2_matrixsymbol"] = sp.MatrixSymbol("\Phi_2",2,1)

"""# Add L1 to L8 as Λ_i
for i in range(1, 9):
    STANDARD_SYMBOLS[f"L{i}"] = sp.Symbol(f"\\Lambda_{{{i}}}")

# Add Lij (Λ_ij) 
for i in range(1, 9):
    for j in range(1, 9):
        STANDARD_SYMBOLS[f"L{i}{j}"] = sp.Symbol(f"\\Lambda_{{{i}{j}}}")

# Add Lijk (Λ_ijk)
for i in range(1, 9):
    for j in range(1, 9):
        for k in range(1, 9):
            STANDARD_SYMBOLS[f"L{i}{j}{k}"] = sp.Symbol(f"\\Lambda_{{{i}{j}{k}}}")

# Add Lijkl (Λ_ijkl)
for i in range(1, 9):
    for j in range(1, 9):
        for k in range(1, 9):
            for l in range(1, 9):
                STANDARD_SYMBOLS[f"L{i}{j}{k}{l}"] = sp.Symbol(f"\\Lambda_{{{i}{j}{k}{l}}}")
"""

STANDARD_SYMBOLS[f"mu11"] = sp.Symbol("\mu_{11}", real="true")
STANDARD_SYMBOLS[f"mu22"] = sp.Symbol("\mu_{22}", real="true")
STANDARD_SYMBOLS[f"L1"] = sp.Symbol("\Lambda_{1}", real="true")
STANDARD_SYMBOLS[f"L2"] = sp.Symbol("\Lambda_{2}", real="true")
STANDARD_SYMBOLS[f"L3"] = sp.Symbol("\Lambda_{3}", real="true")
STANDARD_SYMBOLS[f"L4"] = sp.Symbol("\Lambda_{4}", real="true")
STANDARD_SYMBOLS[f"mu12"] = sp.Symbol("\mu_{12}")
STANDARD_SYMBOLS[f"L5"] = sp.Symbol("\Lambda_{5}")
STANDARD_SYMBOLS[f"L6"] = sp.Symbol("\Lambda_{6}")
STANDARD_SYMBOLS[f"L7"] = sp.Symbol("\Lambda_{7}")

STANDARD_SYMBOLS[f"dmu11"] = sp.Symbol("\delta\mu_{11}", real="true")
STANDARD_SYMBOLS[f"dmu22"] = sp.Symbol("\delta\mu_{22}", real="true")
STANDARD_SYMBOLS[f"dL1"] = sp.Symbol("\delta\Lambda_{1}", real="true")
STANDARD_SYMBOLS[f"dL2"] = sp.Symbol("\delta\Lambda_{2}", real="true")
STANDARD_SYMBOLS[f"dL3"] = sp.Symbol("\delta\Lambda_{3}", real="true")
STANDARD_SYMBOLS[f"dL4"] = sp.Symbol("\delta\Lambda_{4}", real="true")
STANDARD_SYMBOLS[f"dmu12"] = sp.Symbol("\delta\mu_{12}")
STANDARD_SYMBOLS[f"dL5"] = sp.Symbol("\delta\Lambda_{5}")
STANDARD_SYMBOLS[f"dL6"] = sp.Symbol("\delta\Lambda_{6}")
STANDARD_SYMBOLS[f"dL7"] = sp.Symbol("\delta\Lambda_{7}")
STANDARD_SYMBOLS[f"dT1"] = sp.Symbol("\delta T_1", real="true")
STANDARD_SYMBOLS[f"dT2"] = sp.Symbol("\delta T_2", real="true")
STANDARD_SYMBOLS[f"dTCP"] = sp.Symbol("\delta T_{CP}", real="true")
STANDARD_SYMBOLS[f"dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
STANDARD_SYMBOLS[f"dD13"] = sp.Symbol("\delta\Delta_{13}", real="true")
STANDARD_SYMBOLS[f"dD33"] = sp.Symbol("\delta\Delta_{33}", real="true")
STANDARD_SYMBOLS[f"dD34"] = sp.Symbol("\delta\Delta_{34}", real="true")
STANDARD_SYMBOLS[f"dD14"] = sp.Symbol("\delta\Delta_{14}", real="true")
STANDARD_SYMBOLS[f"dD24"] = sp.Symbol("\delta\Delta_{24}", real="true")
STANDARD_SYMBOLS[f"dD67"] = sp.Symbol("\delta\Delta_{67}", real="true")
STANDARD_SYMBOLS[f"dD78"] = sp.Symbol("\delta\Delta_{78}", real="true")



STANDARD_SYMBOLS["g1"] = sp.Symbol("g_1", real="true")
STANDARD_SYMBOLS["g2"] = sp.Symbol("g_2", real="true")