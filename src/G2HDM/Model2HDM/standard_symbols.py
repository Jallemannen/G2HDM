
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
STANDARD_SYMBOLS["Phi1_matrix"] = sp.MatrixSymbol("\Phi_1",2,1)
STANDARD_SYMBOLS["Phi2_matrix"] = sp.MatrixSymbol("\Phi_2",2,1)