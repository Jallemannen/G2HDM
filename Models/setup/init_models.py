

# Imports
import sympy as sp

##### SYMBOLS HIGGS #####
def symbols_higgs(using_omegaCB=False) -> dict:

    SYMBOLS = {}

    # Higgs basis symbols
    SYMBOLS["H1"] = sp.Symbol("\mathcal{H}_1")
    SYMBOLS["H2"] = sp.Symbol("\mathcal{H}_2")
    SYMBOLS["H1_matrix"] = sp.MatrixSymbol("\mathcal{H}_1",2,1)
    SYMBOLS["H2_matrix"] = sp.MatrixSymbol("\mathcal{H}_2",2,1)

    SYMBOLS["Gp"] = sp.Symbol("G^+")
    SYMBOLS["Hp"] = sp.Symbol("H^+")

    SYMBOLS["RGp"] = sp.Symbol("\mathcal{R}(G^+)", real="true")
    SYMBOLS["RHp"] = sp.Symbol("\mathcal{R}(H^+)", real="true")
    SYMBOLS["IGp"] = sp.Symbol("\mathcal{I}(G^+)", real="true")
    SYMBOLS["IHp"] = sp.Symbol("\mathcal{I}(H^+)", real="true")

    SYMBOLS["phi1"] = sp.Symbol("\phi_1", real="true")
    SYMBOLS["phi2"] = sp.Symbol("\phi_2", real="true")
    SYMBOLS["a0"] = sp.Symbol("a^0", real="true")
    SYMBOLS["G0"] = sp.Symbol("G^0", real="true")
    
    # Mass field states symbols 1-8
    SYMBOLS["h1"] = sp.Symbol("h_1", real="true")
    SYMBOLS["h2"] = sp.Symbol("h_2", real="true")
    SYMBOLS["h3"] = sp.Symbol("h_3", real="true")
    SYMBOLS["h4"] = sp.Symbol("h_4", real="true")
    SYMBOLS["h5"] = sp.Symbol("h_5", real="true")
    SYMBOLS["h6"] = sp.Symbol("h_6", real="true")
    SYMBOLS["h7"] = sp.Symbol("h_7", real="true")
    SYMBOLS["h8"] = sp.Symbol("h_8", real="true")

    # VEV Symbols
    SYMBOLS["v"] = sp.Symbol("v", real="true", positive=True)
    SYMBOLS["v1"] = sp.Symbol("v_1", real="true", positive=True)
    SYMBOLS["v2"] = sp.Symbol("v_2", real="true", positive=True)
    SYMBOLS["vCP"] = sp.Symbol("v_{CP}", real="true", positive=True)
    SYMBOLS["vCB"] = sp.Symbol("v_{CB}", real="true", positive=True)
    SYMBOLS["omega"] = sp.Symbol("\omega", real="true", positive=True)
    SYMBOLS["omega1"] = sp.Symbol("\omega_1", real="true", positive=True)
    SYMBOLS["omega2"] = sp.Symbol("\omega_2", real="true", positive=True)
    SYMBOLS["omegaCP"] = sp.Symbol("\omega_{CP}", real="true", positive=True)
    SYMBOLS["omegaCB"] = sp.Symbol("\omega_{CB}", real="true", positive=True)

    ##### Parameter symbols #####

    # Higgs basis V0 parameters
    SYMBOLS["Y1"] = sp.Symbol("Y_1", real="true")
    SYMBOLS["Y2"] = sp.Symbol("Y_2", real="true")
    SYMBOLS["Z1"] = sp.Symbol("Z_1", real="true")
    SYMBOLS["Z2"] = sp.Symbol("Z_2", real="true")
    SYMBOLS["Z3"] = sp.Symbol("Z_3", real="true")
    SYMBOLS["Z4"] = sp.Symbol("Z_4", real="true")

    SYMBOLS["Y12"] = sp.Symbol("Y_{12}")
    SYMBOLS["Z5"] = sp.Symbol("Z_5")
    SYMBOLS["Z6"] = sp.Symbol("Z_6")
    SYMBOLS["Z7"] = sp.Symbol("Z_7")
    
    ##### Counterterm parameter symbols #####

    # Generic basis parameters
    SYMBOLS["dmu1"] = sp.Symbol("\delta\mu_1", real="true")
    SYMBOLS["dmu2"] = sp.Symbol("\delta\mu_2", real="true")
    SYMBOLS["dlambda1"] = sp.Symbol("\delta\lambda_1", real="true")
    SYMBOLS["dlambda2"] = sp.Symbol("\delta\lambda_2", real="true")
    SYMBOLS["dlambda3"] = sp.Symbol("\delta\lambda_3", real="true")
    SYMBOLS["dlambda4"] = sp.Symbol("\delta\lambda_4", real="true")

    SYMBOLS["dmu12"] = sp.Symbol("\delta\mu_{12}")
    SYMBOLS["dlambda5"] = sp.Symbol("\delta\lambda_5")
    SYMBOLS["dlambda6"] = sp.Symbol("\delta\lambda_6")
    SYMBOLS["dlambda7"] = sp.Symbol("\delta\lambda_7")
    
    SYMBOLS["dT1"] = sp.Symbol("\delta T_1", real="true")
    SYMBOLS["dT2"] = sp.Symbol("\delta T_2", real="true")
    SYMBOLS["dTCP"] = sp.Symbol("\delta T_{CP}", real="true")
    SYMBOLS["dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
    
    SYMBOLS["dDelta58"] = sp.Symbol("\delta\Delta_{58}", real="true")
    SYMBOLS["dDelta56"] = sp.Symbol("\delta\Delta_{56}", real="true")

    # Higgs basis VCT parameters
    SYMBOLS["dY1"] = sp.Symbol("\delta Y_1", real="true")
    SYMBOLS["dY2"] = sp.Symbol("\delta Y_2", real="true")
    SYMBOLS["dZ1"] = sp.Symbol("\delta Z_1", real="true")
    SYMBOLS["dZ2"] = sp.Symbol("\delta Z_2", real="true")
    SYMBOLS["dZ3"] = sp.Symbol("\delta Z_3", real="true")
    SYMBOLS["dZ4"] = sp.Symbol("\delta Z_4", real="true")
    
    SYMBOLS["dY12"] = sp.Symbol("\delta Y_{12}")
    SYMBOLS["dZ5"] = sp.Symbol("\delta Z_5")
    SYMBOLS["dZ6"] = sp.Symbol("\delta Z_6")
    SYMBOLS["dZ7"] = sp.Symbol("\delta Z_7")
    
    SYMBOLS["dT"] = sp.Symbol("\delta T_1", real="true")
    SYMBOLS["dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
    
    #d_D33, d_D14, d_D24, d_D67, d_D78
    SYMBOLS["dD33"] = sp.Symbol("\delta\Delta_{33}", real="true")
    SYMBOLS["dD13"] = sp.Symbol("\delta\Delta_{13}", real="true")
    SYMBOLS["dD14"] = sp.Symbol("\delta\Delta_{14}", real="true")
    SYMBOLS["dD24"] = sp.Symbol("\delta\Delta_{24}", real="true")
    SYMBOLS["dD67"] = sp.Symbol("\delta\Delta_{67}", real="true")
    SYMBOLS["dD78"] = sp.Symbol("\delta\Delta_{78}", real="true")
    SYMBOLS["dD58"] = sp.Symbol("\delta\Delta_{58}", real="true")
    SYMBOLS["dD56"] = sp.Symbol("\delta\Delta_{56}", real="true")
    
    ##### Parameters #####
    #(need to keep the key-name the same here)
    
    kwargs = {}
    
    # Higgs basis V0 parameters
    kwargs["V0_params"] = [SYMBOLS["Y1"], SYMBOLS["Y2"], SYMBOLS["Z1"], SYMBOLS["Z2"], SYMBOLS["Z3"], SYMBOLS["Z4"],
                    SYMBOLS["Y12"], SYMBOLS["Z5"], SYMBOLS["Z6"], SYMBOLS["Z7"]]
    
    kwargs["VCT_params"] = [SYMBOLS["dY1"], SYMBOLS["dY2"], SYMBOLS["dZ1"], SYMBOLS["dZ2"], SYMBOLS["dZ3"], SYMBOLS["dZ4"], 
                    SYMBOLS["dY12"], SYMBOLS["dZ5"], SYMBOLS["dZ6"], SYMBOLS["dZ7"], 
                    SYMBOLS["dT1"], SYMBOLS["dT2"], SYMBOLS["dTCP"], SYMBOLS["dTCB"],
                    SYMBOLS["dD13"], SYMBOLS["dD33"], SYMBOLS["dD14"], SYMBOLS["dD24"], SYMBOLS["dD67"], SYMBOLS["dD78"]]
    
    # Field sets
    w1, w2, w3, w4, w5, w6, w7, w8 = [SYMBOLS["omega"], 0, 0, 0, SYMBOLS["omegaCB"], 0, 0 ,0]
    v1, v2, v3, v4, v5, v6, v7, v8 = [SYMBOLS["v"], 0, 0, 0, SYMBOLS["vCB"], 0, 0, 0]
    f1, f2, f3, f4, f5, f6, f7, f8 = [SYMBOLS["phi1"], SYMBOLS["phi2"], SYMBOLS["G0"], SYMBOLS["a0"], SYMBOLS["RGp"], SYMBOLS["RHp"], SYMBOLS["IGp"], SYMBOLS["IHp"]]
    
    if not using_omegaCB:
        w5 = 0
        v5 = 0
    
    kwargs["bgfields"] = [w1, w2, w3, w4, w5, w6, w7, w8]
    kwargs["VEVs"] = [v1, v2, v3, v4, v5, v6, v7, v8]
    kwargs["fields"] = [f1, f2, f3, f4, f5, f6, f7, f8]
                            
    kwargs["massfields"] = [SYMBOLS["h1"], SYMBOLS["h2"], SYMBOLS["h3"], SYMBOLS["h4"], SYMBOLS["h5"], SYMBOLS["h6"], SYMBOLS["h7"], SYMBOLS["h8"]]

    # Basis fields
    kwargs["bgfield1"] = 1/sp.sqrt(2) * sp.Matrix([w5+sp.I*w7, w1+sp.I*w3])
    kwargs["bgfield2"] = 1/sp.sqrt(2) * sp.Matrix([w6+sp.I*w8, w2+sp.I*w4])

    kwargs["field1"] = sp.simplify(sp.Matrix([f5+sp.I*f7, f1+sp.I*f3])/sp.sqrt(2) + kwargs["bgfield1"])
    kwargs["field2"] = sp.simplify(sp.Matrix([f6+sp.I*f8, f2+sp.I*f4])/sp.sqrt(2) + kwargs["bgfield2"])
    
    kwargs["field1_matrixsymbol"] = SYMBOLS["H1_matrix"]
    kwargs["field2_matrixsymbol"] = SYMBOLS["H2_matrix"]
    
    return SYMBOLS, kwargs

##### SYMBOLS HIGGS CB #####
def symbols_higgsCB() -> dict:

    SYMBOLS = {}

    # Higgs basis symbols
    SYMBOLS["H1"] = sp.Symbol("\mathcal{H}_1")
    SYMBOLS["H2"] = sp.Symbol("\mathcal{H}_2")
    SYMBOLS["H1_matrix"] = sp.MatrixSymbol("\mathcal{H}_1",2,1)
    SYMBOLS["H2_matrix"] = sp.MatrixSymbol("\mathcal{H}_2",2,1)

    SYMBOLS["Gp"] = sp.Symbol("G^+")
    SYMBOLS["Hp"] = sp.Symbol("H^+")

    SYMBOLS["RGp"] = sp.Symbol("\mathcal{R}(G^+)", real="true")
    SYMBOLS["RHp"] = sp.Symbol("\mathcal{R}(H^+)", real="true")
    SYMBOLS["IGp"] = sp.Symbol("\mathcal{I}(G^+)", real="true")
    SYMBOLS["IHp"] = sp.Symbol("\mathcal{I}(H^+)", real="true")

    SYMBOLS["phi1"] = sp.Symbol("\phi_1", real="true")
    SYMBOLS["phi2"] = sp.Symbol("\phi_2", real="true")
    SYMBOLS["a0"] = sp.Symbol("a^0", real="true")
    SYMBOLS["G0"] = sp.Symbol("G^0", real="true")
    
    # Mass field states symbols 1-8
    SYMBOLS["h1"] = sp.Symbol("h_1", real="true")
    SYMBOLS["h2"] = sp.Symbol("h_2", real="true")
    SYMBOLS["h3"] = sp.Symbol("h_3", real="true")
    SYMBOLS["h4"] = sp.Symbol("h_4", real="true")
    SYMBOLS["h5"] = sp.Symbol("h_5", real="true")
    SYMBOLS["h6"] = sp.Symbol("h_6", real="true")
    SYMBOLS["h7"] = sp.Symbol("h_7", real="true")
    SYMBOLS["h8"] = sp.Symbol("h_8", real="true")

    # VEV Symbols
    SYMBOLS["v"] = sp.Symbol("v", real="true", positive=True)
    SYMBOLS["v1"] = sp.Symbol("v_1", real="true", positive=True)
    SYMBOLS["v2"] = sp.Symbol("v_2", real="true", positive=True)
    SYMBOLS["vCP"] = sp.Symbol("v_{CP}", real="true", positive=True)
    SYMBOLS["vCB"] = sp.Symbol("v_{CB}", real="true", positive=True)
    SYMBOLS["omega"] = sp.Symbol("\omega", real="true", positive=True)
    SYMBOLS["omega1"] = sp.Symbol("\omega_1", real="true", positive=True)
    SYMBOLS["omega2"] = sp.Symbol("\omega_2", real="true", positive=True)
    SYMBOLS["omegaCP"] = sp.Symbol("\omega_{CP}", real="true", positive=True)
    SYMBOLS["omegaCB"] = sp.Symbol("\omega_{CB}", real="true", positive=True)

    ##### Parameter symbols #####

    # Higgs basis V0 parameters
    SYMBOLS["Y1"] = sp.Symbol("Y_1", real="true")
    SYMBOLS["Y2"] = sp.Symbol("Y_2", real="true")
    SYMBOLS["Z1"] = sp.Symbol("Z_1", real="true")
    SYMBOLS["Z2"] = sp.Symbol("Z_2", real="true")
    SYMBOLS["Z3"] = sp.Symbol("Z_3", real="true")
    SYMBOLS["Z4"] = sp.Symbol("Z_4", real="true")

    SYMBOLS["Y12"] = sp.Symbol("Y_{12}")
    SYMBOLS["Z5"] = sp.Symbol("Z_5")
    SYMBOLS["Z6"] = sp.Symbol("Z_6")
    SYMBOLS["Z7"] = sp.Symbol("Z_7")
    
    ##### Counterterm parameter symbols #####

    # Generic basis parameters
    SYMBOLS["dmu1"] = sp.Symbol("\delta\mu_1", real="true")
    SYMBOLS["dmu2"] = sp.Symbol("\delta\mu_2", real="true")
    SYMBOLS["dlambda1"] = sp.Symbol("\delta\lambda_1", real="true")
    SYMBOLS["dlambda2"] = sp.Symbol("\delta\lambda_2", real="true")
    SYMBOLS["dlambda3"] = sp.Symbol("\delta\lambda_3", real="true")
    SYMBOLS["dlambda4"] = sp.Symbol("\delta\lambda_4", real="true")

    SYMBOLS["dmu12"] = sp.Symbol("\delta\mu_{12}")
    SYMBOLS["dlambda5"] = sp.Symbol("\delta\lambda_5")
    SYMBOLS["dlambda6"] = sp.Symbol("\delta\lambda_6")
    SYMBOLS["dlambda7"] = sp.Symbol("\delta\lambda_7")
    
    SYMBOLS["dT1"] = sp.Symbol("\delta T_1", real="true")
    SYMBOLS["dT2"] = sp.Symbol("\delta T_2", real="true")
    SYMBOLS["dTCP"] = sp.Symbol("\delta T_{CP}", real="true")
    SYMBOLS["dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
    
    SYMBOLS["dDelta58"] = sp.Symbol("\delta\Delta_{58}", real="true")
    SYMBOLS["dDelta56"] = sp.Symbol("\delta\Delta_{56}", real="true")

    # Higgs basis VCT parameters
    SYMBOLS["dY1"] = sp.Symbol("\delta Y_1", real="true")
    SYMBOLS["dY2"] = sp.Symbol("\delta Y_2", real="true")
    SYMBOLS["dZ1"] = sp.Symbol("\delta Z_1", real="true")
    SYMBOLS["dZ2"] = sp.Symbol("\delta Z_2", real="true")
    SYMBOLS["dZ3"] = sp.Symbol("\delta Z_3", real="true")
    SYMBOLS["dZ4"] = sp.Symbol("\delta Z_4", real="true")
    
    SYMBOLS["dY12"] = sp.Symbol("\delta Y_{12}")
    SYMBOLS["dZ5"] = sp.Symbol("\delta Z_5")
    SYMBOLS["dZ6"] = sp.Symbol("\delta Z_6")
    SYMBOLS["dZ7"] = sp.Symbol("\delta Z_7")
    
    SYMBOLS["dT"] = sp.Symbol("\delta T_1", real="true")
    SYMBOLS["dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
    
    #d_D33, d_D14, d_D24, d_D67, d_D78
    SYMBOLS["dD33"] = sp.Symbol("\delta\Delta_{33}", real="true")
    SYMBOLS["dD13"] = sp.Symbol("\delta\Delta_{13}", real="true")
    SYMBOLS["dD14"] = sp.Symbol("\delta\Delta_{14}", real="true")
    SYMBOLS["dD24"] = sp.Symbol("\delta\Delta_{24}", real="true")
    SYMBOLS["dD67"] = sp.Symbol("\delta\Delta_{67}", real="true")
    SYMBOLS["dD78"] = sp.Symbol("\delta\Delta_{78}", real="true")
    SYMBOLS["dD58"] = sp.Symbol("\delta\Delta_{58}", real="true")
    SYMBOLS["dD56"] = sp.Symbol("\delta\Delta_{56}", real="true")
    
    ##### Parameters #####
    #(need to keep the key-name the same here)
    
    kwargs = {}
    
    # Higgs basis V0 parameters
    kwargs["V0_params"] = [SYMBOLS["Y1"], SYMBOLS["Y2"], SYMBOLS["Z1"], SYMBOLS["Z2"], SYMBOLS["Z3"], SYMBOLS["Z4"],
                    SYMBOLS["Y12"], SYMBOLS["Z5"], SYMBOLS["Z6"], SYMBOLS["Z7"]]
    
    kwargs["VCT_params"] = [SYMBOLS["dY1"], SYMBOLS["dY2"], SYMBOLS["dZ1"], SYMBOLS["dZ2"], SYMBOLS["dZ3"], SYMBOLS["dZ4"], 
                    SYMBOLS["dY12"], SYMBOLS["dZ5"], SYMBOLS["dZ6"], SYMBOLS["dZ7"], 
                    SYMBOLS["dT1"], SYMBOLS["dT2"], SYMBOLS["dTCP"], SYMBOLS["dTCB"],
                    SYMBOLS["dD13"], SYMBOLS["dD33"], SYMBOLS["dD14"], SYMBOLS["dD24"], SYMBOLS["dD67"], SYMBOLS["dD78"]]
    
    # Field sets
    w1, w2, w3, w4, w5, w6, w7, w8 = [SYMBOLS["omega"], 0, 0, 0, 0, SYMBOLS["omegaCB"], 0 ,0]
    kwargs["bgfields"] = [w1, w2, w3, w4, w5, w6, w7, w8]
    
    v1, v2, v3, v4, v5, v6, v7, v8 = [SYMBOLS["v"], 0, 0, 0, 0, SYMBOLS["vCB"], 0, 0]
    kwargs["VEVs"] = [v1, v2, v3, v4, v5, v6, v7, v8]
    
    f1, f2, f3, f4, f5, f6, f7, f8 = [SYMBOLS["phi1"], SYMBOLS["phi2"], SYMBOLS["G0"], SYMBOLS["a0"], SYMBOLS["RGp"], SYMBOLS["RHp"], SYMBOLS["IGp"], SYMBOLS["IHp"]]
    kwargs["fields"] = [f1, f2, f3, f4, f5, f6, f7, f8]
                            
    kwargs["massfields"] = [SYMBOLS["h1"], SYMBOLS["h2"], SYMBOLS["h3"], SYMBOLS["h4"], SYMBOLS["h5"], SYMBOLS["h6"], SYMBOLS["h7"], SYMBOLS["h8"]]

    # Basis fields
    kwargs["bgfield1"] = 1/sp.sqrt(2) * sp.Matrix([w5+sp.I*w7, w1+sp.I*w3])
    kwargs["bgfield2"] = 1/sp.sqrt(2) * sp.Matrix([w6+sp.I*w8, w2+sp.I*w4])

    kwargs["field1"] = sp.simplify(sp.Matrix([f5+sp.I*f7, f1+sp.I*f3])/sp.sqrt(2) + kwargs["bgfield1"])
    kwargs["field2"] = sp.simplify(sp.Matrix([f6+sp.I*f8, f2+sp.I*f4])/sp.sqrt(2) + kwargs["bgfield2"])
    
    kwargs["field1_matrixsymbol"] = SYMBOLS["H1_matrix"]
    kwargs["field2_matrixsymbol"] = SYMBOLS["H2_matrix"]
    
    return SYMBOLS, kwargs

##### SYMBOLS GENERIC CB/CP #####
def symbols_gen(using_omegaCB = False, using_omegaCP = True) -> dict:
    
    SYMBOLS = {}

    # Genertic bais symbols
    SYMBOLS["Phi1"] = sp.Symbol("\Phi_1")
    SYMBOLS["Phi2"] = sp.Symbol("\Phi_2")
    SYMBOLS["Phi1_matrix"] = sp.MatrixSymbol("\Phi_1",2,1)
    SYMBOLS["Phi2_matrix"] = sp.MatrixSymbol("\Phi_2",2,1)

    SYMBOLS["phi1p"] = sp.Symbol("\phi_1^+")
    SYMBOLS["phi2p"] = sp.Symbol("\phi_2^+")

    SYMBOLS["Rphi1p"] = sp.Symbol("\mathcal{R}(\phi^+)", real="true") 
    SYMBOLS["Rphi2p"] = sp.Symbol("\mathcal{R}(\phi^+)", real="true") 
    SYMBOLS["Iphi1p"] = sp.Symbol("\mathcal{I}(\phi^+)", real="true")
    SYMBOLS["Iphi2p"] = sp.Symbol("\mathcal{I}(\phi^+)", real="true")

    SYMBOLS["varphi1"] = sp.Symbol(r"\varphi_1", real="true")
    SYMBOLS["varphi2"] = sp.Symbol(r"\varphi_2", real="true")
    SYMBOLS["a1"] = sp.Symbol("a_1", real="true")
    SYMBOLS["a2"] = sp.Symbol("a_2", real="true")
    
    # Mass field states symbols 1-8
    SYMBOLS["h1"] = sp.Symbol("h_1", real="true")
    SYMBOLS["h2"] = sp.Symbol("h_2", real="true")
    SYMBOLS["h3"] = sp.Symbol("h_3", real="true")
    SYMBOLS["h4"] = sp.Symbol("h_4", real="true")
    SYMBOLS["h5"] = sp.Symbol("h_5", real="true")
    SYMBOLS["h6"] = sp.Symbol("h_6", real="true")
    SYMBOLS["h7"] = sp.Symbol("h_7", real="true")
    SYMBOLS["h8"] = sp.Symbol("h_8", real="true")
    
    # VEV Symbols
    SYMBOLS["v"] = sp.Symbol("v", real="true", positive=True)
    SYMBOLS["v1"] = sp.Symbol("v_1", real="true", positive=True)
    SYMBOLS["v2"] = sp.Symbol("v_2", real="true", positive=True)
    SYMBOLS["vCP"] = sp.Symbol("v_{CP}", real="true", positive=True)
    SYMBOLS["vCB"] = sp.Symbol("v_{CB}", real="true", positive=True)
    SYMBOLS["omega"] = sp.Symbol("\omega", real="true", positive=True)
    SYMBOLS["omega1"] = sp.Symbol("\omega_1", real="true", positive=True)
    SYMBOLS["omega2"] = sp.Symbol("\omega_2", real="true", positive=True)
    SYMBOLS["omegaCP"] = sp.Symbol("\omega_{CP}", real="true", positive=True)
    SYMBOLS["omegaCB"] = sp.Symbol("\omega_{CB}", real="true", positive=True)
    
    ##### Parameter symbols #####

    # Generic basis parameters
    SYMBOLS["mu1"] = sp.Symbol("\mu_1", real="true")
    SYMBOLS["mu2"] = sp.Symbol("\mu_2", real="true")
    SYMBOLS["lambda1"] = sp.Symbol("\lambda_1", real="true")
    SYMBOLS["lambda2"] = sp.Symbol("\lambda_2", real="true")
    SYMBOLS["lambda3"] = sp.Symbol("\lambda_3", real="true")
    SYMBOLS["lambda4"] = sp.Symbol("\lambda_4", real="true")

    SYMBOLS["mu12"] = sp.Symbol("\mu_{12}")
    SYMBOLS["lambda5"] = sp.Symbol("\lambda_5")
    SYMBOLS["lambda6"] = sp.Symbol("\lambda_6")
    SYMBOLS["lambda7"] = sp.Symbol("\lambda_7")
    
    # Generic basis parameters
    SYMBOLS["dmu1"] = sp.Symbol("\delta\mu_1", real="true")
    SYMBOLS["dmu2"] = sp.Symbol("\delta\mu_2", real="true")
    SYMBOLS["dlambda1"] = sp.Symbol("\delta\lambda_1", real="true")
    SYMBOLS["dlambda2"] = sp.Symbol("\delta\lambda_2", real="true")
    SYMBOLS["dlambda3"] = sp.Symbol("\delta\lambda_3", real="true")
    SYMBOLS["dlambda4"] = sp.Symbol("\delta\lambda_4", real="true")

    SYMBOLS["dmu12"] = sp.Symbol("\delta\mu_{12}")
    SYMBOLS["dlambda5"] = sp.Symbol("\delta\lambda_5")
    SYMBOLS["dlambda6"] = sp.Symbol("\delta\lambda_6")
    SYMBOLS["dlambda7"] = sp.Symbol("\delta\lambda_7")
    
    SYMBOLS["dT1"] = sp.Symbol("\delta T_1", real="true")
    SYMBOLS["dT2"] = sp.Symbol("\delta T_2", real="true")
    SYMBOLS["dTCP"] = sp.Symbol("\delta T_{CP}", real="true")
    SYMBOLS["dTCB"] = sp.Symbol("\delta T_{CB}", real="true")
    
    SYMBOLS["dD33"] = sp.Symbol("\delta\Delta_{33}", real="true")
    SYMBOLS["dD13"] = sp.Symbol("\delta\Delta_{13}", real="true")
    SYMBOLS["dD14"] = sp.Symbol("\delta\Delta_{14}", real="true")
    SYMBOLS["dD24"] = sp.Symbol("\delta\Delta_{24}", real="true")
    SYMBOLS["dD67"] = sp.Symbol("\delta\Delta_{67}", real="true")
    SYMBOLS["dD78"] = sp.Symbol("\delta\Delta_{78}", real="true")
    SYMBOLS["dD58"] = sp.Symbol("\delta\Delta_{58}", real="true")
    SYMBOLS["dD56"] = sp.Symbol("\delta\Delta_{56}", real="true")
        
    ##### Parameters #####
    #(need to keep the key-name the same here)
    
    kwargs = {}
    
    # Higgs basis V0 parameters
    kwargs["V0_params"] = [SYMBOLS["mu1"], SYMBOLS["mu2"], SYMBOLS["lambda1"], SYMBOLS["lambda2"], SYMBOLS["lambda3"], SYMBOLS["lambda4"],
                    SYMBOLS["mu12"], SYMBOLS["lambda5"], SYMBOLS["lambda6"], SYMBOLS["lambda7"]]
    
    kwargs["VCT_params"] = [SYMBOLS["dmu1"], SYMBOLS["dmu2"], SYMBOLS["dlambda1"], SYMBOLS["dlambda2"], SYMBOLS["dlambda3"], SYMBOLS["dlambda4"],
                    SYMBOLS["dmu12"], SYMBOLS["dlambda5"], SYMBOLS["dlambda6"], SYMBOLS["dlambda7"],
                    SYMBOLS["dT1"], SYMBOLS["dT2"], SYMBOLS["dTCP"], SYMBOLS["dTCB"],
                    SYMBOLS["dD13"], SYMBOLS["dD33"], SYMBOLS["dD14"], SYMBOLS["dD24"], SYMBOLS["dD67"], SYMBOLS["dD78"]]
    
    # Field sets
    w1, w2, w3, w4, w5, w6, w7, w8 = [SYMBOLS["omega1"], SYMBOLS["omega2"], 0, SYMBOLS["omegaCP"], 0, SYMBOLS["omegaCB"], 0 ,0]
    v1, v2, v3, v4, v5, v6, v7, v8 = [SYMBOLS["v1"], SYMBOLS["v2"], 0, SYMBOLS["vCP"], 0, SYMBOLS["vCB"], 0, 0]
    f1, f2, f3, f4, f5, f6, f7, f8 = [SYMBOLS["varphi1"], SYMBOLS["varphi2"], SYMBOLS["a1"], SYMBOLS["a2"],
                                    SYMBOLS["Rphi1p"], SYMBOLS["Rphi2p"], SYMBOLS["Iphi1p"], SYMBOLS["Iphi2p"]]
             
    if not using_omegaCB:
        w6 = 0
        v6 = 0
    
    if not using_omegaCP:
        w4 = 0
        v4 = 0
        
    kwargs["bgfields"] = [w1, w2, w3, w4, w5, w6, w7, w8]
    kwargs["VEVs"] = [v1, v2, v3, v4, v5, v6, v7, v8]
    kwargs["fields"] = [f1, f2, f3, f4, f5, f6, f7, f8]
                            
    kwargs["massfields"] = [SYMBOLS["h1"], SYMBOLS["h2"], SYMBOLS["h3"], SYMBOLS["h4"], SYMBOLS["h5"], SYMBOLS["h6"], SYMBOLS["h7"], SYMBOLS["h8"]]

    # Basis fields
    kwargs["bgfield1"] = 1/sp.sqrt(2) * sp.Matrix([w5+sp.I*w7, w1+sp.I*w3])
    kwargs["bgfield2"] = 1/sp.sqrt(2) * sp.Matrix([w6+sp.I*w8, w2+sp.I*w4])

    kwargs["field1"] = sp.simplify(sp.Matrix([f5+sp.I*f7, f1+sp.I*f3])/sp.sqrt(2) + kwargs["bgfield1"])
    kwargs["field2"] = sp.simplify(sp.Matrix([f6+sp.I*f8, f2+sp.I*f4])/sp.sqrt(2) + kwargs["bgfield2"])
    
    kwargs["field1_matrixsymbol"] = SYMBOLS["Phi1_matrix"]
    kwargs["field2_matrixsymbol"] = SYMBOLS["Phi2_matrix"]
    
    return SYMBOLS, kwargs