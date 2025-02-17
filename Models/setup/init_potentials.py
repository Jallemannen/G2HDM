# Imports
import sympy as sp 

# Custom imports
from src.utils.methods_math import dagger


#################### Potential functions ####################
    
# General Tree-level potential
def potential_V0(x1, x2, params, fields=None, bgfields=None):
    c11,c22,c1,c2,c3,c4,c12,c5,c6,c7 = params
    V_mass = c11 * dagger(x1)*(x1) + c22 * dagger(x2)*(x2) - (c12 *dagger(x1)*(x2) +(c12).conjugate() * dagger(x2)*(x1) )
    #V_lambda14 = c1/2 * dagger(x1).dot(x1)**2 + c2/2 * dagger(x2).dot(x2)**2 + c3 * dagger(x1).dot(x1)*dagger(x2).dot(x2) + c4 * dagger(x1).dot(x2)*dagger(x2).dot(x1)
    V_lambda14 = c1/2 * (dagger(x1)*(x1))*(dagger(x1)*(x1)) + c2/2 * (dagger(x2)*(x2))*(dagger(x2)*(x2)) + c3* (dagger(x1)*(x1))*(dagger(x2)*(x2)) + c4 * (dagger(x1)*(x2))*(dagger(x2)*(x1))
    V_lambda57 = c5/2 * (dagger(x1)*(x2))*(dagger(x1)*(x2)) + c6 * (dagger(x1)*(x1))*(dagger(x1)*(x2)) + c7 * (dagger(x2)*(x2))*(dagger(x1)*(x2))
    V_lambda57_hc = c5.conjugate()/2 * (dagger(x2)*(x1))*(dagger(x2)*(x1)) + c6.conjugate() * (dagger(x1)*(x1))*(dagger(x2)*(x1)) + c7.conjugate() * (dagger(x2)*(x2))*(dagger(x2)*(x1))
    potential = V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc
    
    if isinstance(x1, sp.MatrixSymbol):
        return potential
    else:
        return potential[0]

# Most general counterterm potential
def potential_VCT(x1, x2, params, fields, displayform=False):
    
    f1, f2, f3, f4, f5, f6, f7, f8 = fields[0:8]

    omega1, omega2, omegaCP, omegaCB = fields[8:12]
    
    c11,c22,c1,c2,c3,c4,c12,c5,c6,c7 = params[:10]
    dT1, dT2, dTCP, dTCB = params[10:14]
    dD11, dD12, dD13, dD14 = params[14:18]
    dD22, dD23, dD24 = params[18:21]
    dD33, dD34 = params[21:23]
    dD44 = params[23]
    dD55, dD56, dD57, dD58 = params[24:28]
    dD66, dD67, dD68 = params[28:31]
    dD77, dD78 = params[31:33]
    dD88 = params[33]
    #add off-diagonal terms? and two different params for w and wCB?
    
    # First order
    V_T = dT1 * (f1) + dT2 * (f2) + dTCP * (f4) + dTCB * (f6)
    
    # Second order
    V_mass = c11 * dagger(x1)*(x1) + c22 * dagger(x2)*(x2) - (c12 *dagger(x1)*(x2) + c12.conjugate() * dagger(x2)*(x1) )
    
    # Fourth order
    V_lambda14 = c1/2 * (dagger(x1)*(x1))*(dagger(x1)*(x1)) + c2/2 * (dagger(x2)*(x2))*(dagger(x2)*(x2)) + c3* (dagger(x1)*(x1))*(dagger(x2)*(x2)) + c4 * (dagger(x1)*(x2))*(dagger(x2)*(x1))
    V_lambda57 = c5/2 * (dagger(x1)*(x2))*(dagger(x1)*(x2)) + c6 * (dagger(x1)*(x1))*(dagger(x1)*(x2)) + c7 * (dagger(x2)*(x2))*(dagger(x1)*(x2))
    V_lambda57_hc = c5.conjugate()/2 * (dagger(x2)*(x1))*(dagger(x2)*(x1)) + c6.conjugate() * (dagger(x1)*(x1))*(dagger(x2)*(x1)) + c7.conjugate() * (dagger(x2)*(x2))*(dagger(x2)*(x1))
    
    # Third order
    V_D = 0
    
    # Return
    V1 = V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc
    V2 = V_T + V_D
    
    if displayform:
        return V1, V2
    else:
        return V1[0] + V2

# Counterterm potential
def potential_VCT_higgs(x1, x2, params, fields, bgfields):
    
    # Fields
    phi1, phi2, G0, a0, RGp, RHp, IGp, IHp = fields[0:8]
    omega, omegaCB = bgfields[0], bgfields[4]
    
    # Parameters
    d_m11, d_m22, d_lambda1, d_lambda2, d_lambda3, d_lambda4, d_m12, d_lambda5, d_lambda6, d_lambda7 = params[:10]
    dT1, dT2, dTCP, dTCB = params[10:14]
    d_D13, d_D33, d_D14, d_D24, d_D67, d_D78 = params[14:]
    
    #Second order
    #V_mass = d_m11 * dagger(x1).dot(x1)**2 + d_m22 * dagger(x2).dot(x2)**2 - (d_m12 *dagger(x1).dot(x2) + d_m12 * dagger(x2).dot(x1) )
    V_mass = d_m11 * dagger(x1)*(x1) + d_m22 * dagger(x2)*(x2) - (d_m12 *dagger(x1)*(x2) + d_m12.conjugate() * dagger(x2)*(x1) )
    #Forth order
    V_lambda14 = d_lambda1/2 * (dagger(x1)*x1)**2 + d_lambda2/2 * (dagger(x2)*x2)**2 + d_lambda3 * (dagger(x1)*x1)*(dagger(x2)*x2) + d_lambda4 * (dagger(x1)*x2) * (dagger(x2)*x1)
    V_lambda57 = d_lambda5/2 * (dagger(x1)*x2)**2 + d_lambda6 * (dagger(x1)*x1)*(dagger(x1)*x2) + d_lambda7 * (dagger(x2)*x2)*(dagger(x1)*x2)
    
    V_lambda57_hc = d_lambda5.conjugate()/2 * (dagger(x2)* x1)**2 + d_lambda6.conjugate() * (dagger(x1)*x1)*(dagger(x2)*x1) + d_lambda7.conjugate() * (dagger(x2)*x2)*(dagger(x2)*x1)  #
    #V_lambda57_hc = sp.conjugate(V_lambda57) #
    
    #First order
    #V1_CT = dT * (omega + phi1) + dTCB * (omegaCB + RHp)
    V1_CT = dT1 * (omega + phi1) + dT2 * (phi2) + dTCP * (a0) + dTCB * (omegaCB + RHp)
    #third order
    d_D58, d_D56 = d_D67, d_D78
    V3_CT_1 = d_D33*G0**2*(phi1+omega) + d_D13*(phi1+omega)**2*G0 + d_D14*(phi1+omega)**2*a0  + d_D24*(phi1+omega)*phi2*a0
    V3_CT_2 = (phi1+omega) * (d_D67 * (RHp + omegaCB)*IGp - d_D58*RGp*IHp + d_D56*RGp*(RHp+omegaCB)+d_D78*IGp*IHp)
    
    # Return
    V1 = V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc
    V2 = V1_CT + V3_CT_1 + V3_CT_2
    
    if isinstance(x1, sp.MatrixSymbol):
        return V1, V2
    else:
        return V1[0] + V2
