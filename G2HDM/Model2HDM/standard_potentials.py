
from ..utils.methods_math import dagger

# Always write out expressions explicitly (i.e no parentheses)


def potential_V0(x1,x2,params):
    m11, m22, L1, L2, L3, L4, L12, L5, L6, L7 = params[:10]
    V_mass = m11 * dagger(x1)*(x1) + m22 * dagger(x2)*(x2) - L12 *dagger(x1)*(x2) - L12.conjugate() * dagger(x2)*(x1) 
    V_lambda14 = L1/2 * (dagger(x1)*x1)*(dagger(x1)*x1) + L2/2 * (dagger(x2)*x2)*(dagger(x2)*x2) + L3 * (dagger(x1)*x1)*(dagger(x2)*x2) + L4 * (dagger(x1)*x2) * (dagger(x2)*x1)
    V_lambda57 = L5/2 * (dagger(x1)*x2)*(dagger(x1)*x2) + L6 * (dagger(x1)*x1)*(dagger(x1)*x2) + L7 * (dagger(x2)*x2)*(dagger(x1)*x2)
    V_lambda57_hc = L5.conjugate()/2 * (dagger(x2)*(x1))*(dagger(x2)*(x1)) + L6.conjugate() * (dagger(x1)*(x1))*(dagger(x2)*(x1)) + L7.conjugate() * (dagger(x2)*(x2))*(dagger(x2)*(x1))
    return V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc




# OBS NEED TO CHECK RODER OF omega_i! may be different fr gen and higgs!
def potential_VCT_gen(x1, x2, params, fields, bgfields):
    
    # Fields
    phi1, phi2, G0, a0, RGp, RHp, IGp, IHp = fields[0:8]
    omega1, omega2, omegaCP, omegaCB = bgfields[0], bgfields[1], bgfields[3], bgfields[5]
    
    # Parameters
    d_m11, d_m22, d_lambda1, d_lambda2, d_lambda3, d_lambda4, d_m12, d_lambda5, d_lambda6, d_lambda7 = params[:10]
    dT1, dT2, dTCP, dTCB = params[10:14]
    #d_D13 = params[14]

    #Second order
    V_mass = d_m11 * dagger(x1)*(x1) + d_m22 * dagger(x2)*(x2) - (d_m12 *dagger(x1)*(x2) + d_m12.conjugate() * dagger(x2)*(x1) )
    
    #Forth order
    V_lambda14 = d_lambda1/2 * (dagger(x1)*x1)**2 + d_lambda2/2 * (dagger(x2)*x2)**2 + d_lambda3 * (dagger(x1)*x1)*(dagger(x2)*x2) + d_lambda4 * (dagger(x1)*x2) * (dagger(x2)*x1)
    V_lambda57 = d_lambda5/2 * (dagger(x1)*x2)**2 + d_lambda6 * (dagger(x1)*x1)*(dagger(x1)*x2) + d_lambda7 * (dagger(x2)*x2)*(dagger(x1)*x2)
    V_lambda57_hc = d_lambda5.conjugate()/2 * (dagger(x2)* x1)**2 + d_lambda6.conjugate() * (dagger(x1)*x1)*(dagger(x2)*x1) + d_lambda7.conjugate() * (dagger(x2)*x2)*(dagger(x2)*x1)  #

    #First order
    #V1_CT = dT * (omega + phi1) + dTCB * (omegaCB + RHp)
    V1_CT = dT1 * (omega1 + phi1) + dT2 * (omega2 +phi2) + dTCP * (omegaCP +a0) + dTCB * (omegaCB + RHp)

    #Third order
    V3_CT_1 =  0 #d_D13*(phi1+omega1)**2*G0 

    # Return
    V1 = V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc
    V2 = V1_CT + V3_CT_1

    return V1, V2 # Return as Matrix, scalar

# OBS NEED TO CHECK RODER OF omega_i! may be different fr gen and higgs!
def potential_VCT_gen_expanded(x1, x2, params, fields, bgfields):
    # Fields
    phi1, phi2, G0, a0, RGp, RHp, IGp, IHp = fields[0:8]
    omega1, omega2, omegaCP, omegaCB = bgfields[0], bgfields[1], bgfields[3], bgfields[5]
    
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
    V1_CT = dT1 * (omega1 + phi1) + dT2 * (omega2 +phi2) + dTCP * (omegaCP +a0) + dTCB * (omegaCB + RHp)
    #third order
    d_D58, d_D56 = d_D67, d_D78
    V3_CT_1 = d_D33*G0**2*(phi1+omega1)+ d_D14*(phi1+omega1)**2*a0  + d_D24*(phi1+omega1)*phi2*a0
    V3_CT_1 +=  d_D13*(phi1+omega1)**2*G0 
    V3_CT_2 = (phi1+omega1) * (d_D67 * (RHp + omegaCB)*IGp - d_D58*RGp*IHp + d_D56*RGp*(RHp+omegaCB)+d_D78*IGp*IHp)
    
    # Return
    V1 = V_mass + V_lambda14 + V_lambda57 + V_lambda57_hc
    V2 = V1_CT + V3_CT_1 + V3_CT_2

    return V1, V2 # Return as Matrix, scalar