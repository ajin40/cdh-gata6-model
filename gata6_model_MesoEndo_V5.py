'''
NANOG-GATA6-FOXA2-FOXF1 MODEL with separated protein and rna
MODEL A V2
In this system, FOXF1 is activated at at low GATA6 and inhibited at high GATA6,
FOXA2 is activated by GATA6
We may observe few differences in dgata6 due to the dominating hill-like equation in the system
'''

import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from matplotlib import colors

def nanog_rna(nanog, NANOG, GATA6, nanog_basal_prod, nanog_max, GATA6_Km, GATA6_n, NANOG_Km, NANOG_n):
    return nanog_max * ((NANOG/NANOG_Km) ** NANOG_n) / ((1 + (GATA6/GATA6_Km) ** GATA6_n) * (1 + (NANOG/NANOG_Km) ** NANOG_n)) + nanog_basal_prod - nanog

def NANOG(nanog, NANOG, beta_n):
    return -beta_n * (NANOG - nanog)

def gata6_rna(gata6, GATA6, NANOG, dox, copy_num, gata6_basal_prod, gata6_max, NANOG_Km, NANOG_n, dox_Km, dox_n, GATA6_Km, GATA6_n):
    return gata6_max * copy_num * (((GATA6/GATA6_Km) ** GATA6_n) * (dox/dox_Km) ** dox_n) / ((1 + (NANOG/NANOG_Km) ** NANOG_n) * (1 + (GATA6/GATA6_Km) ** GATA6_n) * (1 + (dox/dox_Km) ** dox_n)) + gata6_basal_prod - gata6

def GATA6(gata6, GATA6, beta_g):
    return -beta_g * (GATA6 - gata6)

def foxa2_rna(foxa2, FOXA2, GATA6, FOXF1, foxa2_basal_prod, foxa2_max, GATA6_Km, GATA6_n, FOXF1_Km, FOXF1_n, FOXA2_Km, FOXA2_n):
    return foxa2_max * (((GATA6/GATA6_Km) ** GATA6_n) * ((FOXA2/FOXA2_Km) ** FOXA2_n)) / ((1 + (FOXF1/FOXF1_Km) ** FOXF1_n) * (1 + (GATA6/GATA6_Km) ** GATA6_n) * (1 + (FOXA2/FOXA2_Km) ** FOXA2_n)) + foxa2_basal_prod - foxa2

def FOXA2(foxa2, FOXA2, beta_a):
    return -beta_a * (FOXA2 - foxa2)

def foxf1_rna(foxf1, FOXF1, GATA6, FOXA2, foxf1_basal_prod, foxf1_max, GATA6_Km, GATA6_n, FOXA2_Km, FOXA2_n, FOXF1_Km, FOXF1_n):
    return foxf1_max * (((GATA6/GATA6_Km) ** GATA6_n) * ((FOXF1/FOXF1_Km) ** FOXF1_n)) / ((1 + (FOXA2/FOXA2_Km) ** FOXA2_n) * (1 + (GATA6/GATA6_Km) ** GATA6_n) * (1 + (FOXF1/FOXF1_Km) ** FOXF1_n)) + foxf1_basal_prod - foxf1

def FOXF1(foxf1, FOXF1, beta_f):
    return -beta_f * (FOXF1 - foxf1)

def dudt(X, t, dox, copy_n):
    basal_prod = [0.1, 0.1, 0.1, 0.1]
    Kms = [4, #GATA6 inhibition of NANOG
        3, #NANOG self-activation
        #################################
        2, #NANOG inhibition of GATA6
        0.025, #Dox induced activation of GATA6
        1.7, #GATA6 self-activation
        #################################
        10, #GATA6 induced activation of FOXA2
        1.25, #FOXF1 inhibition of FOXA2
        1, #FOXA2 self-activation
        #################################
        4.5, #GATA6 induced activation of FOXF1
        0.8, #FOXA2 inhibition of FOXF1
        1  #FOXF1 self-activation
    ]
    ns =  [1, #GATA6 inhibition of NANOG
        2, #NANOG self-activation
        #################################
        1.2, #NANOG inhibition of GATA6
        0.8, #Dox induced activation of GATA6
        1, #GATA6 self-activation
        #################################
        2, #GATA6 induced activation of FOXA2
        1, #FOXF1 inhibition of FOXA2
        0.7, #FOXA2 self-activation
        #################################
        1, #GATA6 induced activation of FOXF1
        8, #FOXA2 inhibition of FOXF1
        0.5 #FOXF1 self-activation
    ]
    vmaxes = [8, #nanog
            2.5, #gata6
            14, #foxa2
            12  #foxf1
    ]
    betas = [0.2, #NANOG 
            0.2, #GATA6
            0.2, #FOXA2
            0.2  #FOXF1
    ]
    
    dnanog_rna = nanog_rna(X[0], X[1], X[2], basal_prod[0], vmaxes[0], Kms[0], ns[0], Kms[1], ns[1])
    dNANOG_prot = NANOG(X[0], X[1], betas[0])
    dgata6_rna = gata6_rna(X[2], X[3], X[1], dox, copy_n, basal_prod[1], vmaxes[1], Kms[2], ns[2], Kms[3], ns[3], Kms[4], ns[4])
    dGATA6_prot = GATA6(X[2], X[3], betas[1])
    dfoxa2_rna = foxa2_rna(X[4], X[5], X[3], X[7], basal_prod[2], vmaxes[2], Kms[5], ns[5], Kms[6], ns[6], Kms[7], ns[7])
    dFOXA2_prot = FOXA2(X[4], X[5], betas[2])
    dfoxf1_rna = foxf1_rna(X[6], X[7], X[3], X[5], basal_prod[3], vmaxes[3], Kms[8], ns[8], Kms[9], ns[9], Kms[10], ns[10])
    dFOXF1_prot = FOXF1(X[6], X[7], betas[3])
    return X[0] + dnanog_rna * t, X[1] + dNANOG_prot * t, X[2] + dgata6_rna * t, X[3] + dGATA6_prot * t, X[4] + dfoxa2_rna * t, X[5] + dFOXA2_prot * t, X[6] + dfoxf1_rna * t, X[7] + dFOXF1_prot * t