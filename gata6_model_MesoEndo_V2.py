'''
NANOG-GATA6-FOXA2-FOXF1 MODEL with separated protein and rna
MODEL A V2
In this system, FOXF1 is activated at at low GATA6 and inhibited at high GATA6,
FOXA2 is activated by GATA6
We may observe few differences in dgata6 due to the dominating hill-like equation in the system
'''

def dnanog_rna(nanog, NANOG, GATA6, a1, a2, a3, n, m, KM_NN, KM_GN):
    return (a1 * NANOG ** n)/(KM_NN ** n + NANOG ** n) * (a2 * KM_GN ** m)/(KM_GN ** m + GATA6 ** m) - a3 * nanog

def dgata6_rna(gata6, NANOG, GATA6, dox, p, a1, a2, a3, n, m, KM_GG, KM_NG):
    return dox * p + (a1 * GATA6 ** n)/(KM_GG ** n + GATA6 ** n) * (a2 * KM_NG ** m)/(KM_NG ** m + NANOG ** m) - a3 * gata6

def dfoxa2_rna(foxa2, GATA6, FOXF1, a0, a1, a2, n, m, KM_12, KM_G2):
    return (a0 * GATA6 ** n)/(KM_G2 ** n + GATA6 ** n) * (a1 * KM_12 ** m)/(KM_12 ** m + FOXF1 ** m) - a2 * foxa2

def dfoxf1_rna(foxf1, GATA6, FOXA2, a0, a1, a2, a3, n, m, KM_21, KM_GA, KM_GI):
    return a0 * ((GATA6 ** n)/ (KM_GA ** n + GATA6 ** n) * (KM_GI ** n)/(KM_GI ** n + GATA6 ** n)) * (a2 * KM_21 ** m)/(KM_21 ** m + FOXA2 ** m) - a3 * foxf1

def dNANOG(NANOG, nanog, a1, a2):
    return a1 * nanog - a2 * NANOG

def dGATA6(GATA6, gata6, a1, a2):
    return a1 * gata6 - a2 * GATA6

def dFOXA2(FOXA2, foxa2, a1, a2):
    return a1 * foxa2 - a2 * FOXA2

def dFOXF1(FOXF1, foxf1, a1, a2):
    return a1 * foxf1 - a2 * FOXF1

def dudt(U, t, dox, p):
    nanog, gata6, foxa2, foxf1, NANOG, GATA6, FOXA2, FOXF1 = U
    a0 = 4
    b0 = 4
    a1 = 4
    b1 = 4
    a2 = 4
    b2 = 4
    a3 = 0.5
    b3 = 0.5
    KM_NN, KM_GN, KM_GG, KM_NG, KM_11, KM_22, KM_12, KM_21 = [5, 5, 5, 5, 5, 5, 5, 5]
    KM_G1A, KM_G1I, KM_G2 = [40, 55, 55]
    n = m = 4
    gata6_n = 8
    fox_n = 2

    protein_degradation_rate = 0.125
    translation_rate = 0.167

    dnanog_dt = dnanog_rna(nanog, NANOG, GATA6, a1, a2, a3, n, m, KM_NN, KM_GN) * t
    dgata6_dt = dgata6_rna(gata6, NANOG, GATA6, dox, p, b1, b2, b3, n, m, KM_GG, KM_NG) * t
    dGATA6_dt = dGATA6(GATA6, gata6, translation_rate, protein_degradation_rate) * t
    dNANOG_dt = dNANOG(NANOG, nanog, translation_rate, protein_degradation_rate) * t
    dfoxa2_dt = dfoxa2_rna(foxa2, GATA6, FOXF1, a0, a1, a3, gata6_n, fox_n, KM_12, KM_G2) * t
    dfoxf1_dt = dfoxf1_rna(foxf1, GATA6, FOXA2, b0, b0, b2, b3, gata6_n, fox_n, KM_21, KM_G1A, KM_G1I) * t
    dFOXA2_dt = dFOXA2(FOXA2, foxa2, translation_rate, protein_degradation_rate) * t
    dFOXF1_dt = dFOXF1(FOXF1, foxf1, translation_rate, protein_degradation_rate) * t 
    return nanog + dnanog_dt, gata6 + dgata6_dt, foxa2 + dfoxa2_dt, foxf1 + dfoxf1_dt, NANOG + dNANOG_dt, GATA6 + dGATA6_dt, FOXA2 + dFOXA2_dt, FOXF1 + dFOXF1_dt, dNANOG_dt, dGATA6_dt, dFOXA2_dt, dFOXF1_dt