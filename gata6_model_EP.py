import numpy as np

## REMEMBER 1 ts = 2 hr in this model.

def dnanog(NANOG, GATA6, a1, a2, a3, a4, a5):
    return (1 + a1 * NANOG ** a2)/(1 + NANOG ** a2) * (1 + a3 * GATA6 ** a4) / (1 + GATA6 ** a4) - a5 * NANOG

def dgata6_end(NANOG, GATA6, a1, a2, a3, a4, a5):
    return (1 + a1 * GATA6 ** a2)/(1 + GATA6 ** a2) * (1 + a3 * NANOG ** a4) / (1 + NANOG ** a4) - a5 * GATA6

def dgata6_syn(dox, p, GATA6, a6):
    return dox * p - GATA6 * a6

def dudt(NANOG, GATA6, GATA6_end, GATA6_syn, dox, p, t):
    a1 = 6
    a2 = 3.5
    a3 = 0.01
    a4 = 1.5
    a5 = 0.4

    b1 = 6
    b2 = 2.5
    b3 = 0.07
    b4 = 3.5
    b5 = 0.5
    b6 = 0.8

    dnanog_dt = dnanog(NANOG, GATA6, a1, a2, a3, a4, a5) * t
    dgata6_end_dt = dgata6_end(NANOG, GATA6_end, b1, b2, b3, b4, b5) * t
    dgata6_syn_dt = dgata6_syn(dox, p, GATA6_syn, b6) * t
    NANOG += dnanog_dt
    GATA6_end += dgata6_end_dt
    GATA6_syn += dgata6_syn_dt
    GATA6 = GATA6_end + GATA6_syn
    return NANOG, GATA6, GATA6_end, GATA6_syn