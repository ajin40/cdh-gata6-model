'''
NANOG-GATA6-FOXA2-FOXF1 MODEL with separated protein and rna
MODEL A V2
In this system, FOXF1 is activated at at low GATA6 and inhibited at high GATA6,
FOXA2 is activated by GATA6
We may observe few differences in dgata6 due to the dominating hill-like equation in the system
'''
def hill_act(X, K, n):
    return X**n / (K**n + X**n)

def hill_inh(X, K, n):
    return K**n / (K**n + X**n)

def dudt(xyz, t, dox, p):
    
    nanog, gata6, y1, y2, z1, z2 = xyz


    optimized_params = [5.37352557, 0.5, 0.15164532, 0.5, 3.41323608,
    8.08063479, 0.5, 8.07576539, 5.52822741, 0.5, 4.33834965,
    7.0079098, 5.77644675, 6.86557976, 1.17319128, 5.2299828,  9.7703752,
    8.80489012, 8.41532703, 4.81912212]


    c, d, e, f, g, h, i, j, k, l = optimized_params[:10]
    KMxy1, KMxy2, KM11, KM12, KM1x, KM1y, KM22, KM21, KM2x, KM2y = optimized_params[10:]
    nx, n11, n12, n1x, n1y, n22, n21, n2x, n2y = [2]*9

    b0 = 7
    a1 = 4
    b1 = 2.1
    a2 = 4
    b2 = 4
    a3 = 0.5
    b3 = 0.5
    KM_NN, KM_GN, KM_GG, KM_NG, KM_dox = [5, 5, 5, 5, 4.5]
    n = m = 4

    dox_induction = 12
    if t > 0:
        dox_induction = dox
    dnanog_dt = a1 * hill_act(nanog, KM_NN, n) * a2 * hill_inh(gata6, KM_GN, m) - a3 * nanog
    dgata6_dt = b0 * hill_act(dox_induction*p, KM_dox, n) + b1 * hill_act(gata6, KM_GG, n) * b2 * hill_inh(nanog, KM_NG, m) - b3 * gata6
    dy1_dt = c * hill_act(gata6, KMxy1, nx) - d * y1
    dy2_dt = e * hill_act(gata6, KMxy2, nx) - f * y2
    dz1_dt = g * hill_act(gata6, KM1x, n1x) * hill_inh(y1, KM1y, n1y) - i * z1 + h * hill_act(z1, KM11, n11) * hill_inh(z2, KM12, n12)
    dz2_dt = j * hill_act(gata6, KM2x, n2x) * hill_inh(y2, KM2y, n2y) - l * z2 + k * hill_act(z2, KM22, n22) * hill_inh(z1, KM21, n21)

    return nanog + dnanog_dt * t, gata6 + dgata6_dt * t, y1 + dy1_dt * t, y2 + dy2_dt * t, z1 + dz1_dt * t, z2 + dz2_dt * t, dnanog_dt, dgata6_dt, dz1_dt, dz2_dt

