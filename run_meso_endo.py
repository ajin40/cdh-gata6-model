import sys
from cdh_gata6_coupled_Meso_Endo import *

if __name__ == '__main__':
    induction1 = int(sys.argv[1])
    conc = float(sys.argv[2])

    if sys.platform == 'win32':
        outputs = "C:\\Users\\ajin40\\Documents\\sim_outputs\\cdh_gata6_sims\\outputs"
    elif sys.platform == 'darwin':
        outputs = "/Users/andrew/Projects/sim_outputs/cdh_gata6_sims/outputs"
    else:
        print('exiting...')
    a = parameter_sweep_abm(0, outputs, induction1, conc, final_ts=240)