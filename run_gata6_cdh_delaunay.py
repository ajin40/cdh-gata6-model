import sys
from cdh_gata6_NN import *

if __name__ == '__main__':
    induction1 = int(sys.argv[1])
    induction2 = int(sys.argv[2])
    conc = float(sys.argv[3])
    cdh1_ratio = float(sys.argv[4])
    cdh6_ratio = float(sys.argv[5])

    if sys.platform == 'win32':
        outputs = "C:\\Users\\ajin40\\Documents\\sim_outputs\\cdh_gata6_sims\\outputs"
    elif sys.platform == 'darwin':
        outputs = "/Users/andrew/Projects/sim_outputs/cdh_gata6_sims/outputs"
    else:
        print('exiting...')
    a = parameter_sweep_abm(0, outputs, induction1, induction2, conc,
                            cdh1_ratio, cdh6_ratio, final_ts=120)