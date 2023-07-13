import sys
from cdh_gata6_coupled import *

if __name__ == '__main__':
    induction1 = int(sys.argv[1])
    conc = float(sys.argv[2])
    a = parameter_sweep_abm(0, "/Users/andrew/PycharmProjects/ST_CHO_adhesion_model/", induction1, conc, final_ts=120)