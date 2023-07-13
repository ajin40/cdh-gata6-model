import sys
from cdh_gata6 import *

if __name__ == '__main__':
    induction1 = int(sys.argv[1])
    induction2 = int(sys.argv[2])
    conc = float(sys.argv[3])
    cdh1_ratio = float(sys.argv[4])
    cdh6_ratio = float(sys.argv[5])
    a = parameter_sweep_abm(0, "/Users/andrew/PycharmProjects/ST_CHO_adhesion_model/", induction1, induction2, conc,
                            cdh1_ratio, cdh6_ratio, final_ts=120)