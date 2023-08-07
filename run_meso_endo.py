import sys
from cdh_gata6_coupled_Meso_Endo import *
import datetime

def parameter_sweep_abm(par, directory, dox_induction_step, induction_value, final_ts=45):
    """ Run model with specified parameters
        :param par: simulation number
        :param directory: Location of model outputs. A folder titled 'outputs' is required
        :param RR: Adhesion value for R-R cell interactions
        :param YY: Adhesion value for Y-Y cell interactions
        :param RY: Adhesion value for R-Y cell interactions
        :param dox_ratio: final ratio of Red cells at simulation end. 1 - (dox_ratio + aba_ratio) = # of remaining uncommitted blue cells.
        :param aba_ratio: final ratio of Yellow cells at simulation end.
        :param final_ts: Final timestep. 60 ts = 96h, 45 ts = 72h
        :type par int
        :type directory: String
        :type RR: float
        :type YY: float
        :type RY: float
        :type dox_ratio: float
        :type aba_ratio: float
        :type final_ts: int
    """
    if sys.platform == 'win32':
        model_params = {
            "dox_induction_step": 12,
            "induction_value": 0.8,
            "gata6_threshold": 40,
            "foxa2_threshold": 15,
            "end_step": 120,
            "PACE": False,
            "cuda": True
        }
    elif sys.platform == 'darwin':
        model_params = {
            "dox_induction_step": 12,
            "induction_value": 0.8,
            "gata6_threshold": 40,
            "foxa2_threshold": 15,
            "end_step": 120,
            "PACE": False,
            "cuda": False
        }
    elif sys.platform =='linux':
        model_params = {
            "dox_induction_step": 12,
            "induction_value": 0.8,
            "gata6_threshold": 40,
            "foxa2_threshold": 15,
            "end_step": 120,
            "PACE": False,
            "cuda": False
        }
    name = f'{datetime.date.today()}_MESO_ENDO_{induction_value}dox_at_{dox_induction_step}'
    sim = GATA6_Adhesion_Coupled_Simulation(model_params)
    start_sweep(sim, directory, name)
    return par, sim.image_quality, sim.image_quality, 3, final_ts/sim.sub_ts

def start_sweep(sim, output_dir, name):
    """ Configures/runs the model based on the specified
        simulation mode.
    """
    # check that the output directory exists and get the name/mode for the simulation
    output_dir = backend.check_output_dir(output_dir)
    sim.name = backend.check_existing(name, output_dir, new_simulation=True)
    sim.set_paths(output_dir)

    sim.full_setup()
    sim.run_simulation()

if __name__ == '__main__':
    induction1 = int(sys.argv[1])
    conc = float(sys.argv[2])

    if sys.platform == 'win32':
        outputs = "C:\\Users\\ajin40\\Documents\\sim_outputs\\cdh_gata6_sims\\outputs"
    elif sys.platform == 'darwin':
        outputs = "/Users/andrew/Projects/sim_outputs/cdh_gata6_sims/outputs"
    elif sys.platform =='linux':
        outputs = '/home/ajin40/models/model_outputs'
    else:
        print('I did not plan for another system platform... exiting...')
    a = parameter_sweep_abm(0, outputs, induction1, conc, final_ts=240)