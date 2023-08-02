import numpy as np
from numba import jit
from simulation import Simulation, record_time
import backend
import cv2
import gata6_model_MesoEndo_V2 as RD
import datetime
import sys

@jit(nopython=True, parallel=True)
def get_neighbor_forces(number_edges, edges, edge_forces, locations, center, types, radius, r_e=1.01,
                        u_00=1, u_11=1, u_22=1, u_33=1, u_repulsion=10000):
    adhesion_values = np.ones((4, 4))
    adhesion_values[0, 0] = u_00
    adhesion_values[1, 1] = u_11
    adhesion_values[2, 2] = u_22
    adhesion_values[3, 3] = u_33
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]
        # get cell positions
        cell_1_loc = locations[cell_1] - center
        cell_2_loc = locations[cell_2] - center

        # get new location position
        vec = cell_2_loc - cell_1_loc
        dist2 = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2

        # based on the distance apply force differently
        if dist2 == 0:
            edge_forces[index][0] = 0
            edge_forces[index][1] = 0
        else:
            dist = dist2 ** (1/2)
            if 0 < dist2 < (2 * radius) ** 2:
                edge_forces[index][0] = -1 * u_repulsion * (vec / dist)
                edge_forces[index][1] = 1 * u_repulsion * (vec / dist)
            else:
                # get the cell type
                cell_1_type = types[cell_1]
                cell_2_type = types[cell_2]
                u = adhesion_values[cell_1_type, cell_2_type]
                # get value prior to applying type specific adhesion const
                value = (dist - r_e) * (vec / dist)
                edge_forces[index][0] = u * value
                edge_forces[index][1] = -1 * u * value
    return edge_forces


@jit(nopython=True, parallel=True)
def get_gravity_forces(number_cells, locations, center, well_rad, net_forces, grav=1):
    for index in range(number_cells):
        new_loc = locations[index] - center
        grav_vector = np.array([-2*1/well_rad*new_loc[0], -2*1/well_rad*new_loc[1], -1])
        mag = (grav_vector[0] ** 2 + grav_vector[1] ** 2 + grav_vector[2] ** 2) ** (1/2)
        net_forces[index] = grav * grav_vector/mag
    return net_forces


@jit(nopython=True)
def convert_edge_forces(number_edges, edges, edge_forces, neighbor_forces):
    for index in range(number_edges):
        # get indices of cells in edge
        cell_1 = edges[index][0]
        cell_2 = edges[index][1]

        neighbor_forces[cell_1] += edge_forces[index][0]
        neighbor_forces[cell_2] += edge_forces[index][1]

    return neighbor_forces

def seed_cells(num_agents, center, radius):
    theta = 2 * np.pi * np.random.rand(num_agents).reshape(num_agents, 1)
    phi = np.pi/2 * np.random.rand(num_agents).reshape(num_agents, 1)
    rad = radius * np.sqrt(np.random.rand(num_agents)).reshape(num_agents, 1)
    x = rad * np.cos(theta) * np.sin(phi) + center[0]
    y = rad * np.sin(theta) * np.sin(phi) + center[1]
    z = rad * np.cos(phi) + center[2]
    locations = np.hstack((x, y, z))
    return locations

def calculate_rate(combined_percent, end_step, induction_step):
    return 1 - np.power(1-combined_percent, 1/(end_step-induction_step))

def calculate_transition_rate(combined_percent, t, induction_step):
    transition_rate = 1 - (1-combined_percent) ** (t - induction_step)
    return transition_rate

class GATA6_Adhesion_Coupled_Simulation(Simulation):
    """ This class inherits the Simulation class allowing it to run a
        simulation with the proper functionality.
    """
    def __init__(self, model_params):
        # initialize the Simulation object
        Simulation.__init__(self)

        self.default_parameters = {
            "num_to_start": 1000,
            "cuda": False,
            "size": [1, 1, 1],
            "well_rad": 50,
            "output_values": True,
            "output_images": True,
            "image_quality": 500,
            "video_quality": 500,
            "fps": 5,
            "cell_rad": 0.5,
            "velocity": 0.3,
            "initial_seed_ratio": 0.5,
            "cell_interaction_rad": 3.2,
            "replication_type": 'Contact_Inhibition',
            "sub_ts": 600,
            "u_00": 1,
            "u_11": 30,
            "u_22": 30,
            "u_33": 30,
            "u_repulsion": 10000,
            "alpha": 10,
            "gravity": 10,
            "PACE": False
        }
        self.model_parameters(self.default_parameters)
        self.model_parameters(model_params)
        self.model_params = model_params

        # aba/dox/cho ratio
        self.nanog_color = np.array([50, 50, 255], dtype=int) #blue
        self.gata6_color = np.array([255, 255, 50], dtype=int) #yellow
        self.foxa2_color = np.array([255, 50, 50], dtype=int) #red
        self.foxf1_color = np.array([50, 255, 50]) #green

        self.dox = 0

        self.initial_seed_rad = self.well_rad * self.initial_seed_ratio
        self.dim = np.asarray(self.size)
        self.size = self.dim * self.well_rad
        self.center = np.array([self.size[0] / 2, self.size[1] / 2, 0])

    def setup(self):
        """ Overrides the setup() method from the Simulation class.
        """
        # determine the number of agents for each cell type, pre-seeded
        # num_aba = int(self.num_to_start * self.aba_ratio)
        # num_dox = int(self.num_to_start * self.dox_ratio)
        # num_cho = int(self.num_to_start * self.cho_ratio)

        # Seeding cells with stochastic transition rates
        num_aba = 0
        num_dox = 0
        num_nanog = int(self.num_to_start)

        # add agents to the simulation
        self.add_agents(0, agent_type="GATA6")
        self.add_agents(0, agent_type="FOXA2")
        self.add_agents(0, agent_type="FOXF1")
        self.add_agents(num_nanog, agent_type="NANOG")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "division_set", "div_thresh")
        self.indicate_arrays('copy_number',
                             'UN_TO_ME_counter',
                             'ME_TO_E_counter',
                             'ME_TO_M_counter')

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.center, self.initial_seed_rad)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # Define cell types, 2 is ABA, 1 is DOX, 0 is non-cadherin expressing cho cells
        self.cell_type = self.agent_array(dtype=int, initial={"NANOG": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"NANOG": lambda: self.nanog_color,
                                                                     "GATA6": lambda: self.gata6_color,
                                                                     "FOXA2": lambda: self.foxa2_color,
                                                                     "FOXF1": lambda: self.foxf1_color})

        # setting division times (in seconds):
        # Not used in model
        self.div_thresh = self.agent_array(initial={"NANOG": lambda: 18, "GATA6": lambda: 51, "FOXA2": lambda: 51, "FOXF1": lambda: 51})
        self.division_set = self.agent_array(initial={"NANOG": lambda: np.random.rand() * 18, "GATA6": lambda: 0, "FOXA2": lambda: 0, "FOXF1": lambda: 0})

        #indicate and create graphs for identifying neighbors
        self.indicate_graphs("neighbor_graph")
        self.neighbor_graph = self.agent_graph()

        # Intracellular Concentrations:
        self.intracellular_gradient_names = ['nanog_rna_conc',
                                            'gata6_rna_conc',
                                            'foxa2_rna_conc',
                                            'foxf1_rna_conc',
                                            'NANOG_conc',
                                            'GATA6_conc',
                                            'FOXA2_conc',
                                            'FOXF1_conc']
        for name in self.intracellular_gradient_names:
            self.indicate_arrays(name)
            self.__dict__[name] = self.agent_array(initial=0)

        self.intracellular_dProt_names = ['dNANOG',
                                        'dGATA6',
                                        'dFOXA2',
                                        'dFOXF1']
        
        for name in self.intracellular_dProt_names:
            self.indicate_arrays(name)
            self.__dict__[name] = self.agent_array(initial=0)

        self.nanog_rna_conc = np.random.normal(32, size=self.number_agents)
        self.NANOG_conc = np.random.normal(43, size=self.number_agents)
        self.copy_number = self.agent_array(initial=0)
        self.copy_number = np.random.uniform(2, 16, size=self.number_agents)

        #cell state transition
        self.UN_TO_ME_counter = np.zeros(self.number_agents)
        self.ME_TO_M_counter = np.zeros(self.number_agents)
        self.ME_TO_E_counter = np.zeros(self.number_agents)
        self.solve_odes()
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()

        # save parameters to text file
        self.save_params(self.model_params)

        # record initial values
        self.step_values()
        self.step_image()

    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform subset force calculations
        num_sub_steps = 1
        if self.number_agents - len(np.argwhere(self.cell_type == 0)) > 0:
            # It is taking too long to simulate cell-cell interactions where no sorting is really happening
            num_sub_steps = self.sub_ts
        for i in range(num_sub_steps):
            # get all neighbors within threshold (1.6 * diameter)
            self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()
        # get the following data. We can generate images at each time step, but right now that is not needed.


        # Update cell fate
        self.solve_odes()
        self.cell_fate()


        # add/remove agents from the simulation
        self.reproduce(1)
        self.update_populations()
        print(f'Num GATA6: {len(np.argwhere(self.cell_type == 1))},\nNum FOXA2: {len(np.argwhere(self.cell_type == 2))},\nNum FOXF1: {len(np.argwhere(self.cell_type==3))}')

        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        self.create_video()

    @record_time
    def update_populations(self):
        """ Adds/removes agents to/from the simulation by adding/removing
            indices from the cell arrays and any graphs.
        """
        # get indices of hatching/dying agents with Boolean mask
        add_indices = np.arange(self.number_agents)[self.hatching]
        remove_indices = np.arange(self.number_agents)[self.removing]

        # count how many added/removed agents
        num_added = len(add_indices)
        num_removed = len(remove_indices)
        # go through each agent array name
        for name in self.array_names:
            # copy the indices of the agent array data for the hatching agents
            copies = self.__dict__[name][add_indices]

            # add indices to the arrays
            self.__dict__[name] = np.concatenate((self.__dict__[name], copies), axis=0)

            # if locations array
            if name == "locations":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # move distance of radius in random direction
                    vec = self.radii[i] * self.random_vector()
                    self.__dict__[name][mother] += vec
                    self.__dict__[name][daughter] -= vec

            # reset division time
            if name == "division_set":
                # go through the number of cells added
                for i in range(num_added):
                    # get mother and daughter indices
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # set division counter to zero
                    self.__dict__[name][mother] = 0
                    self.__dict__[name][daughter] = 0

            # set new division threshold
            if name == "division_threshold":
                # go through the number of cells added
                for i in range(num_added):
                    # get daughter index
                    daughter = self.number_agents + i

                    # set division threshold based on cell type
                    self.__dict__[name][daughter] = 1
            
            if 'conc' in name:
                for i in range(num_added):
                    mother = add_indices[i]
                    daughter = self.number_agents + i

                    # split concentrations in 2
                    self.__dict__[name][daughter] = self.__dict__[name][mother] / 2
                    self.__dict__[name][mother] = self.__dict__[name][mother] / 2

            # remove indices from the arrays
            self.__dict__[name] = np.delete(self.__dict__[name], remove_indices, axis=0)

        # go through each graph name
        for graph_name in self.graph_names:
            # add/remove vertices from the graph
            self.__dict__[graph_name].add_vertices(num_added)
            self.__dict__[graph_name].delete_vertices(remove_indices)

        # change total number of agents and print info to terminal
        self.number_agents += num_added
        # print("\tAdded " + str(num_added) + " agents")
        # print("\tRemoved " + str(num_removed) + " agents")

        # clear the hatching/removing arrays for the next step
        self.hatching[:] = False
        self.removing[:] = False

    def solve_odes(self):
        if self.current_step > self.dox_induction_step:
            self.dox = self.induction_value
        # Updating GATA6 + NANOG
        values = RD.dudt(self.get_concentrations(),
                            self.dox,
                            self.copy_number,
                            1)
        i = 0
        for name in self.intracellular_gradient_names + self.intracellular_dProt_names:
            self.__dict__[name] = values[i]
            i+= 1
        

    def cell_fate(self):
        '''
        # Using Chad's Thresholding rules... may need to fit the probability functions 
        self.UN_TO_ME_counter[self.GATA6_conc > self.gata6_threshold] += 1
        self.UN_TO_ME_counter[self.GATA6_conc < self.gata6_threshold] -= 1
        self.UN_TO_ME_counter[self.UN_TO_ME_counter < 0] = 0

        self.ME_TO_E_counter[((self.FOXA2_conc > self.foxa2_threshold) * (self.cell_type == 1))] += 1
        self.ME_TO_E_counter[((self.FOXA2_conc < self.foxa2_threshold) * (self.cell_type == 1))] -= 1
        self.ME_TO_M_counter[((self.FOXA2_conc < self.foxa2_threshold) * (self.cell_type == 1))] += 1
        self.ME_TO_M_counter[((self.FOXA2_conc > self.foxa2_threshold) * (self.cell_type == 1))] -= 1
        self.ME_TO_M_counter[self.ME_TO_M_counter < 0] = 0
        self.ME_TO_E_counter[self.ME_TO_E_counter < 0] = 0

        transition_prob_ME = self.UN_TO_ME_counter ** 4 / (self.UN_TO_ME_counter ** 4 + 4 ** 4)
        transition_prob_E = self.ME_TO_E_counter ** 4 / (self.ME_TO_E_counter ** 4 + 24 ** 4)
        transition_prob_M = self.ME_TO_M_counter ** 4 / (self.ME_TO_M_counter ** 4 + 24 ** 4)
        x = np.random.rand(self.number_agents)
        self.cell_type[((x < transition_prob_ME) * (self.cell_type == 0))] = 1
        self.cell_type[((x < transition_prob_E) * (self.cell_type == 1))] = 2
        self.cell_type[((x < transition_prob_M) * (self.cell_type == 1))] = 3
        '''
        # SS cell_fate determiniation
        SS = 1e-02
        self.cell_type[((self.GATA6_conc > self.gata6_threshold) * (self.cell_type == 0))] = 1
        self.cell_type[((self.FOXA2_conc > self.foxa2_threshold) * (self.cell_type == 1) * (abs(self.dFOXA2) < SS))] = 2
        self.cell_type[((self.FOXA2_conc < self.foxa2_threshold) * (self.cell_type == 1) * (abs(self.dFOXF1) < SS))] = 3

        self.update_colors()

    def update_colors(self):
        color_ref = np.array([self.nanog_color, self.gata6_color, self.foxa2_color, self.foxf1_color])
        self.colors = color_ref[self.cell_type]

    @record_time
    def move_parallel(self):
        edges = np.asarray(self.neighbor_graph.get_edgelist())
        num_edges = len(edges)
        edge_forces = np.zeros((num_edges, 2, 3))
        neighbor_forces = np.zeros((self.number_agents, 3))
        # get adhesive/repulsive forces from neighbors and gravity forces
        edge_forces = get_neighbor_forces(num_edges, edges, edge_forces, self.locations, self.center, self.cell_type,
                                          self.cell_rad, u_00=self.u_00, u_11=self.u_11, u_22=self.u_22, u_33=self.u_33,
                                          u_repulsion=self.u_repulsion)
        neighbor_forces = convert_edge_forces(num_edges, edges, edge_forces, neighbor_forces)
        noise_vector = np.ones((self.number_agents, 3)) * self.alpha * (2 * np.random.rand(self.number_agents, 3) - 1)
        neighbor_forces = neighbor_forces + noise_vector
        if self.gravity > 0:
            net_forces = np.zeros((self.number_agents, 3))
            gravity_forces = get_gravity_forces(self.number_agents, self.locations, self.center,
                                                self.well_rad*2, net_forces, grav=self.gravity)
            neighbor_forces = neighbor_forces + gravity_forces
        for i in range(self.number_agents):
            vec = neighbor_forces[i]
            sum = vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2
            if sum != 0:
                neighbor_forces[i] = neighbor_forces[i] / (sum ** (1/2))
            else:
                neighbor_forces[i] = 0
        # update locations based on forces
        self.locations += 2 * self.velocity * self.cell_rad * neighbor_forces
        # check that the new location is within the space, otherwise use boundary values
        self.locations = np.where(self.locations > self.well_rad, self.well_rad, self.locations)
        self.locations = np.where(self.locations < 0, 0, self.locations)

    def step_image(self, background=(0, 0, 0), origin_bottom=True):
        """ Creates an image of the simulation space.

            :param background: The 0-255 RGB color of the image background.
            :param origin_bottom: If true, the origin will be on the bottom, left of the image.
            :type background: tuple
            :type origin_bottom: bool
        """
        # only continue if outputting images
        if self.output_images:
            # get path and make sure directory exists
            backend.check_direct(self.images_path)

            # get the size of the array used for imaging in addition to the scaling factor
            x_size = self.image_quality
            scale = x_size / self.size[0]
            y_size = int(np.ceil(scale * self.size[1]))

            # create the agent space background image and apply background color
            image = np.zeros((y_size, x_size, 3), dtype=np.uint8)
            background = (background[2], background[1], background[0])
            image[:, :] = background

            # go through all of the agents
            for index in range(self.number_agents):
                # get xy coordinates, the axis lengths, and color of agent
                x, y = int(scale * self.locations[index][0]), int(scale * self.locations[index][1])
                major, minor = int(scale * self.radii[index]), int(scale * self.radii[index])
                color = (int(self.colors[index][2]), int(self.colors[index][1]), int(self.colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, (0,0,0), 1)

            # if the origin should be bottom-left flip it, otherwise it will be top-left
            if origin_bottom:
                image = cv2.flip(image, 0)

            # save the image as a PNG
            image_compression = 4  # image compression of png (0: no compression, ..., 9: max compression)
            file_name = f"{self.name}_image_{self.current_step}.png"
            cv2.imwrite(self.images_path + file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, image_compression])

    #Unused..
    @record_time
    def reproduce(self, ts):
        """ If the agent meets criteria, hatch a new agent.
        """
        # increase division counter by time step for all agents
        self.division_set += ts
        #go through all agents marking for division if over the threshold
        if self.replication_type == 'Contact_Inhibition':
            adjacency_matrix = self.neighbor_graph.get_adjacency()
            for index in range(self.number_agents):
                if self.division_set[index] > self.div_thresh[index]:
                    # 12 is the maximum number of cells that can surround a cell
                    if np.sum(adjacency_matrix[index,:]) < 12:
                        self.mark_to_hatch(index)
        if self.replication_type == 'Default':
            for index in range(self.number_agents):
                if self.division_set[index] > self.div_thresh[index]:
                    self.mark_to_hatch(index)
        if self.replication_type == 'None':
            return

    @classmethod
    def simulation_mode_0(cls, name, output_dir, model_params):
        """ Creates a new brand new simulation and runs it through
            all defined steps.
        """
        # make simulation instance, update name, and add paths
        sim = cls(model_params)
        sim.name = name
        sim.set_paths(output_dir)

        # set up the simulation agents and run the simulation
        sim.full_setup()
        sim.run_simulation()

    def save_params(self, params):
        """ Add the instance variables to the Simulation object based
            on the keys and values from a YAML file.
        """

        # iterate through the keys adding each instance variable
        with open(self.main_path + "parameters.txt", "w") as parameters:
            for key in list(params.keys()):
                parameters.write(f"{key}: {params[key]}\n")
        parameters.close()

    @classmethod
    def start_sweep(cls, output_dir, model_params, name):
        """ Configures/runs the model based on the specified
            simulation mode.
        """
        # check that the output directory exists and get the name/mode for the simulation
        output_dir = backend.check_output_dir(output_dir)

        name = backend.check_existing(name, output_dir, new_simulation=True)
        cls.simulation_mode_0(name, output_dir, model_params)

    def get_concentrations(self):
        conc_list = []
        for name in self.intracellular_gradient_names:
            conc_list.append(self.__dict__[name])
        return conc_list


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
    model_params = {
        "dox_induction_step": dox_induction_step,
        "induction_value": induction_value,
        "gata6_threshold": 40,
        "foxa2_threshold": 15,
        "end_step": final_ts,
        "PACE": False
    }
    name = f'{datetime.date.today()}_MESO_ENDO_{induction_value}dox_at_{dox_induction_step}'
    sim = GATA6_Adhesion_Coupled_Simulation(model_params)
    sim.start_sweep(directory, model_params, name)
    return par, sim.image_quality, sim.image_quality, 3, final_ts/sim.sub_ts


if __name__ == "__main__":
    model_params = {
        "dox_induction_step": 12,
        "induction_value": 0.8,
        "gata6_threshold": 40,
        "foxa2_threshold": 15,
        "end_step": 120,
        "PACE": False
    }
    sim = GATA6_Adhesion_Coupled_Simulation(model_params)
    if sys.platform == 'win32':
        sim.start("C:\\Users\\ajin40\\Documents\\sim_outputs\\cdh_gata6_sims\\outputs", model_params)
    elif sys.platform == 'darwin':
        sim.start("/Users/andrew/Projects/sim_outputs/cdh_gata6_sims/outputs", model_params)
    else:
        print('exiting...')
