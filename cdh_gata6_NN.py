import numpy as np
from numba import jit
from simulation import Simulation, record_time
import backend
import cv2
import gata6_model_EP as EP1
import sys

def get_neighbor_forces(edge_forces, num_agents, cell_types, locations, radii, center, attraction_threshold, r_e=1.01,
                        u_bb=5, u_rb=1, u_yb=1, u_rr=20, u_ry=12, u_yy=30, u_rep=10000, grav=2, well_rad=325):
    
    adhesion_values = np.reshape(np.array([u_bb, u_rb, u_yb, u_yb, u_rr, u_ry, u_rb, u_ry, u_yy]), (3, 3))
    for i in range(num_agents):
        vecs, dist, repulsion, adhesion = calculate_pariwise_distances(locations[i], locations, center, radii[i] * 2, radii[i] * attraction_threshold)
        adh_dist = dist[adhesion].reshape((len(dist[adhesion]), 1))
        rep_dist = dist[repulsion].reshape((len(dist[repulsion]), 1))
        net_force_repulsion = u_rep * np.sum((vecs[repulsion] / rep_dist), axis=0)
        # print(f'{i}: {adhesion_values[cell_types[i], cell_types[adhesion]].reshape(-1, 1)}')
        # print(f'{i}: {(vecs[adhesion] * (adh_dist - r_e) / adh_dist)}')
        net_force_attraction = np.sum(adhesion_values[cell_types[i], cell_types[adhesion]].reshape(-1, 1) * (vecs[adhesion] * (adh_dist - r_e) / adh_dist), axis=0)
        # Gravity forces
        new_loc = locations[i] - center
        new_loc_sum = new_loc[0] ** 2 + new_loc[1] ** 2 + new_loc[2] ** 2
        edge_forces[i] += -1 * net_force_repulsion + net_force_attraction -grav * (new_loc / well_rad) * new_loc_sum ** (1/2)
    return edge_forces

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

def calculate_pariwise_distances(cell, locations, center, repulsion_distance, attraction_distance):
    '''
    Calculate pairwise distances between a cell and all other cells (locations)
    repulsion and adhesion are truth vectors of the two c onditions
    Return 1 if true, 0 if false
    '''
    cell_loc = cell - center
    vecs = locations - center - cell_loc
    dist = np.sqrt(np.sum(np.square(vecs), axis=1))
    repulsion = (dist < repulsion_distance) * (dist > 0)
    adhesion = (dist > repulsion_distance) * (dist <= attraction_distance)
    return vecs, dist, repulsion, adhesion


class GATA6_Adhesion_Simulation_NN(Simulation):
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
            "well_rad": 30,
            "output_values": True,
            "output_images": True,
            "image_quality": 300,
            "video_quality": 300,
            "fps": 5,
            "cell_rad": 0.5,
            "velocity": 0.3,
            "initial_seed_ratio": 0.5,
            "cell_interaction_rad": 3.2,
            "replication_type": None,
            "sub_ts": 600,
            "u_bb": 1,
            "u_rb": 1,
            "u_yb": 1,
            "u_rr": 30,
            "u_repulsion": 10000,
            "alpha": 10,
            "gravity": 2,
            "u_yy": 40,
            "u_ry": 1,
            "PACE": False
        }
        self.model_parameters(self.default_parameters)
        self.model_parameters(model_params)
        self.model_params = model_params

        # aba/dox/cho ratio
        self.nanog_ratio = 1 - (self.aba_ratio + self.inducer3_ratio)
        self.nanog_color = np.array([50, 50, 255], dtype=int) #blue
        self.gata6_color = np.array([50, 255, 50], dtype=int) #green
        self.foxa2_color = np.array([103, 46, 145], dtype=int) #purple

        self.cdh6_color = np.array([255, 255, 50], dtype=int) #yellow
        self.cdh1_color = np.array([255, 50, 50], dtype=int) #red

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
        self.add_agents(num_aba, agent_type="GATA6")
        self.add_agents(num_dox, agent_type="FOXA2")
        self.add_agents(num_nanog, agent_type="NANOG")

        # indicate agent arrays and create the arrays with initial conditions
        self.indicate_arrays("locations", "radii", "colors", "cell_type", "cdh_type", "division_set", "div_thresh")
        self.indicate_arrays('nanog_conc',
                             'gata6_conc',
                             'gata6_end_conc',
                             'gata6_syn_conc',
                             'copy_number',
                             'UN_TO_ME_counter')

        # generate random locations for cells
        self.locations = seed_cells(self.number_agents, self.center, self.initial_seed_rad)
        self.radii = self.agent_array(initial=lambda: self.cell_rad)

        # Define cell types, 2 is ABA, 1 is DOX, 0 is non-cadherin expressing cho cells
        self.cdh_type = self.agent_array(dtype=int, initial={"NANOG": lambda: 0})
        self.cell_type = self.agent_array(dtype=int, initial={"NANOG": lambda: 0})
        self.colors = self.agent_array(dtype=int, vector=3, initial={"NANOG": lambda: self.nanog_color,
                                                                     "GATA6": lambda: self.gata6_color,
                                                                     "FOXA2": lambda: self.foxa2_color})
        self.outline_colors = self.agent_array(dtype=int, vector=3, initial={"NANOG": lambda: (0, 0, 0),
                                                                             "NANOG": lambda: (0, 0, 0),
                                                                             "NANOG": lambda: (0, 0, 0)})

        # setting division times (in seconds):
        # Not used in model
        self.div_thresh = self.agent_array(initial={"NANOG": lambda: 1, "GATA6": lambda: 1, "FOXA2": lambda: 1})
        self.division_set = self.agent_array(initial={"NANOG": lambda: 0, "GATA6": lambda: 0, "FOXA2": lambda: 0})

        #indicate and create graphs for identifying neighbors
        #self.indicate_graphs("neighbor_graph")

        self.nanog_conc = self.agent_array(initial=0)
        self.nanog_conc = np.random.normal(14, size=self.number_agents)
        self.gata6_conc = self.agent_array(initial=0)
        self.gata6_end_conc = self.agent_array(initial=0)
        self.gata6_end_conc = np.random.rand(self.number_agents) * 0.15
        self.gata6_syn_conc = self.agent_array(initial=0)
        self.copy_number = self.agent_array(initial=0)
        self.copy_number = np.random.normal(5, size=self.number_agents)

        self.gata6_conc = self.gata6_syn_conc + self.gata6_end_conc

        #stochastic gene circuit
        self.transition_rate = calculate_rate(self.inducer3_ratio + self.aba_ratio, self.end_step, self.cdh_induction_step)
        self.transition = np.random.rand(self.number_agents)

        #cell state transition
        self.UN_TO_ME_counter = np.zeros(self.number_agents)
        self.solve_odes()

        # save parameters to text file
        self.save_params(self.model_params)

        # record initial values
        self.step_values()
        self.step_image()

    @record_time
    def step(self):
        """ Overrides the step() method from the Simulation class.
        """
        # preform subset force calculations
        for i in range(self.sub_ts):
            # get all neighbors within threshold (1.6 * diameter)
            #self.get_neighbors(self.neighbor_graph, self.cell_interaction_rad * self.cell_rad)
            # move the cells and track total repulsion vs adhesion forces
            self.move_parallel()
        # get the following data. We can generate images at each time step, but right now that is not needed.

        # add/remove agents from the simulation
        # self.update_populations()
        print(f'Num_FOXA2: {len(np.argwhere(self.cell_type == 2))}, Num_GATA6: {len(np.argwhere(self.cell_type == 1))}')

        # Update cell fate
        # right now the model updates every 2 hours..
        if self.current_step % 2 == 0:
            self.solve_odes()
        self.cell_fate()

        self.step_values()
        self.step_image()
        self.temp()
        self.data()

    def end(self):
        """ Overrides the end() method from the Simulation class.
        """
        self.step_values()
        self.step_image()
        #self.create_video()

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
        self.nanog_conc, self.gata6_conc, self.gata6_end_conc, self.gata6_syn_conc = EP1.dudt(
            self.nanog_conc,
            self.gata6_conc,
            self.gata6_end_conc,
            self.gata6_syn_conc,
            self.dox,
            self.copy_number,
            1)

    def cell_fate(self):
        if self.current_step > self.cdh_induction_step:
            # Updating Cadherins
            transition_threshold = calculate_transition_rate(self.transition_rate, self.current_step, self.cdh_induction_step)
            committed_cells = self.transition < transition_threshold
            for i in range(self.number_agents):
                if self.cdh_type[i] == 0 and committed_cells[i]:
                    self.cdh_type[i] = np.random.binomial(1, self.aba_ratio / (self.inducer3_ratio + self.aba_ratio),  1) + 1
        self.UN_TO_ME_counter[self.gata6_conc > self.gata6_threshold] += 1
        self.UN_TO_ME_counter[self.gata6_conc < self.gata6_threshold] -= 1
        self.UN_TO_ME_counter[self.UN_TO_ME_counter < 0] = 0

        transition_prob = self.UN_TO_ME_counter ** 4 / (self.UN_TO_ME_counter ** 4 + 16)
        x = np.random.rand(self.number_agents)
        self.cell_type[((x < transition_prob) * (self.cell_type == 0))] = 1

        self.update_colors()

    def update_colors(self):
        ref = np.array([(0,0,0), self.cdh1_color, self.cdh6_color])
        color_ref = np.array([self.nanog_color, self.gata6_color, self.foxa2_color])
        self.outline_colors = ref[self.cdh_type]
        self.colors = color_ref[self.cell_type]

    @record_time
    def move_parallel(self):
        neighbor_forces = np.zeros((self.number_agents, 3))
        neighbor_forces = get_neighbor_forces(neighbor_forces, self.number_agents, self.cell_type, self.locations, self.radii, self.center, self.cell_interaction_rad, 
                                              u_bb=self.u_bb, u_rb=self.u_rb, u_rr=self.u_rr, u_yb=self.u_yb, u_ry=self.u_ry, u_yy=self.u_yy, u_rep=self.u_repulsion)
        noise_vector = 1 * np.random.uniform(-1, 1, (self.number_agents, 3)) * self.alpha
        neighbor_forces = neighbor_forces + noise_vector
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
                outline_color = (int(self.outline_colors[index][2]),
                                 int(self.outline_colors[index][1]),
                                 int(self.outline_colors[index][0]))

                # draw the agent and a black outline to distinguish overlapping agents
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, color, -1)
                image = cv2.ellipse(image, (x, y), (major, minor), 0, 0, 360, outline_color, 2)

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

    @classmethod
    def simulation_mode_2(cls, name, output_dir, model_params):
        """ Turns existing simulation images into a video.

            :param name: The name of the simulation.
            :param output_dir: Path to simulation output directory.
            :type name: str
            :type output_dir: str
        """
        # make simulation object for video/path information
        sim = cls(model_params)
        sim.name = name
        sim.set_paths(output_dir)

        # compile all simulation images into a video
        sim.create_video()

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

def parameter_sweep_abm(par, directory, dox_induction_step, cdh_induction_step, induction_value, dox_ratio, aba_ratio, final_ts=45):
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
        "cdh_induction_step": cdh_induction_step,
        "dox_induction_step": dox_induction_step,
        "induction_value": induction_value,
        "gata6_threshold": 6,
        "end_step": final_ts,
        "inducer3_ratio": dox_ratio,
        "aba_ratio": aba_ratio,
        "PACE": False
    }
    name = f'{induction_value}dox_at_{dox_induction_step}_cdh_at_{cdh_induction_step}dox{dox_ratio}_aba{aba_ratio}_Delaunay'
    sim = GATA6_Adhesion_Simulation_NN(model_params)
    sim.start_sweep(directory, model_params, name)
    return par, sim.image_quality, sim.image_quality, 3, final_ts/sim.sub_ts

if __name__ == "__main__":
    model_params = {
        "cdh_induction_step": 12,
        "dox_induction_step": 12,
        "induction_value": 0.5,
        "gata6_threshold": 6,
        "end_step": 120,
        "inducer3_ratio": 0.5,
        "aba_ratio": 0.2,
        "PACE": False
    }
    sim = GATA6_Adhesion_Simulation_NN(model_params)
    if sys.platform == 'win32':
        sim.start("C:\\Users\\ajin40\\Documents\\sim_outputs\\cdh_gata6_sims\\outputs")
    elif sys.platform == 'darwin':
        sim.start("/Users/andrew/Projects/sim_outputs/cdh_gata6_sims/outputs", model_params)
    else:
        print('exiting...')
