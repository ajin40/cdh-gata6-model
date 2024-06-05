import numpy as np

"""
Here are a list of all the reactions for the complete GATA6 CNV model (33 species, 66 RXNs):
P0_NANOG + NANOG -> P1_NANOG
P1_NANOG -> P0_NANOG + NANOG
P1_NANOG + NANOG -> P2_NANOG
P2_NANOG -> P1_NANOG + NANOG
P0_NANOG + GATA6 -> Pi_NANOG
Pi_NANOG -> P0_NANOG + GATA6

P0_GATA6 + GATA6 -> P1_GATA6
P1_GATA6 -> P0_GATA6 + GATA6
P1_GATA6 + GATA6 -> P2_GATA6
P2_GATA6 -> P1_GATA6 + GATA6
P0_GATA6 + NANOG -> Pi_GATA6
Pi_GATA6 -> P0_GATA6 + NANOG

P0_synGATA6 + TETo_DOX -> P1_synGATA6
P1_synGATA6 -> TETo_DOX + P0_synGATA6

P0_FOXA2 + GATA6 -> P1_FOXA2
P1_FOXA2 -> P0_FOXA2 + GATA6
P1_FOXA2 + GATA6 -> P2_FOXA2
P2_FOXA2 -> P1_FOXA2 + GATA6
P2_FOXA2 + FOXA2 -> P3_FOXA2
P3_FOXA2 -> P2_FOXA2 + FOXA2
P3_FOXA2 + FOXA2 -> P4_FOXA2
P4_FOXA2 -> P3_FOXA2 + FOXA2
P0_FOXA2 + FOXF1 -> Pi_FOXA2
Pi_FOXA2 -> P0_FOXA2 + FOXF1

P0_FOXF1 + GATA6 -> P1_FOXF1
P1_FOXF1 -> P0_FOXF1 + GATA6
P1_FOXF1 + GATA6 -> P2_FOXF1
P2_FOXF1 -> P1_FOXF1 + GATA6
P2_FOXF1 + FOXF1 -> P3_FOXF1
P3_FOXF1 -> P2_FOXF1 + FOXF1
P3_FOXF1 + FOXF1 -> P4_FOXF1
P4_FOXF1 -> P3_FOXF1 + FOXF1
P0_FOXF1 + FOXA2 -> Pi_FOXF1
Pi_FOXF1 -> P0_FOXF1 + FOXA2

P0_NANOG -> NANOG_MRNA
P1_NANOG -> NANOG_MRNA
P2_NANOG -> NANOG_MRNA
NANOG_MRNA -> NANOG

P0_GATA6 -> GATA6_MRNA
P1_GATA6 -> GATA6_MRNA
P2_GATA6 -> GATA6_MRNA
P0_synGATA6 -> GATA6_MRNA
P1_synGATA6 -> GATA6_MRNA
GATA6_MRNA -> GATA6

P0_FOXA2 -> FOXA2_MRNA
P1_FOXA2 -> FOXA2_MRNA
P2_FOXA2 -> FOXA2_MRNA
P3_FOXA2 -> FOXA2_MRNA
P4_FOXA2 -> FOXA2_MRNA
FOXA2_MRNA -> FOXA2

P0_FOXF1 -> FOXF1_MRNA
P1_FOXF1 -> FOXF1_MRNA
P2_FOXF1 -> FOXF1_MRNA
P3_FOXF1 -> FOXF1_MRNA
P4_FOXF1 -> FOXF1_MRNA
FOXF1_MRNA -> FOXF1

TETo + DOX -> TETo_DOX
TETo_DOX -> TETo + DOX

NANOG_MRNA ->
GATA6-MRNA ->
FOXA2_MRNA ->
FOXF1_MRNA ->
NANOG ->
GATA6 ->
FOXA2 ->
FOXF1 ->

"""

"""
List of Species:
0: P0_NANOG
1: P1_NANOG
2: P2_NANOG
3: Pi_NANOG
4: P0_GATA6
5: P1_GATA6
6: P2_GATA6
7: Pi_GATA6
8: P0_synGATA6
9: P1_synGATA6
10: P0_FOXA2
11: P1_FOXA2
12: P2_FOXA2
13: P3_FOXA2
14: P4_FOXA2
15: Pi_FOXA2
16: P0_FOXF1
17: P1_FOXF1
18: P2_FOXF1
19: P3_FOXF1
20: P4_FOXF1
21: Pi_FOXF1
22: TETo
23: TETo_DOX
24: DOX
25: NANOG_MRNA
26: NANOG
27: GATA6_MRNA
28: GATA6
29: FOXA2_MRNA
30: FOXA2
31: FOXF1_MRNA
32: FOXF1
"""
class StochasticCNV():
    def __init__(self,
                 initial_dox,
                 gata6_copy_number,
                 initial_conditions='initial_conditions.csv',
                 stoichiometry_file='reaction_stoichiometry.csv',
                 reaction_rates='reaction_rates.csv'):
        self.reaction_stoichiometries = np.loadtxt(stoichiometry_file, skiprows=1, delimiter=',', dtype=int)
        self.reaction_rates = np.loadtxt(reaction_rates, delimiter=',', dtype=float)
        self.state_vector = np.loadtxt(initial_conditions, delimiter=',', dtype=float)
        self.state_vector[24] = initial_dox
        self.state_vector[8] = gata6_copy_number
        self.reaction_propensities = np.zeros(len(self.reaction_stoichiometries))
        self.critical_rxns = np.zeros(len(self.reaction_propensities), dtype=bool)

        self.g = np.zeros(len(self.state_vector))
        self.mu = np.zeros(len(self.state_vector))
        self.sigma = np.zeros(len(self.state_vector))
        self.dt = 0.0001
        self.t = 0

        self.ts = [self.t]
        self.states = [self.state_vector]
        self.list_rxns = []
        self.update_propensities()

    
    def update_propensities(self):
        # p_NANOG binding reactions
        self.reaction_propensities[0] = self.reaction_rates[0] * self.state_vector[0] * self.state_vector[26]
        self.reaction_propensities[1] = self.reaction_rates[1] * self.state_vector[1]
        self.reaction_propensities[2] = self.reaction_rates[2] * self.state_vector[1] * self.state_vector[26]
        self.reaction_propensities[3] = self.reaction_rates[3] * self.state_vector[2]
        self.reaction_propensities[4] = self.reaction_rates[4] * self.state_vector[0] * self.state_vector[28]
        self.reaction_propensities[5] = self.reaction_rates[5] * self.state_vector[3]
        # p_GATA6 binding reactions
        self.reaction_propensities[6] = self.reaction_rates[6] * self.state_vector[4] * self.state_vector[28]
        self.reaction_propensities[7] = self.reaction_rates[7] * self.state_vector[5]
        self.reaction_propensities[8] = self.reaction_rates[8] * self.state_vector[5] * self.state_vector[28]
        self.reaction_propensities[9] = self.reaction_rates[9] * self.state_vector[6]
        self.reaction_propensities[10] = self.reaction_rates[10] * self.state_vector[4] * self.state_vector[26]
        self.reaction_propensities[11] = self.reaction_rates[11] * self.state_vector[7]
        # p_synGATA6 binding reactions
        self.reaction_propensities[12] = self.reaction_rates[12] * self.state_vector[8] * self.state_vector[23]
        self.reaction_propensities[13] = self.reaction_rates[13] * self.state_vector[9]
        # p_FOXA2 binding reactions
        self.reaction_propensities[14] = self.reaction_rates[14] * self.state_vector[10] * self.state_vector[28]
        self.reaction_propensities[15] = self.reaction_rates[15] * self.state_vector[11]
        self.reaction_propensities[16] = self.reaction_rates[16] * self.state_vector[11] * self.state_vector[28]
        self.reaction_propensities[17] = self.reaction_rates[17] * self.state_vector[12]
        self.reaction_propensities[18] = self.reaction_rates[18] * self.state_vector[12] * self.state_vector[30]
        self.reaction_propensities[19] = self.reaction_rates[19] * self.state_vector[13]
        self.reaction_propensities[20] = self.reaction_rates[20] * self.state_vector[13] * self.state_vector[30]
        self.reaction_propensities[21] = self.reaction_rates[21] * self.state_vector[14]
        self.reaction_propensities[22] = self.reaction_rates[22] * self.state_vector[10] * self.state_vector[32]
        self.reaction_propensities[23] = self.reaction_rates[23] * self.state_vector[15]
        # p_FOXF1 binding reactions
        self.reaction_propensities[24] = self.reaction_rates[24] * self.state_vector[16] * self.state_vector[28]
        self.reaction_propensities[25] = self.reaction_rates[25] * self.state_vector[17]
        self.reaction_propensities[26] = self.reaction_rates[26] * self.state_vector[17] * self.state_vector[28]
        self.reaction_propensities[27] = self.reaction_rates[27] * self.state_vector[18]
        self.reaction_propensities[28] = self.reaction_rates[28] * self.state_vector[18] * self.state_vector[30]
        self.reaction_propensities[29] = self.reaction_rates[29] * self.state_vector[19]
        self.reaction_propensities[30] = self.reaction_rates[30] * self.state_vector[19] * self.state_vector[30]
        self.reaction_propensities[31] = self.reaction_rates[31] * self.state_vector[20]
        self.reaction_propensities[32] = self.reaction_rates[32] * self.state_vector[16] * self.state_vector[32]
        self.reaction_propensities[33] = self.reaction_rates[33] * self.state_vector[21]
        # TETo + DOX reactions
        self.reaction_propensities[34] = self.reaction_rates[34] * self.state_vector[22] * self.state_vector[24]
        self.reaction_propensities[35] = self.reaction_rates[35] * self.state_vector[23]
        # NANOG production reactions
        self.reaction_propensities[36] = self.reaction_rates[36] * self.state_vector[0]
        self.reaction_propensities[37] = self.reaction_rates[37] * self.state_vector[1]
        self.reaction_propensities[38] = self.reaction_rates[38] * self.state_vector[2]
        self.reaction_propensities[39] = self.reaction_rates[39] * self.state_vector[25]
        # GATA6 production reactions
        self.reaction_propensities[40] = self.reaction_rates[40] * self.state_vector[4]
        self.reaction_propensities[41] = self.reaction_rates[41] * self.state_vector[5]
        self.reaction_propensities[42] = self.reaction_rates[42] * self.state_vector[6]
        self.reaction_propensities[43] = self.reaction_rates[43] * self.state_vector[8]
        self.reaction_propensities[44] = self.reaction_rates[44] * self.state_vector[9]
        self.reaction_propensities[45] = self.reaction_rates[45] * self.state_vector[27]
        # FOXA2 production reactions
        self.reaction_propensities[46] = self.reaction_rates[46] * self.state_vector[10]
        self.reaction_propensities[47] = self.reaction_rates[47] * self.state_vector[11]
        self.reaction_propensities[48] = self.reaction_rates[48] * self.state_vector[12]
        self.reaction_propensities[49] = self.reaction_rates[49] * self.state_vector[13]
        self.reaction_propensities[50] = self.reaction_rates[50] * self.state_vector[14]
        self.reaction_propensities[51] = self.reaction_rates[51] * self.state_vector[29]        
        # FOXF1 production reactions
        self.reaction_propensities[52] = self.reaction_rates[52] * self.state_vector[16]
        self.reaction_propensities[53] = self.reaction_rates[53] * self.state_vector[17]
        self.reaction_propensities[54] = self.reaction_rates[54] * self.state_vector[18]
        self.reaction_propensities[55] = self.reaction_rates[55] * self.state_vector[19]
        self.reaction_propensities[56] = self.reaction_rates[56] * self.state_vector[20]
        self.reaction_propensities[57] = self.reaction_rates[57] * self.state_vector[31]
        # mRNA degradation reactions
        self.reaction_propensities[58] = self.reaction_rates[58] * self.state_vector[25]
        self.reaction_propensities[59] = self.reaction_rates[59] * self.state_vector[27]
        self.reaction_propensities[60] = self.reaction_rates[60] * self.state_vector[29]
        self.reaction_propensities[61] = self.reaction_rates[61] * self.state_vector[31]
        # Protein degradation reactions
        self.reaction_propensities[62] = self.reaction_rates[62] * self.state_vector[26]
        self.reaction_propensities[63] = self.reaction_rates[63] * self.state_vector[28]
        self.reaction_propensities[64] = self.reaction_rates[64] * self.state_vector[30]
        self.reaction_propensities[65] = self.reaction_rates[65] * self.state_vector[32]

    def run_tau_leaping(self, tf, save_file=False, file_name=''):
        recalculate = True
        while self.t < tf:
            if recalculate:
                self.update_propensities()
                tau2 = self.critical_reactions_leap()
                tau1 = self.noncritical_reaction_leap()
                if tau1 < tau2:
                    self.dt = tau1
                    n_reactions = np.random.poisson(self.reaction_propensities * self.dt).reshape((len(self.reaction_propensities), 1))
                    n_reactions[self.critical_rxns] = 0
                else:
                    self.dt = tau2
                    n_reactions = np.random.poisson(self.reaction_propensities * self.dt).reshape((len(self.reaction_propensities), 1))
                    n_reactions[self.critical_rxns] = 0
                    critical_rxn_propensities = self.reaction_propensities * self.critical_rxns
                    event = sample_discrete(critical_rxn_propensities)
                    n_reactions[event] = 1
            temp_vector = self.state_vector
            temp_vector += np.sum(self.reaction_stoichiometries * n_reactions, axis=0)
            if np.amin(temp_vector) >= 0:         
                self.t += self.dt
                self.state_vector = temp_vector
                self.ts.append(self.t)
                self.states.append(self.state_vector)
                self.list_rxns.append(n_reactions)
                recalculate=True
            else:
                recalculate=False
                self.dt = self.dt/2


        ts = np.array(self.ts)
        states = np.array(self.states)
        if save_file:
            np.savetxt(f'{file_name}_ts.csv', ts)
            np.savetxt(f'{file_name}_states.csv', states, delimiter=',')
            return ts, states
        else:
            return ts, states

    def run_gillespie(self, tf, save_file=False, file_name=''):
        while self.t < tf:
            self.update_propensities()
            prop_sum = np.sum(self.reaction_propensities)
            time = -np.log(1. - np.random.random()) / prop_sum
            rxn_probs = np.random.random() * prop_sum
            event = len(self.reaction_propensities) - sum(np.cumsum(self.reaction_propensities) > rxn_probs)
            self.state_vector += self.reaction_stoichiometries[event,:]
            self.t += time

            self.ts.append(self.t)
            self.states.append(self.state_vector)

        ts = np.array(self.ts)
        states = np.array(self.states)
        if save_file:
            np.savetxt(f'{file_name}_ts.csv', ts)
            np.savetxt(f'{file_name}_states.csv', states, delimiter=',')
            return ts, states
        else:
            return ts, states

    def noncritical_reaction_leap(self, eps=0.03):
        max_mu = np.zeros(len(self.state_vector))
        max_sigma = np.zeros(len(self.state_vector))
        for i in range(len(self.state_vector)):
                # from Cao et al. (2006) g_i is 1 if the highest order reaction is 1, 2 f the highest order reaction is 2,
                # Unless any reaction requires two molecules, which in this case is never true. 
            self.g[i] = np.sum(self.reaction_stoichiometries[np.argmin(self.reaction_stoichiometries[:,i]), :] < 0)
            self.mu[i] = np.sum(self.reaction_stoichiometries[:,i] * ~self.critical_rxns * self.reaction_propensities)
            self.sigma[i] = np.sum(np.square(self.reaction_stoichiometries[:,i]) * ~self.critical_rxns * self.reaction_propensities)
            if self.mu[i] > 0:
                max_mu[i] = max(1, eps * self.state_vector[i] / self.g[i]) / abs(self.mu[i])
            else:
                max_mu[i] = np.inf
            if self.sigma[i] > 0:
                max_sigma[i] = max(1, eps * self.state_vector[i] / self.g[i]) ** 2 / self.sigma[i]
            else:
                max_sigma[i] = np.inf
        tau = np.amin([max_mu, max_sigma])
        return tau
    
    

    def identify_critical_reactions(self, nc):
        self.critical_rxns[:] = 0
        L_j = np.zeros(len(self.reaction_propensities))
        for j in range(len(self.reaction_propensities)):
            if self.reaction_propensities[j] > 0:
                # skip
                for i in range(len(self.reaction_stoichiometries[j, :])):
                    if self.reaction_stoichiometries[j, i] < 0:
                        L_j[j] += self.reaction_stoichiometries[j, i]
        L_j_index = np.argsort(L_j)
        for i in range(nc):
            if L_j[L_j_index[i]] >= 0:
                break
            else:
                self.critical_rxns[L_j_index[i]] = 1
        return self.critical_rxns
    
    def critical_reactions_leap(self, nc=10):
        critical_rxns = self.identify_critical_reactions(nc)
        prop_sum = np.sum(self.reaction_propensities[critical_rxns])
        tau = np.random.exponential(scale=1/prop_sum)
        return tau

            
def sample_discrete(probs):
    q = np.random.rand()
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1