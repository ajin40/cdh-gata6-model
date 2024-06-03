import scipy
import scipy.integrate
import numpy as np
from Chemical import *

class GATA6_response_model:

    """
    CNV Model is a base class for the four-gene system.

    The set of Chemical states (all 5 mRNAs, all 4 proteins, and all 10
    promoter states) have a defined ordering, as stored in the
    dictionary chemIndex, which maps Chemicals to integer indices in
    the ordering.  This ordering is necessary to communicate with utilities
    (such as scipy.integrate.odeint) that require chemical amounts to
    be stored as a 1D array.  The GetStateVector method returns the
    Chemical amounts as a scipy array in the appropriate order, and
    the SetStateVector method sets the Chemical amounts based on a
    supplied array.
    """

    def __init__(self,
                 GATA6,
                 activated_transcription_rate,
                 double_activated_transcription_rate,
                 triple_activated_transcription_rate,
                 quad_activated_transcription_rate,
                 P_act_binding,
                 P_inh_binding,
                 P1_act_unbinding,
                 P1_inh_unbinding,
                 P1_act_binding,
                 P2_act_unbinding,
                 P2_act_binding,
                 P3_act_unbinding,
                 P3_act_binding,
                 P4_act_unbinding,
                 FOXA2_prot=0,
                 FOXF1_prot=0,
                 mRNA_degradation_rate=np.log(2) / 120.,
                 protein_degradation_rate=np.log(2) / 600.,
                 translation_rate=0.0167,
                 unocc_transcription_rate=5.0e-08,
                 inhibited_transcription_rate=[5.0e-08, 5.0e-08]):
        
        self.t = 0.
        self.chemIndex = {}
        self.reactions = []

        # Rates
        self.mRNA_degradation_rate = mRNA_degradation_rate
        self.protein_degradation_rate = protein_degradation_rate
        self.translation_rate = translation_rate
        self.unocc_transcription_rate = unocc_transcription_rate

        self.p1_transcription_rate = activated_transcription_rate
        self.p2_transcription_rate = double_activated_transcription_rate
        self.p3_transcription_rate = triple_activated_transcription_rate
        self.p4_transcription_rate = quad_activated_transcription_rate
        self.inh_transcription_rate = inhibited_transcription_rate

        self.P_act_binding = P_act_binding
        self.P_inh_binding = P_inh_binding
        self.P1_act_unbinding = P1_act_unbinding
        self.P1_inh_unbinding = P1_inh_unbinding
        self.P1_act_binding = P1_act_binding
        self.P2_act_unbinding = P2_act_unbinding
        self.P2_act_binding = P2_act_binding
        self.P3_act_unbinding = P3_act_unbinding
        self.P3_act_binding = P3_act_binding
        self.P4_act_unbinding = P4_act_unbinding
        
        # mRNAs
        self.foxa2_mrna = Chemical(0.0)
        self.foxf1_mrna = Chemical(0.0) 
        
        # Proteins
        self.FOXA2_prot = Chemical(FOXA2_prot)
        self.FOXF1_prot = Chemical(FOXF1_prot)
        self.GATA6_prot = Chemical(GATA6)
        # Promoter States
        # Unbound (Default) each promoter has 2 copies. except exogenous gata6 (The synthetic Circuit)
        self.Pfoxa2 = Chemical(2.0)
        self.Pfoxf1 = Chemical(2.0)
        
        # Single bound GATA6 activated States
        self.Pfoxa2_GATA6 = Chemical(0.0)
        self.Pfoxf1_GATA6 = Chemical(0.0)

        # Double bound GATA6 activated States
        self.Pfoxa2_GATA6_GATA6 = Chemical(0.0)
        self.Pfoxf1_GATA6_GATA6 = Chemical(0.0)
        
        # Single bound inhibited states
        self.Pfoxa2_FOXF1 = Chemical(0.0)
        self.Pfoxf1_FOXA2 = Chemical(0.0)
        
        # Double bound activated states + single bound self-activator
        self.Pfoxa2_GATA6_GATA6_FOXA2 = Chemical(0.0)
        self.Pfoxf1_GATA6_GATA6_FOXF1 = Chemical(0.0)
        
        # Cooperatively double bound active states
        self.Pfoxa2_GATA6_GATA6_FOXA2_FOXA2 = Chemical(0.0)
        self.Pfoxf1_GATA6_GATA6_FOXF1_FOXF1 = Chemical(0.0)
        
        # References for indexing

        self.mRNAs = [self.foxa2_mrna, self.foxf1_mrna]
        self.proteins = [self.FOXA2_prot, self.FOXF1_prot, self.GATA6_prot]
        self.P0 = [self.Pfoxa2, self.Pfoxf1]
        self.P1_active = [self.Pfoxa2_GATA6, self.Pfoxf1_GATA6]
        self.P2_active = [self.Pfoxa2_GATA6_GATA6, self.Pfoxf1_GATA6_GATA6]
        self.P1_inhib = [self.Pfoxa2_FOXF1, self.Pfoxf1_FOXA2]
        self.P3_active = [self.Pfoxa2_GATA6_GATA6_FOXA2, self.Pfoxf1_GATA6_GATA6_FOXF1]
        self.P4_active = [self.Pfoxa2_GATA6_GATA6_FOXA2_FOXA2, self.Pfoxf1_GATA6_GATA6_FOXF1_FOXF1]

        for chem in self.mRNAs:
            self.AddChemical(chem)
        for chem in self.proteins:
            self.AddChemical(chem)
        for chem in self.P0:
            self.AddChemical(chem)
        for chem in self.P1_active:
            self.AddChemical(chem)
        for chem in self.P1_inhib:
            self.AddChemical(chem)
        for chem in self.P2_active:
            self.AddChemical(chem)
        for chem in self.P3_active:
            self.AddChemical(chem)
        for chem in self.P4_active:
            self.AddChemical(chem)
            
            
        # Adding mRNA reactions        
        # mRNA into Protein        
        self.AddReaction(CatalyzedSynthesisReaction(self.foxa2_mrna, self.FOXA2_prot, self.translation_rate))
        self.AddReaction(CatalyzedSynthesisReaction(self.foxf1_mrna, self.FOXF1_prot, self.translation_rate))

        # #Protein Degradation
        self.AddReaction(DegradationReaction(self.FOXF1_prot, self.protein_degradation_rate))
        self.AddReaction(DegradationReaction(self.FOXA2_prot, self.protein_degradation_rate))

        # non-activated mRNA production
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1, self.foxf1_mrna, self.unocc_transcription_rate))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2, self.foxa2_mrna, self.unocc_transcription_rate))

        # mRNA degradation
        self.AddReaction(DegradationReaction(self.foxf1_mrna, self.mRNA_degradation_rate))
        self.AddReaction(DegradationReaction(self.foxa2_mrna, self.mRNA_degradation_rate))

        # Activated mRNA production
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6, self.foxa2_mrna, self.p1_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6, self.foxf1_mrna, self.p1_transcription_rate[1]))

        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6_GATA6, self.foxa2_mrna, self.p2_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6_GATA6, self.foxf1_mrna, self.p2_transcription_rate[1]))

        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6_GATA6_FOXA2, self.foxa2_mrna, self.p3_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6_GATA6_FOXF1, self.foxf1_mrna, self.p3_transcription_rate[1]))

        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6_GATA6_FOXA2_FOXA2, self.foxa2_mrna, self.p4_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6_GATA6_FOXF1_FOXF1, self.foxf1_mrna, self.p4_transcription_rate[1]))

        # Inhibited mRNA production
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_FOXF1, self.foxa2_mrna, self.inh_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_FOXA2, self.foxf1_mrna, self.inh_transcription_rate[1]))

        ## P0 -> P1
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2, self.GATA6_prot, self.Pfoxa2_GATA6, self.P_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6, self.Pfoxa2, self.GATA6_prot, self.P1_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1, self.GATA6_prot, self.Pfoxf1_GATA6, self.P_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6, self.Pfoxf1, self.GATA6_prot, self.P1_act_unbinding[1])
        )

        ### P0 -> Inhibited
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2, self.FOXF1_prot, self.Pfoxa2_FOXF1, self.P_inh_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_FOXF1, self.Pfoxa2, self.FOXF1_prot, self.P1_inh_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1, self.FOXA2_prot, self.Pfoxf1_FOXA2, self.P_inh_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_FOXA2, self.Pfoxf1, self.FOXA2_prot, self.P1_inh_unbinding[1])
        )
      
        ### P1 -> P2
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2_GATA6, self.GATA6_prot, self.Pfoxa2_GATA6_GATA6, self.P1_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6_GATA6, self.Pfoxa2_GATA6, self.GATA6_prot, self.P2_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1_GATA6, self.GATA6_prot, self.Pfoxf1_GATA6_GATA6, self.P1_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6_GATA6, self.Pfoxf1_GATA6, self.GATA6_prot, self.P2_act_unbinding[1])
        )

        ## P2 -> P3
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2_GATA6_GATA6, self.FOXA2_prot, self.Pfoxa2_GATA6_GATA6_FOXA2, self.P2_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6_GATA6_FOXA2, self.Pfoxa2_GATA6_GATA6, self.FOXA2_prot, self.P3_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1_GATA6_GATA6, self.FOXF1_prot, self.Pfoxf1_GATA6_GATA6_FOXF1, self.P2_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6_GATA6_FOXF1, self.Pfoxf1_GATA6_GATA6, self.FOXF1_prot, self.P3_act_unbinding[1])
        )
        ## P3 -> P4
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2_GATA6_GATA6_FOXA2, self.FOXA2_prot, self.Pfoxa2_GATA6_GATA6_FOXA2_FOXA2, self.P2_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6_GATA6_FOXA2_FOXA2, self.Pfoxa2_GATA6_GATA6_FOXA2, self.FOXA2_prot, self.P3_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1_GATA6_GATA6_FOXF1, self.FOXF1_prot, self.Pfoxf1_GATA6_GATA6_FOXF1_FOXF1, self.P2_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6_GATA6_FOXF1_FOXF1, self.Pfoxf1_GATA6_GATA6_FOXF1, self.FOXF1_prot, self.P3_act_unbinding[1])
        )
        self.rates = np.zeros(len(self.reactions))
        for rIndex, r in enumerate(self.reactions):
            self.rates[rIndex] = r.GetRate()

    def AddChemical(self, chemical):
        self.chemIndex[chemical] = len(self.chemIndex)

    def GetChemicalIndex(self, chemical):
        return self.chemIndex[chemical]

    def AddReaction(self, reaction):
        self.reactions.append(reaction)

    def GetStateVector(self):
        c = np.zeros(len(self.chemIndex))
        for chem, index in list(self.chemIndex.items()):
            c[index] = chem.amount
        return c

    def SetFromStateVector(self, c):
        for chem, index in list(self.chemIndex.items()):
            chem.amount = c[index]

class DeterministicResponse (GATA6_response_model):

    """
    DeterministicRespone is a deterministic implementation of
    the three-gene repressilator system, with time evolution implemented
    by summing up all reactions as appropriate to form a differential
    equation describing the time rate of change of all chemical
    constituents in the model.
    """

    def dcdt(self, c, t):
        """dcdt(self, c, t) returns the instantaneous time rate of
        change of the DeterministicRepressilator system, given chemical
        concentration state vector c and current time t, for use in
        integration by scipy.integrate.odeint.

        dcdt loops through all reactions defined in the Repressilator
        system, computes the rates of those reactions, and increments
        those elements in a dc_dt array that are affected by the
        reaction under consideration.

        the fully assembled dc_dt array is returned by this method.
        """
        self.SetFromStateVector(c)
        dc_dt = np.zeros(len(self.chemIndex), dtype=float)
        for r in self.reactions:
            rate = r.GetRate()
            for chem, dchem in list(r.stoichiometry.items()):
                dc_dt[self.chemIndex[chem]] += (dchem * rate)
        return dc_dt

    def Run(self, tmax, dt):
        """Run(self, tmax, dt) integrates the DeterministicRepressilator
        for a time tmax, returning the trajectory at time steps as
        specified by dt, by calling scipy.integrate.odeint with the
        self.dcdt method describing the time derivative of the system

        Run should return the time array on which the trajectory is computed,
        along with the trajectory corresponding to those time points.
        """
        eps = 1.0e-06
        ts = np.arange(0, tmax + eps, dt)
        c = self.GetStateVector()
        traj = scipy.integrate.odeint(self.dcdt, c, ts)
        self.SetFromStateVector(traj[-1])

        return ts, traj

def RunDeterministicResponse(GATA6, 
                             activated_transcription_rate,
                             double_activated_transcription_rate,
                             triple_activated_transcription_rate,
                             quad_activated_transcription_rate,
                             inhibited_transcription_rate,
                             P_act_binding,
                             P_inh_binding,
                             P1_act_unbinding,
                             P1_inh_unbinding,
                             P1_act_binding,
                             P2_act_unbinding,
                             P2_act_binding,
                             P3_act_unbinding,
                             P3_act_binding,
                             P4_act_unbinding,
                             FOXA2_prot=0,
                             FOXF1_prot=0,
                             T=10000.,
                             dt=1., 
                             plots=False,
                             plotPromoter=False,
                             unocc_transcription_rate=5.0e-08):

    dr = DeterministicResponse(GATA6=GATA6,
                               FOXA2_prot=FOXA2_prot,
                               FOXF1_prot=FOXF1_prot,
                               unocc_transcription_rate=unocc_transcription_rate,
                               activated_transcription_rate=activated_transcription_rate,
                               double_activated_transcription_rate=double_activated_transcription_rate,
                               triple_activated_transcription_rate=triple_activated_transcription_rate,
                               quad_activated_transcription_rate=quad_activated_transcription_rate,
                               inhibited_transcription_rate=inhibited_transcription_rate,
                               P_act_binding=P_act_binding,
                               P_inh_binding=P_inh_binding,
                               P1_act_unbinding=P1_act_unbinding,
                               P1_inh_unbinding=P1_inh_unbinding,
                               P1_act_binding=P1_act_binding,
                               P2_act_unbinding=P2_act_unbinding,
                               P2_act_binding=P2_act_binding,
                               P3_act_unbinding=P3_act_unbinding,
                               P3_act_binding=P3_act_binding,
                               P4_act_unbinding=P4_act_unbinding)
    dts, dtraj = dr.Run(T, dt)
    curvetypes = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-']
    if plots:
        import pylab
        pylab.figure(1)
        for i in range(len(dr.mRNAs)):
            pylab.plot(dts, dtraj[:, dr.chemIndex[dr.mRNAs[i]]], curvetypes[i])
            pylab.legend(['foxa2', 'foxf1'])
        pylab.figure(2)
        curvetypes = ['r-', 'g-', 'm-', 'b-', 'k-', 'c-']
        for i in range(len(dr.proteins)):
            pylab.plot(dts, dtraj[:, dr.chemIndex[dr.proteins[i]]],curvetypes[i])
            pylab.legend(['FOXA2', 'FOXF1', 'GATA6'])
        if plotPromoter:
            pylab.figure(3)
            for i in range(2):
                promoter_state = (0.99 + 0.01 * i) *\
                                 (dtraj[:, dr.chemIndex[dr.P0[i]]]
                                  + 2. * dtraj[:, dr.chemIndex[dr.P2_active[i]]])
                pylab.plot(dts, promoter_state, curvetypes[i])
        pylab.show()
    return dr, dts, dtraj

from CNV_model import *
import random

class StochasticResponse (GATA6_response_model):

    """
    StochasticRepressilator is a stochastic implementation of
    the three-gene repressilator system, with time evolution implemented
    using Gillespie's Direct Method, as described in detail in Step.
    """

    def ComputeReactionRates(self):
        """ComputeReactionRates computes the current rate for every
        reaction defined in the network, and stores the rates in self.rates."""
        for index, r in enumerate(self.reactions):
            self.rates[index] = r.GetRate()

    def Step(self, dtmax):
        """Step(self, dtmax) implements Gillespie's Direct Simulation Method,
        executing at most one reaction and returning the time increment
        required for that reaction to take place.  If no reaction is executed,
        the specified maximal time increment dtmax is returned.

        (1) all reaction rates are computed
        (2) a total rate for all reactions is found
        (3) a random time is selected, to be drawn from an exponential
            distribution with mean value given by the inverse of the total
            rate, e.g., ran_dtime = -scipy.log(1.-random.random())/total_rate
        (4) if the random time is greater than the time interval under
            consideration (dtmax), then no reaction is executed and dtmax
            is returned
        (5) otherwise, a reaction is chosen at random with relative
            probabilities given by the relative reaction rates;
            this is done by
            (5a) uniformly drawing a random rate from the interval from
                 [0., total rate)
            (5b) identifying which reaction rate interval corresponds to
                 the randomly drawn rate, e.g.,

                 |<-------------------total rate---------------------->|
                 |<----r0----->|<-r1->|<--r2-->|<-----r3----->|<--r4-->|
                 |                                 X                   |

                 Randomly drawn rate X lands in interval r3
        (6) the chosen reaction is executed
        (7) the time at which the reaction is executed is returned
        """
        self.ComputeReactionRates()
        total_rate = sum(self.rates)
        # get exponentially distributed time
        ran_time = -np.log(1. - random.random()) / total_rate
        if ran_time > dtmax:
            return dtmax
        # get uniformly drawn rate in interval defined by total_rate
        ran_rate = total_rate * random.random()
        # find interval corresponding to random rate
        reac_index = len(self.rates) - sum(np.cumsum(self.rates) > ran_rate)
        reaction = self.reactions[reac_index]
        # execute specified reaction
        for chem, dchem in list(reaction.stoichiometry.items()):
            chem.amount += dchem
        # return time at which reaction takes place
        return ran_time

    def Run(self, T, delta_t=0.0):
        """Run(self, T, delta_t) runs the StochasticRepressilator for
        a specified time interval T, returning the trajectory at
        specified time intervals delta_t (or, if delta_t == 0.0, after
        every reaction)
        """
        tfinal = self.t + T
        if delta_t == 0.:
            ts = [self.t]
            trajectory = [self.GetStateVector()]
            while self.t < tfinal:
                dt = self.Step(tfinal - self.t)
                self.t += dt
                ts.append(self.t)
                trajectory.append(self.GetStateVector())
            return np.array(ts), np.array(trajectory)
        else:
            eps = 1.0e-06
            ts = np.arange(0., T + eps, delta_t)
            trajectory = np.zeros((len(ts), len(self.chemIndex)),
                                     float)
            trajectory[0] = self.GetStateVector()
            tindex = 0
            while self.t < tfinal:
                dt = self.Step(ts[tindex + 1] - self.t)
                self.t += dt
                if self.t >= ts[tindex + 1]:
                    tindex += 1
                    for chem, cindex in list(self.chemIndex.items()):
                        trajectory[tindex][cindex] = chem.amount
            return ts, trajectory

def RunStochasticResponse(GATA6, 
                             activated_transcription_rate,
                             double_activated_transcription_rate,
                             triple_activated_transcription_rate,
                             quad_activated_transcription_rate,
                             inhibited_transcription_rate,
                             P_act_binding,
                             P_inh_binding,
                             P1_act_unbinding,
                             P1_inh_unbinding,
                             P1_act_binding,
                             P2_act_unbinding,
                             P2_act_binding,
                             P3_act_unbinding,
                             P3_act_binding,
                             P4_act_unbinding,
                             FOXA2_prot=0,
                             FOXF1_prot=0,
                             T=10000.,
                             dt=1., 
                             plots=False,
                             plotPromoter=False,
                             unocc_transcription_rate=5.0e-08):
    """RunStochasticRepressilator(tmax, dt, plots=False, plotPromoter=False)
    creates and runs a StochasticRepressilator for the specified time
    interval T, returning the trajectory in time increments dt,
    optionally using pylab to make plots of mRNA,
    protein, and promoter amounts along the trajectory.
    """
    sr = StochasticResponse(GATA6=GATA6,
                               unocc_transcription_rate=unocc_transcription_rate,
                               activated_transcription_rate=activated_transcription_rate,
                               double_activated_transcription_rate=double_activated_transcription_rate,
                               triple_activated_transcription_rate=triple_activated_transcription_rate,
                               quad_activated_transcription_rate=quad_activated_transcription_rate,
                               inhibited_transcription_rate=inhibited_transcription_rate,
                               P_act_binding=P_act_binding,
                               P_inh_binding=P_inh_binding,
                               P1_act_unbinding=P1_act_unbinding,
                               P1_inh_unbinding=P1_inh_unbinding,
                               P1_act_binding=P1_act_binding,
                               P2_act_unbinding=P2_act_unbinding,
                               P2_act_binding=P2_act_binding,
                               P3_act_unbinding=P3_act_unbinding,
                               P3_act_binding=P3_act_binding,
                               P4_act_unbinding=P4_act_unbinding,
                               FOXA2_prot=FOXA2_prot,
                               FOXF1_prot=FOXF1_prot)
    sts, straj = sr.Run(T, dt)
    curvetypes = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-']
    if plots:
        import pylab
        pylab.figure(1)
        for i in range(len(sr.mRNAs)):
            pylab.plot(sts, straj[:, sr.chemIndex[sr.mRNAs[i]]], curvetypes[i])
            pylab.legend(['foxa2', 'foxf1'])
            pylab.title('mRNA')
        pylab.figure(2)
        curvetypes = ['r-', 'g-', 'm-', 'b-', 'k-', 'c-']
        for i in range(len(sr.proteins)):
            pylab.plot(sts, straj[:, sr.chemIndex[sr.proteins[i]]],curvetypes[i])
            pylab.legend(['FOXA2', 'FOXF1', 'GATA6'])
            pylab.title('Protein')
        if plotPromoter:
            pylab.figure(3)
            for i in range(2):
                promoter_state = (0.99 + 0.01 * i) *\
                                 (straj[:, sr.chemIndex[sr.P0[i]]]
                                  + 2. * straj[:, sr.chemIndex[sr.P2_active[i]]])
                pylab.plot(sts, promoter_state, curvetypes[i])
                pylab.title('Promoter States')
        pylab.show()
    return sr, sts, straj