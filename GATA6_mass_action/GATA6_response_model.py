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
                 P_act_binding,
                 P_inh_binding,
                 P1_act_unbinding,
                 P1_inh_unbinding,
                 P1_act_binding,
                 P2_act_unbinding,
                 mRNA_degradation_rate=np.log(2) / 120.,
                 protein_degradation_rate=np.log(2) / 600.,
                 translation_rate=0.167,
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

        self.act_transcription_rate = activated_transcription_rate
        self.double_act_transcription_rate = double_activated_transcription_rate
        self.inh_transcription_rate = inhibited_transcription_rate
        self.P_act_binding = P_act_binding
        self.P_inh_binding = P_inh_binding
        self.P1_act_unbinding = P1_act_unbinding
        self.P1_inh_unbinding = P1_inh_unbinding
        self.P1_act_binding = P1_act_binding
        self.P2_act_unbinding = P2_act_unbinding
        
        # mRNAs
        self.foxa2_mrna = Chemical(0.0)
        self.foxf1_mrna = Chemical(0.0) 
        
        # Proteins
        self.FOXA2_prot = Chemical(0.0)
        self.FOXF1_prot = Chemical(0.0)
        self.GATA6_prot = Chemical(GATA6)
        # Promoter States
        # Unbound (Default) each promoter has 2 copies. except exogenous gata6 (The synthetic Circuit)
        self.Pfoxa2 = Chemical(2.0)
        self.Pfoxf1 = Chemical(2.0)
        
        # Single bound activated States
        self.Pfoxa2_GATA6 = Chemical(0.0)
        self.Pfoxf1_GATA6 = Chemical(0.0)
        self.Pfoxa2_FOXA2 = Chemical(0.0)
        self.Pfoxf1_FOXF1 = Chemical(0.0)
        
        # Single bound inhibited states
        self.Pfoxa2_FOXF1 = Chemical(0.0)
        self.Pfoxf1_FOXA2 = Chemical(0.0)
        
        # Double bound activated states
        self.Pfoxa2_GATA6_FOXA2 = Chemical(0.0)
        self.Pfoxf1_GATA6_FOXF1 = Chemical(0.0)
        
        # Double bound inhibited states
        self.Pfoxa2_GATA6_FOXF1 = Chemical(0.0)
        self.Pfoxf1_GATA6_FOXA2 = Chemical(0.0)
        
        # References for indexing

        self.mRNAs = [self.foxa2_mrna, self.foxf1_mrna]
        self.proteins = [self.FOXA2_prot, self.FOXF1_prot, self.GATA6_prot]
        self.P0 = [self.Pfoxa2, self.Pfoxf1]
        self.P1_active = [self.Pfoxa2_FOXA2, self.Pfoxf1_FOXF1, self.Pfoxa2_GATA6, self.Pfoxf1_GATA6]
        self.P1_active_mRNA_ref = [0, 1, 0, 1]
        self.P1_inhib = [self.Pfoxa2_FOXF1, self.Pfoxf1_FOXA2]
        self.P2_active = [self.Pfoxa2_GATA6_FOXA2, self.Pfoxf1_GATA6_FOXF1]
        self.P2_inhib = []

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
        # for chem in self.P2_inhib:
        #     self.AddChemical(chem)
            
            
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
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_FOXA2, self.foxa2_mrna, self.act_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_FOXF1, self.foxf1_mrna, self.act_transcription_rate[1]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6, self.foxa2_mrna, self.act_transcription_rate[2]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6, self.foxf1_mrna, self.act_transcription_rate[3]))

        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_GATA6_FOXA2, self.foxa2_mrna, self.double_act_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_GATA6_FOXF1, self.foxf1_mrna, self.double_act_transcription_rate[1]))

        # Inhibited mRNA production
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxa2_FOXF1, self.foxa2_mrna, self.inh_transcription_rate[0]))
        self.AddReaction(CatalyzedSynthesisReaction(self.Pfoxf1_FOXA2, self.foxf1_mrna, self.inh_transcription_rate[1]))

        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2, self.FOXA2_prot, self.Pfoxa2_FOXA2, self.P_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_FOXA2, self.Pfoxa2, self.FOXA2_prot, self.P1_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1, self.FOXF1_prot, self.Pfoxf1_FOXF1, self.P_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_FOXF1, self.Pfoxf1, self.FOXF1_prot, self.P1_act_unbinding[1])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2, self.GATA6_prot, self.Pfoxa2_GATA6, self.P_act_binding[2])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6, self.Pfoxa2, self.GATA6_prot, self.P1_act_unbinding[2])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1, self.GATA6_prot, self.Pfoxf1_GATA6, self.P_act_binding[3])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6, self.Pfoxf1, self.GATA6_prot, self.P1_act_unbinding[3])
        )
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
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2_FOXA2, self.GATA6_prot, self.Pfoxa2_GATA6_FOXA2, self.P1_act_binding[0])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6_FOXA2, self.Pfoxa2_FOXA2, self.GATA6_prot, self.P2_act_unbinding[0])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1_FOXF1, self.GATA6_prot, self.Pfoxf1_GATA6_FOXF1, self.P1_act_binding[1])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6_FOXF1, self.Pfoxf1_FOXF1, self.GATA6_prot, self.P2_act_unbinding[1])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxa2_GATA6, self.FOXA2_prot, self.Pfoxa2_GATA6_FOXA2, self.P1_act_binding[2])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxa2_GATA6_FOXA2, self.Pfoxa2_GATA6, self.FOXA2_prot, self.P2_act_unbinding[2])
        )
        self.AddReaction(
            HeterodimerBindingReaction(self.Pfoxf1_GATA6, self.FOXF1_prot, self.Pfoxf1_GATA6_FOXF1, self.P1_act_binding[3])
        )
        self.AddReaction(
            HeterodimerUnbindingReaction(self.Pfoxf1_GATA6_FOXF1, self.Pfoxf1_GATA6, self.FOXF1_prot, self.P2_act_unbinding[3])
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
    DeterministicRepressilator is a deterministic implementation of
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
                             inhibited_transcription_rate,
                             P_act_binding,
                             P_inh_binding,
                             P1_act_unbinding,
                             P1_inh_unbinding,
                             P1_act_binding,
                             P2_act_unbinding,
                             T=10000.,
                             dt=1., 
                             plots=False,
                             plotPromoter=False,
                             unocc_transcription_rate=5.0e-08):

    dr = DeterministicResponse(GATA6=GATA6,
                               unocc_transcription_rate=unocc_transcription_rate,
                               activated_transcription_rate=activated_transcription_rate,
                               double_activated_transcription_rate=double_activated_transcription_rate,
                               inhibited_transcription_rate=inhibited_transcription_rate,
                               P_act_binding=P_act_binding,
                               P_inh_binding=P_inh_binding,
                               P1_act_unbinding=P1_act_unbinding,
                               P1_inh_unbinding=P1_inh_unbinding,
                               P1_act_binding=P1_act_binding,
                               P2_act_unbinding=P2_act_unbinding)
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