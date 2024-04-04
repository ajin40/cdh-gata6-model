import numpy as np
from Chemical import *

class CNV_model:

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

    For reference
    self.act_transcription_rate = ['pNANOG', 'pGATA6_endo', 'pGATA6_exo', 'pFOXA2_GATA6', 'pFOXF1_GATA6', 'pFOXA2_FOXA2', 'pFOXF1_FOXF1']
    self.double_act_transcription_rate = ['pFOXA2_++', 'pFOXF1++']
    self.inh_transcription_rate = ['pNANOG_GATA6', 'pGATA6_NANOG', 'pFOXA2_FOXF1', 'pFOXF1_FOXA2']
    self.P_act_binding = ['pNANOG_NANOG', 'pGATA6_endogenous_GATA6', 'pGATA6_exo_TETO', 'pFOXA2_FOXA2', 'pFOXF1_FOXF1', 'pFOXA2_GATA6', 'pFOXF1_GATA6'] #P_binding
    self.P_inh_binding = ['pNANOG_GATA6', 'pGATA6_NANOG', 'pFOXA2_FOXF1', 'pFOXF1_FOXA2']
    self.P1_act_unbinding = ['pNANOG_NANOG', 'pGATA6_endogenous_GATA6', 'pGATA6_exo_TETO', 'pFOXA2_FOXA2', 'pFOXF1_FOXF1', 'pFOXA2_GATA6', 'pFOXF1_GATA6'] #P1_unbinding
    self.P1_inh_unbinding = ['pNANOG_GATA6', 'pGATA6_NANOG', 'pFOXA2_FOXF1', 'pFOXF1_FOXA2']
    self.P1_act_binding = ['pFOXA2_FOXA2_GATA6', 'pFOXF1_FOXF1_GATA6', 'pFOXA2_GATA6_FOXA2', 'pFOXF1_GATA6_FOXF1']
    self.P2_act_unbinding = ['pFOXA2_FOXA2_GATA6', 'pFOXF1_FOXF1_GATA6', 'pFOXA2_GATA6_FOXA2', 'pFOXF1_GATA6_FOXF1'] #P2_unbinding

    """

    def __init__(self,
                 dox,
                 copy_number,
                 mRNA_degradation_rate=np.log(2.) / 120.,
                 protein_degradation_rate=np.log(2.) / 600.,
                 translation_rate=0.167,
                 unocc_transcription_rate=5.0e-08,
                 activated_transcription_rate = [0.25, 0.25, 0.5, 5.0e-4, 5.0e-4, 5.0e-6, 5.0e-6],
                 double_activated_transcription_rate = [0.25, 0.25],
                 inhibited_transcription_rate=[5.0e-08, 5.0e-8, 5.0e-8, 5.0e-8],
                 P_act_binding=[1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0],
                 P_inh_binding=[1, 1, 1, 0.5],
                 P1_act_unbinding=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                 P1_inh_unbinding=[1.0, 1.0, 1.0, 1.0],
                 P1_act_binding=[1.0, 1.0, 1.0, 1.0],
                 P2_act_unbinding=[1.0, 1.0, 1.0, 1.0],
                 dox_binding_rate=0.05,
                 dox_unbinding_rate=0.1):
        self.t = 0.
        self.chemIndex = {}
        self.reactions = []

        # Rates
        self.mRNA_degradation_rate = mRNA_degradation_rate
        self.protein_degradation_rate = protein_degradation_rate
        self.translation_rate = translation_rate
        self.unocc_transcription_rate = unocc_transcription_rate
        self.dox_binding_rate = dox_binding_rate
        self.dox_unbinding_rate = dox_unbinding_rate
        self.act_transcription_rate = activated_transcription_rate
        self.double_act_transcription_rate = double_activated_transcription_rate
        self.inh_transcription_rate = inhibited_transcription_rate
        self.P_act_binding = P_act_binding
        self.P_inh_binding = P_inh_binding
        self.P1_act_unbinding = P1_act_unbinding
        self.P1_inh_unbinding = P1_inh_unbinding
        self.P1_act_binding = P1_act_binding
        self.P2_act_unbinding = P2_act_unbinding

        # small molecule
        self.dox_concentration = round(dox * 10 * 6.02 * 5.23 / 444.4) # Cell volume (5.23 * 10^-16) is in meters cubed
        self.dox = Chemical(self.dox_concentration)
        self.AddChemical(self.dox)
        
        # Proteins
        self.NANOG = Chemical(12000.0)
        self.GATA6 = Chemical(0.0)
        self.FOXA2 = Chemical(0.0)
        self.FOXF1 = Chemical(0.0)
        self.TETO = Chemical(20000.0)
        self.TETO_dox = Chemical(0.0)
        
        # mRNAs
        self.nanog = Chemical(80.0)
        self.gata6 = Chemical(0.0)
        self.foxa2 = Chemical(0.0)
        self.foxf1 = Chemical(0.0)
        
        # Promoter States
        # Unbound (Default) each promoter has 2 copies. except exogenous gata6 (The synthetic Circuit)
        self.Pnanog = Chemical(2.0)
        self.Pgata6 = Chemical(2.0)
        self.Pgata6_exo = Chemical(copy_number) ## THIS IS WHERE CNV WOULD BE IMPORTANT
        self.Pfoxa2 = Chemical(2.0)
        self.Pfoxf1 = Chemical(2.0)
        
        # Single bound activated States
        self.Pnanog_NANOG = Chemical(0.0)
        self.Pgata6_GATA6 = Chemical(0.0)
        self.Pgata6_exo_TETO_dox = Chemical(0.0)
        self.Pfoxa2_GATA6 = Chemical(0.0)
        self.Pfoxf1_GATA6 = Chemical(0.0)
        self.Pfoxa2_FOXA2 = Chemical(0.0)
        self.Pfoxf1_FOXF1 = Chemical(0.0)
        
        # Single bound inhibited states
        self.Pnanog_GATA6 = Chemical(0.0)
        self.Pgata6_NANOG = Chemical(0.0)
        self.Pfoxa2_FOXF1 = Chemical(0.0)
        self.Pfoxf1_FOXA2 = Chemical(0.0)
        
        # Double bound activated states
        self.Pfoxa2_GATA6_FOXA2 = Chemical(0.0)
        self.Pfoxf1_GATA6_FOXF1 = Chemical(0.0)
        
        # Double bound inhibited states
        self.Pfoxa2_GATA6_FOXF1 = Chemical(0.0)
        self.Pfoxf1_GATA6_FOXA2 = Chemical(0.0)
        
        # References for indexing
        self.P0_unocc_transcription_ref = [0, 1, 1, 2, 3]
        self.P0_inhib_binding_ref = [1, 0, 3, 2]
        self.P1_active_mRNA_ref = [0, 1, 1, 2, 3, 2, 3]

        self.mRNAs = [self.nanog, self.gata6, self.foxa2, self.foxf1]
        self.proteins = [self.NANOG, self.GATA6, self.TETO_dox, self.FOXA2, self.FOXF1, self.TETO]
        self.P0 = [self.Pnanog, self.Pgata6, self.Pgata6_exo, self.Pfoxa2, self.Pfoxf1]
        self.P1_active = [self.Pnanog_NANOG, self.Pgata6_GATA6, self.Pgata6_exo_TETO_dox, self.Pfoxa2_GATA6, self.Pfoxf1_GATA6, self.Pfoxa2_FOXA2, self.Pfoxf1_FOXF1]

        self.P1_inhib = [self.Pnanog_GATA6, self.Pgata6_NANOG, self.Pfoxa2_FOXF1, self.Pfoxf1_FOXA2]
        self.P2_active = [self.Pfoxa2_GATA6_FOXA2, self.Pfoxf1_GATA6_FOXF1]
        self.P2_inhib = []

        # self.P2_inhib = [self.Pfoxa2_GATA6_FOXF1, self.Pfoxf1_GATA6_FOXA2]
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
        for chem in self.P2_inhib:
            self.AddChemical(chem)
            
        self.AddReaction(
            HeterodimerBindingReaction(self.TETO,
                                      self.dox,
                                      self.TETO_dox,
                                      self.dox_binding_rate))
        
        self.AddReaction(
            HeterodimerUnbindingReaction(self.TETO_dox,
                                      self.TETO,
                                      self.dox,
                                      self.dox_unbinding_rate))
            
        # Adding mRNA reactions
        for i in range(len(self.mRNAs)):
            self.AddReaction(
                DegradationReaction(self.mRNAs[i], self.mRNA_degradation_rate))
            if i > 1:
                self.AddReaction(
                    CatalyzedSynthesisReaction(self.mRNAs[i], self.proteins[i+1],
                                          self.translation_rate))
            else:
                self.AddReaction(
                    CatalyzedSynthesisReaction(self.mRNAs[i], self.proteins[i],
                                            self.translation_rate))
                
        for i in range(len(self.proteins)):
            if i == 2 or i == 5:
                continue
            else:
                self.AddReaction(
                    DegradationReaction(self.proteins[i], self.protein_degradation_rate))
                
        for i in range(len(self.P0)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.P0[i],
                                           self.mRNAs[self.P0_unocc_transcription_ref[i]],
                                           self.unocc_transcription_rate))
        for i in range(len(self.P1_active)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.P1_active[i],
                                           self.mRNAs[self.P1_active_mRNA_ref[i]],
                                           self.act_transcription_rate[i]))
                                           
        for i in range(len(self.P1_inhib)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.P1_inhib[i],
                                           self.mRNAs[i],
                                           self.inh_transcription_rate[i]))
            
        for i in range(len(self.P2_active)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.P2_active[i],
                                           self.mRNAs[i+2],
                                           self.double_act_transcription_rate[i]))
            
        for i in range(len(self.P2_inhib)):
            self.AddReaction(
                CatalyzedSynthesisReaction(self.P2_inhib[i],
                                           self.mRNAs[i+2],
                                           self.inh_transcription_rate[i]))
        
        # Activating P0 reactions 10
        for i in range(len(self.P0)):
            self.AddReaction(
                HeterodimerBindingReaction(self.P0[i],
                                           self.proteins[i],
                                           self.P1_active[i],
                                           self.P_act_binding[i]))
            self.AddReaction(
                HeterodimerUnbindingReaction(self.P1_active[i],
                                             self.P0[i],
                                             self.proteins[i],
                                             self.P1_act_unbinding[i]))
        # Two more P0 -> P1 reactions 4
        for i in range(2):
            ref = [3, 4]
            self.AddReaction(
                HeterodimerBindingReaction(self.P0[ref[i]],
                                           self.proteins[1],
                                           self.P1_active[ref[i]],
                                           self.P_act_binding[len(self.P0) + i]))
            self.AddReaction(
                HeterodimerUnbindingReaction(self.P1_active[ref[i]],
                                             self.P0[ref[i]],
                                             self.proteins[1],
                                             self.P1_act_unbinding[len(self.P0) + i]))
            
        # Inhibiting P0 reactions 8
        for i in range(len(self.P0)-1):
            ref = [0, 1, 3, 4]
            self.AddReaction(
                HeterodimerBindingReaction(self.P0[ref[i]],
                                           self.proteins[self.P0_inhib_binding_ref[i]],
                                           self.P1_inhib[i],
                                           self.P_inh_binding[i]))
            self.AddReaction(
                HeterodimerUnbindingReaction(self.P1_inhib[i],
                                             self.P0[ref[i]],
                                             self.proteins[self.P0_inhib_binding_ref[i]],
                                             self.P1_inh_unbinding[i]))
                
        # Activating P1 -> P2 reactions 8
        for i in range(3, len(self.P1_active)):
            if i > 4:
                self.AddReaction(
                    HeterodimerBindingReaction(self.P1_active[i],
                                            self.proteins[1],
                                            self.P2_active[i-5],
                                            self.P1_act_binding[i-3]))
                self.AddReaction(
                    HeterodimerUnbindingReaction(self.P2_active[i-5],
                                                self.P1_active[i],
                                                self.proteins[1],
                                                self.P2_act_unbinding[i-3]))
            else:
                self.AddReaction(
                    HeterodimerBindingReaction(self.P1_active[i],
                                            self.proteins[i],
                                            self.P2_active[i-3],
                                            self.P1_act_binding[i-3]))
                self.AddReaction(
                    HeterodimerUnbindingReaction(self.P2_active[i-3],
                                                self.P1_active[i],
                                                self.proteins[i],
                                                self.P2_act_unbinding[i-3]))

        self.rates = np.zeros(len(self.reactions), float)
        for rIndex, r in enumerate(self.reactions):
            self.rates[rIndex] = r.GetRate()

    def AddChemical(self, chemical):
        self.chemIndex[chemical] = len(self.chemIndex)

    def GetChemicalIndex(self, chemical):
        return self.chemIndex[chemical]

    def AddReaction(self, reaction):
        self.reactions.append(reaction)

    def GetStateVector(self):
        c = np.zeros(len(self.chemIndex), float)
        for chem, index in list(self.chemIndex.items()):
            c[index] = chem.amount
        return c

    def SetFromStateVector(self, c):
        for chem, index in list(self.chemIndex.items()):
            chem.amount = c[index]
