from CNV_model import *
import scipy.integrate

class DeterministicCNV (CNV_model):

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
        for index, r in enumerate(self.reactions):
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

def RunDeterministicCNV(dox, copy_number=2, T=100., dt=1., plots=False,
                                  plotPromoter=False):
    """RunDeterministicRepressilator(tmax, dt, plots=False, plotPromoter=False)
    creates and runs a DeterministicRepressilator for the specified time
    interval T, returning the trajectory in time increments dt,
    optionally using pylab to make plots of mRNA, protein,
    and promoter amounts along the trajectory.
    """
    dr = DeterministicCNV(dox, copy_number)
    dts, dtraj = dr.Run(T, dt)
    curvetypes = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-']
    if plots:
        import pylab
        pylab.figure(1)
        for i in range(len(dr.mRNAs)):
            pylab.plot(dts, dtraj[:, dr.chemIndex[dr.mRNAs[i]]], curvetypes[i])
            pylab.legend(['nanog', 'gata6', 'foxa2', 'foxf1'])
        pylab.figure(2)
        curvetypes = ['r-', 'g-', 'm', 'b-', 'k-', 'c-']
        for i in range(len(dr.proteins)):
            pylab.plot(
                dts,
                dtraj[
                    :,
                    dr.chemIndex[
                        dr.proteins[i]]],
                curvetypes[i])
            pylab.legend(['NANOG', 'GATA6', 'TETO-dox', 'FOXA2', 'FOXF1', 'TETO'])
        if plotPromoter:
            pylab.figure(3)
            for i in range(3):
                promoter_state = (0.99 + 0.01 * i) *\
                                 (dtraj[:, dr.chemIndex[dr.P1[i]]]
                                  + 2. * dtraj[:, dr.chemIndex[dr.P2[i]]])
                pylab.plot(dts, promoter_state, curvetypes[i])
        pylab.show()
    return dr, dts, dtraj