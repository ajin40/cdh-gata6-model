from CNV_model import *
import random

class StochasticCNV (CNV_model):

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

def RunStochasticCNV(dox, copy_number=2.0, T=100., dt=1., plots=False, plotPromoter=False):
    """RunStochasticRepressilator(tmax, dt, plots=False, plotPromoter=False)
    creates and runs a StochasticRepressilator for the specified time
    interval T, returning the trajectory in time increments dt,
    optionally using pylab to make plots of mRNA,
    protein, and promoter amounts along the trajectory.
    """
    sr = StochasticCNV(dox, copy_number)
    sts, straj = sr.Run(T, dt)
    curvetypes = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-']
    if plots:
        import pylab
        pylab.figure(1)
        pylab.title("RNAs")
        for i in range(len(sr.mRNAs)):
            pylab.plot(sts, straj[:, sr.chemIndex[sr.mRNAs[i]]], curvetypes[i])
            pylab.legend(['nanog', 'gata6', 'foxa2', 'foxf1'])
        curvetypes = ['r-', 'g-', 'm', 'b-', 'k-', 'c-']
        pylab.figure(2)
        pylab.title("Proteins")
        for i in range(len(sr.proteins)):
            pylab.plot(
                sts,
                straj[
                    :,
                    sr.chemIndex[
                        sr.proteins[i]]],
                curvetypes[i])
            pylab.legend(['NANOG', 'GATA6', 'TETO-dox', 'FOXA2', 'FOXF1', 'TETO'])
        if plotPromoter:
            pylab.figure(3)
            pylab.title("Promoter States")
            for i in range(3):
                promoter_state = (0.99 + 0.01 * i) *\
                                 (straj[:, sr.chemIndex[sr.P1[i]]]
                                  + 2. * straj[:, sr.chemIndex[sr.P2[i]]])
                pylab.plot(sts, promoter_state, curvetypes[i])
        pylab.show()
    return sr, sts, straj