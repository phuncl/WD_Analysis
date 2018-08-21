""""""

import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict

#from matplotlib import rc
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
#rc('text', usetex=True)

####    DEFINITIONS

# physical constants
AMU = 1.67377e-24
Msun = 1.989e33

elmnt = {
1: 1.00,
2: 4.00,
6: 12.01,
7: 14.01,
8: 16.00,
11: 22.99,
12: 24.31,
13: 26.98,
14: 28.09,
15: 30.97,
16: 32.06,
20: 40.08,
21: 44.96,
22: 47.87,
23: 50.94,
24: 52.00,
25: 54.94,
26: 55.85,
27: 58.93,
28: 58.69,
}

# stellar parameters
prlx = 0.01017
errprlx = 0.00008
errfrac = errprlx/prlx
d = 1/prlx
errd=d*errfrac

# q, Mwd, t_sink - Koester 2018 priv. comm.
q = 10**(-10.3288) # mass fraction of convection zone
Mwd = 0.673 * Msun
Mcvz = q * Mwd

# SOLAR; number z atoms per atom Si - calculated from data in Lodders 2010, table 3, recommended A(X) 
SOLAR = {
1:  29309.,
6:  7.1945,
7:  2.1232,
8:  15.740,
11: 0.057148,
12: 1.0162,
13: 0.084528,
14: 1.,
15: 0.082604,
16: 0.42364,
20: 0.059841,
21: 3.4435e-5,
22: 0.0024946,
23: 0.00028642,
24: 0.013092,
25: 0.0092683,
26: 0.84528,
27: 0.023281,
28: 0.048641,
}

# CI; number z atoms per atom Si  - Lodders 2010, table 2
CI = {
1:  5.13,
6:  0.76,
7:  0.0553,
8:  7.63,
11: 0.057,
12: 1.03,
13: 0.0827,
14: 1.,
15: 0.00819,
16: 0.438,
20: 0.0604,
21: 0.0000344,
22: 0.002470,
23: 0.00028,
24: 0.0134,
25: 0.00922,
26: 0.87,
27: 0.00225,
28: 0.0483,
}

# BE; number z atoms per atom Si - calculated from Allegre 2001, table 4
BE = {
1:  1e-20,
6:  0.0099415,
7:  7.4269e-6,
8:  1.8968,
11: 0.010936,
12: 0.92398,
13: 0.088129,
14: 1.,
15: 0.0040351,
16: 0.026901,
20: 0.094737,
21: 5.9649e-5,
22: 0.0044678,
23: 0.00054386,
24: 0.024795,
25: 0.0081287,
26: 1.6842,
27: 0.0047018,
28: 0.098830,
}

# metal oxide forming elements and their ratio of atoms to oxygen
metal_oxides = {
#6:  2,
11: 0.5,
12: 1,
13: 1.5,
14: 2,
20: 1,
26: 1,
28: 1,
}

class metal:
    """Calculate atmospheric mass for each element.
    Make useable for calculating relative abundances
    in parent body material at different phases
    
    Inputs:
    el = element symbol e.g. Fe
    Z
    A
    measured abundance
    abundance error
    diffusion timescale
    """
    def __init__(self, el, z, abund, errabund, tz, ):
        # input quantities
        self.el = el
        self.z = z
        self.abund = abund
        self.err_abund = errabund
        self.tz = tz
        
        self.nab = sample_log10normal(self.abund, self.err_abund)
        
        upper = abund+errabund
        lower = abund-errabund
        
        # calculated quantities
        self.nabund = 10**abund * nHe # numerical abundance ratio wrt He
        self.err_nabund = 2.303 * self.nabund * self.err_abund # error
        
        self.mass = self.nabund * elmnt[self.z] * AMU # mass of metal in convection zone
        self.err_mass = self.err_nabund/self.nabund * Mcvz
        
        self.upper_nabund = 10**upper * nHe
        self.lower_nabund = 10**lower * nHe


class pb:
    """
    Hand compositions this to do parent body calculations
    e.g. oxygen excess
    """
    def __init__(self, phase=''):
        self.composition = {} # add NUMBERS of atoms to this
        self.comp = {} # add ABUNDANCES of atoms/Si to this
        self.numbers = {}
        self.phase = phase
        """ as relative abundances are counted vs silicon,
        this nuber abundance is that * number of Si inferred by whatever calculation
        """
    
    def number(self, a, b):
        self.numbers[a] = b
    
    
    def dothething(self, ):
        """Numbers here for heavy metals are less than actually calculated """
        o_samples = np.copy(self.numbers[8])
        np.random.shuffle(o_samples)
        on = np.median(o_samples)
        z_total = np.zeros(len(o_samples))
        self.zonumbers = {}
        for x in [m for m in metal_oxides if m in [d.z for d in detections]]:
            z_total += self.numbers[x]*metal_oxides[x]
            np.random.shuffle(z_total)
            mmed = np.median(self.numbers[x])*metal_oxides[x]/on
            self.zonumbers[x] = [mmed,
                                [[mmed - self.numbers[x][166667]*metal_oxides[x]/on],
                                [self.numbers[x][833333]*metal_oxides[x]/on - mmed]]]
        self.zz = np.copy(z_total)
        oxs_samples = (o_samples - z_total)/on # number of unexplained o atoms - can be more than 1 if sampled O > median O
        self.oxs_lineup = np.sort(oxs_samples)
        """
        plt.figure()
        a = plt.gca()
        a.hist(oxs_samples, 500)
        a.axvline(np.median(oxs_samples), color='red')
        #a.axvline(np.mean(oxs_samples), color='orange')
        a.axvline(self.oxs_lineup[166667], color='green')
        a.axvline(self.oxs_lineup[833333], color='green')
        plt.show(block=False)
        """
        print('O excess = {:.3f} + {:.4f} - {:.4f}'.format(np.median(oxs_samples), 
                                                self.oxs_lineup[833333]-np.median(oxs_samples),
                                                (np.median(oxs_samples)-self.oxs_lineup[166667])))
        
        # calculate a water mass fraction in parent body if Oxs > 0
        self.zonumbers[8] = [np.median(oxs_samples),
                [(np.median(oxs_samples)-self.oxs_lineup[166667]), self.oxs_lineup[833333]-np.median(oxs_samples)]]


def plot_zo(ax, y, dat):
    """"""
    rt = 0
    ax.barh(y, dat[26][0], xerr=dat[26][1], left=rt, color='firebrick', label='Fe', capsize=5)
    rt += dat[26][0]
    ax.barh(y, dat[14][0], xerr=dat[14][1], left=rt, color='coral', label='Si', capsize=5)
    rt += dat[14][0]
    ax.barh(y, dat[12][0], xerr=dat[12][1], left=rt, color='gold', label='Mg', capsize=5)
    rt += dat[12][0]
    #ax.barh(y, dat[20][0], xerr=dat[20][1], left=rt, color='lightgreen', label='Ca')
    #rt += dat[20][0]
    #ax.barh(y, dat[13][0], xerr=dat[13][1], left=rt, color='pink', label='Al')
    #rt += dat[13][0]
    #ax.barh(y, dat[28][0], xerr=dat[28][1], left=rt, color='orchid', label='Ni')
    #rt += dat[28][0]
    others = dat[20][0] + dat[13][0] + dat[28][0]
    othererrlo = np.sqrt(dat[20][1][0][0]**2 + dat[13][1][0][0]**2 + dat[28][1][0][0]**2)
    othererrhi = np.sqrt(dat[20][1][1][0]**2 + dat[13][1][1][0]**2 + dat[28][1][1][0]**2)
    ax.barh(y, others, xerr=[[othererrlo],[othererrhi]], left=rt, color='forestgreen', label='Al, Ca, Ni', capsize=5)
    rt += others
    if dat[8][0] > 0:
        #ax.barh(y, dat[8][0], left=rt, xerr=dat[8][1], color='dodgerblue', label='O excess', ecolor='dimgrey')
        #ax.barh(y, dat[8][0], left=rt, color='navy', label='O excess',)
        ax.barh(y, 1-rt, left=rt, color='royalblue', label='O excess',)
        ax.barh(y, dat[8][1][0]+dat[8][1][1], left=1-dat[8][1][0], color='silver', height=0.1)
    else:
        ax.barh(y, dat[8][1][0]+dat[8][1][1], left=rt-dat[8][1][0], color='silver', height=0.1)
    # this doesn't quite add up to 1 - related to median as value?


def sample_log10normal(x, e):
    """x = exponent, e = error in x
    returns n=10^x, and nhi=upper and nlo=lower 1 sigma bounds"""
    samp_x = np.random.normal(x, e, 10**6)
    samp_n = np.sort(np.power(10, samp_x)) * nHe
    n = np.median(samp_n)
    # inidicies bracketing middle 67%
    nlo = n - samp_n[166667]
    nhi = samp_n[833333] - n
    """
    print(nlo, n, nhi)
    plt.figure()
    a=plt.gca()
    a.hist(samp_n, 100)
    a.axvline(n-nlo, color='k')
    a.axvline(n, color='r')
    a.axvline(n+nhi, color='k')
    plt.title('{}+/-{}'.format(x, e))
    plt.show()
    """
    return n, nlo, nhi, samp_n
    
    

####    MAIN

# convection zone is *evenly mixed* He and H, with only trace metals
# so mass is contributed by He and H

abund_H = -1.1
si_accrate = 7.7049E+07

# mass of one unit of the wd atmosphere 
# i.e. mass of 1 He plus appropriate fraction of one atom H
mHeH = AMU * (4 + 1*np.power(10, abund_H))
mHeH_upp = AMU* (4 + 1*np.power(10, abund_H+0.3))
mHeH_low = AMU* (4 + 1*np.power(10, abund_H-0.3))

# calculate nHe
nHe = Mcvz / (mHeH)
nHe_upp = Mcvz / (mHeH_low)
nHe_low = Mcvz / (mHeH_upp)

# calculate nH from nHe
nH = nHe * np.power(10, abund_H)
nH_upp = nHe * np.power(10, abund_H+0.3)
nH_low = nHe * np.power(10, abund_H-0.3)

# metals
C = metal('C', 6, -6.0, 0.1, 683.82)
O = metal('O', 8, -4.7, 0.1, 770.92)
Mg = metal('Mg', 12, -5.5, 0.1, 567.62)
Al = metal('Al', 13, -6.5, 0.1, 489.34)
Si = metal('Si', 14, -5.6, 0.1, 435.27)
P = metal('P', 15, -7.0, 0.1, 351.12)
S = metal('S', 16, -6.6, 0.1, 393.88)
Ca = metal('Ca', 20, -6.5, 0.1, 436.25)
Fe = metal('Fe', 26, -5.6, 0.1, 285.95)
Ni = metal('Ni', 28, -6.7, 0.1, 270.55)

detections = [C, O, Mg, Al, Si, P, S, Ca, Fe, Ni] # should be SORTED by Z

minimum_mass = np.sum([d.nabund*elmnt[d.z]*AMU for d in detections])
print('Minimum Mass of parent body = {:.3e} g'.format(minimum_mass))

# parent body calculations

early_pb = pb('early')
steady_pb = pb('steady')
late1350_pb = pb('Late')

plt.figure()
a = plt.gca()
a.set_xlabel('Element')
a.semilogy()
a.set_ylabel('[Z/Si] / [Z/Si]_BulkEarth')

e_data = np.asarray([[BE[e]/BE[14], CI[e]/CI[14], SOLAR[e]/SOLAR[14]] for e in CI if e in [d.z for d in detections]])
plt.plot(range(len(detections)), e_data[:,1]/e_data[:,0], c='orange', ls='--', label='CI Chond.')
plt.plot(range(len(detections)), e_data[:,2]/e_data[:,0], c='red', ls='--', label='Solar')
a.set_xlim(auto=False)
a.set_ylim(auto=False)
plt.axhline(1, c='grey', ls='--', label='Bulk Earth')

for dcount, d in enumerate(detections):
    # get number abundance normalised by Si for early phase, steady state - plot on figure
    
    early_ZSi, early_lo, early_hi = d.nab[0:3]/Si.nabund
    early_samp = d.nab[3]/Si.nabund
    early_pb.number(d.z, early_samp*Si.nabund)
    early_err = [[early_lo/BE[d.z]], [early_hi/BE[d.z]]]
    a.errorbar(dcount-0.05, early_ZSi/BE[d.z], yerr=early_err, ms=4, fmt='go', label='Early Phase')
    # error bars are same size in all phases - could just show on one point type?

    steady_ZSi, steady_lo, steady_hi = d.nab[0:3]/Si.nabund * Si.tz/d.tz
    steady_samp = d.nab[3]/Si.nabund * Si.tz/d.tz
    steady_pb.number(d.z, steady_samp*Si.nabund)
    steady_err = [[steady_lo/BE[d.z]],[steady_hi/BE[d.z]]]
    #a.scatter(dcount, steady_ZSi/BE[d.z], 8, c='b', marker='s')
    a.errorbar(dcount, steady_ZSi/BE[d.z], yerr=steady_err, ms=4, fmt='rs', label='Steady Phase')
    
    late1350_ZSi, late1350_lo, late1350_hi = d.nab[:3]/Si.nabund * np.exp(1350*(1/d.tz - 1/Si.tz))
    late1350_samp = d.nab[3]/Si.nabund * np.exp(1350*(1/d.tz - 1/Si.tz))
    late1350_pb.number(d.z, late1350_samp*Si.nabund)
    late_err = [[late1350_lo/BE[d.z]],[late1350_hi/BE[d.z]]]
    #a.scatter(dcount+.05, late1350_ZSi/BE[d.z], 10, c='hotpink', marker='*')
    a.errorbar(dcount+0.05, late1350_ZSi/BE[d.z], yerr=late_err, ms=4, color='navy', fmt='*', label='Late Phase')

handles, labels = a.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.xticks(range(0, len([d.el for d in detections])), [d.el for d in detections])
plt.show(block=False)

early_pb.dothething()
steady_pb.dothething()
late1350_pb.dothething()

plt.figure()
a = plt.gca()
a.set_xlabel('Oxygen source')
a.set_ylabel('Accretion Phase')
xvals=a.get_xticks()
a.set_xticks(np.arange(0,6.,0.5))
a.set_xticklabels(['{:.0%}'.format(x) for x in np.arange(0,6.,0.5) if x <=1])

plot_zo(a, 1, early_pb.zonumbers)
plot_zo(a, 2, steady_pb.zonumbers)
plot_zo(a, 3, late1350_pb.zonumbers)

a.axvline(1, color='grey', ls='--')
handles, labels = a.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show(block=False)


