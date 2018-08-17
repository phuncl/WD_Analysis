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
6:  2,
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
        self.phase = phase
        """ as relative abundances are counted vs silicon,
        this nuber abundance is that * number of Si inferred by whatever calculation
        """
    
    def add_elmnt(self, z, n, e):
        self.composition[z] = [n, e]
        
        
    def add_elmntsamp(self, z, n):
        """Contains Z/Si vals scaled by phase of accretion correction"""
        self.comp[z] = n
    
    
    def cal(self, ):
        self.omc = np.copy(self.comp[8])
        self.Of = {}
        self.loerrs, self.hierrs = [], []
        runo = 1
        runerlo = []
        runerhi = []
        
        runznum = np.zeros(100000,)
        self.oxyfrac = {}
        
        for x in [m for m in metal_oxides if m in [d.z for d in detections]]:
            np.random.shuffle(self.comp[8])
            np.random.shuffle(self.comp[x])
            zofracsamp = self.comp[x]/self.comp[8] # thus the median and error for each metal will include oxygen error
            self.omc -= zofracsamp
            zofracsamp = np.sort(zofracsamp)
            # calculate median, -err, +err
            zmed, zlo, zhi = np.median(zofracsamp), np.median(zofracsamp)-zofracsamp[len(zofracsamp)//6], zofracsamp[-1*len(zofracsamp)//6]-np.median(zofracsamp)
            self.Of[x] = [zmed, zlo, zhi]
            runo -= zmed # for direct calculation
            runerlo.append(zlo)
            runerhi.append(zhi)
            
            runznum += self.comp[x]*Si.nabund*metal_oxides[x] # want to add this UNSORTED
            zonum = (self.comp[x]*Si.nabund*metal_oxides[x]).sort() # no. O atoms taken by this metal
            zonum_med = np.median(zonum)
            zonum_lo, zonum_hi = zonum_med-zonum[166667], zonum[833333]-zonum_med
            self.oxyfrac[x] = [zonum_med, zonum_min, zonum_man]
            
        np.sort(self.omc)
        self.MC_ostats = [np.median(self.omc)/np.median(self.comp[8]),
                            (np.median(self.omc)-self.omc[166667])/np.median(self.comp[8]), 
                            (self.omc[833333]-np.median(self.omc))/np.median(self.comp[8])]
        self.dir_ostats = [runo, np.sqrt(np.sum(np.square(runerlo))), np.sqrt(np.sum(np.square(runerhi)))]
        
        self.znum_ofrac = runznum/(self.comp[8]*Si.nabund) # divide out range of allowed
        np.sort(self.znum_ofrac)
        self.zresults = [np.median(self.znum_ofrac),
                            np.median(self.znum_ofrac)-self.znum_ofrac[133333],
                            self.znum_ofrac[866667]-np.median(self.znum_ofrac)]
        self.ofractot = np.sum([self.oxyfrac[k][0] for k in self.oxyfrac])
        self.ofracminus = np.sqrt(np.mean(np.square([self.oxyfrac[k][1] for k in self.oxyfrac])))
        self.ofracplus = np.sqrt(np.mean(np.square([self.oxyfrac[k][2] for k in self.oxyfrac])))
        
        
        print('From MC: O xs fraction = {:.3f} + {:.4f} - {:.4f}'.format(self.MC_ostats[0], self.MC_ostats[2], self.MC_ostats[1]))
        print('From direct: O xs fraction = {:.3f} + {:.4f} - {:.4f}'.format(self.dir_ostats[0], self.dir_ostats[2], self.dir_ostats[1]))
        
        print('From numeric MC: O xs fraction = {:.3f} + {:.4f} - {:.4f}'.format(self.dir_ostats[0], self.dir_ostats[2], self.dir_ostats[1]))
        print('From numeric direct: O xs fraction = {:.3f} + {:.4f} - {:.4f}'.format(self.dir_ostats[0], self.dir_ostats[2], self.dir_ostats[1]))
    
    def calco(self, ):
        self.mc_o = self.comp[8]
        self.O = np.mean(self.mc_o)
        self.Ofrac = {}
        self.Oxs_sample = self.comp[8]
        np.random.shuffle(self.Oxs_sample)
        for x in [m for m in metal_oxides if m in [d.z for d in detections]]:
            np.random.shuffle(self.comp[x])
            np.random.shuffle(self.mc_o)
            zo = np.sort(self.comp[x]/self.O)
            self.Ofrac[x] = (np.mean(zo), np.mean(zo)-zo[166667], zo[833333]-np.mean(zo))
            np.random.shuffle(self.comp[x])
            self.Oxs_sample -= self.comp[x]
        # present fractional numbers
        self.Oxs_sample.sort()
        self.Oexcess = np.mean(self.Oxs_sample)/self.O
        self.err_Oexcess = [self.Oxs_sample[166667]/self.O, self.Oxs_sample[833333]/self.O] # estimate from percentiles
        #self.e2_Oexcess = [np.sqrt(np.sum(np.square(self.loerrs))), np.sqrt(np.sum(np.square(self.hierrs)))] #estimate from RMS sum
        print('Fraction of O in excess in {} phase = {:.3f} + {:.4f} - {:.4f}'.format(self.phase, self.Oexcess, self.err_Oexcess[0], self.err_Oexcess[1]))
        #print('Fraction of O in excess in {} phase = {:.3f} + {:.4f} - {:.4f}'.format(self.phase, self.Oexcess, self.e2_Oexcess[0], self.e2_Oexcess[1]))
        
    
    def parent_comp(self,):
        self.nOxs = self.composition[8][0]
        for x in [m for m in metal_oxides if m in [d.z for d in detections]]:
            self.nOxs -= self.composition[x][0]*metal_oxides[x]
        if self.nOxs < 0:
            self.nOxs = 0
        self.h2omass = self.nOxs * AMU * 18
        self.h2omassfrac = self.h2omass/minimum_mass
        print('Water mass fraction of {:.2f} ({:.2e} g)'.format(self.h2omassfrac, self.h2omass))
        """
        self.nOxssbase = np.log10([self.composition[8][0]]*10000)
        self.nOxss = np.log10([self.composition[8][0]]*10000)
        for x in [m for m in metal_oxides if m in [d.z for d in detections]]:
            self.nOxss = np.log10(np.power(10, self.nOxss) - np.power(10, np.log10(self.composition[x][0]), np.log10(self.composition[x][1]), 10000))
            #self.nOxss = np.log10(np.power(10, self.nOxss) - np.random.lognormal(), 10000))
        self.nOxs, self.err_nOxs = 10**np.mean(self.nOxss), np.std(self.nOxss)
        """

def plot_O(ax, comp, y):
    """"""
    o = comp[8][0]
    run_tot = 0
    ax.barh(y, comp[26][0]/o, color='red', label='Fe')
    run_tot += comp[26][0]/o
    ax.barh(y, comp[14][0]*2/o, left=run_tot, color='orange', label='Si')
    run_tot += comp[14][0]*2/o
    ax.barh(y, comp[12][0]/o, left=run_tot, color='pink', label='Mg')
    run_tot += comp[12][0]/o
    ax.barh(y, comp[13][0]*1.5/o, left=run_tot, color='purple', label='Al')
    run_tot += comp[13][0]*1.5/o
    ax.barh(y, comp[20][0]/o, left=run_tot, color='green', label='Ca')
    run_tot += comp[20][0]/o
    ax.barh(y, comp[28][0]/o, left=run_tot, color='brown', label='Ni')
    run_tot += comp[28][0]/o
    ax.barh(y, comp[6][0]*2/o, left=run_tot, color='yellow', label='C')
    run_tot += comp[6][0]*2/o

    if run_tot < 1:
        ax.barh(y, 1-run_tot, left=run_tot, color='dodgerblue', label='O excess')
    return


def ep(x, lo, hi, y):
    xlo = x-lo
    xhi = hi-x
    return([xlo,y], [xhi,y])


def oxs_plot(ax, Ofr, y):
    """"""
    rt = 0
    ax.barh(y, Ofr[26][0], xerr=[[Ofr[26][1]],[Ofr[26][2]]], left=rt, color='red', label='Fe')
    rt += Ofr[26][0]
    ax.barh(y, Ofr[14][0]*2, xerr=[[Ofr[14][1]],[Ofr[14][2]]], left=rt, color='orange', label='Si')
    rt += Ofr[14][0]*2
    ax.barh(y, Ofr[12][0], xerr=[[Ofr[12][1]],[Ofr[12][2]]], left=rt, color='pink', label='Mg')
    rt += Ofr[12][0]
    #ax.barh(y, Ofr[13][0], xerr=[[Ofr[13][1]],[Ofr[13][2]]], left=rt, color='purple', label='Al')
    #rt += Ofr[13][0]
    #ax.barh(y, Ofr[20][0], xerr=[[Ofr[20][1]],[Ofr[20][2]]], left=rt, color='green', label='Ca')
    #rt += Ofr[20][0]
    #ax.barh(y, Ofr[28][0], left=rt, color='brown', label='Ni')
    #rt += Ofr[28][0]
    alcani = Ofr[13][0]*1.5+Ofr[20][0]+Ofr[28][0]
    ax.barh(y, alcani, xerr = [[np.sqrt(np.mean(Ofr[13][1]**2+Ofr[20][1]**2+Ofr[28][1]**2))],
                                [np.sqrt(np.mean(Ofr[13][2]**2+Ofr[20][2]**2+Ofr[28][2]**2))]],
                                left=rt, color='green', label='Al, Ca, Ni')
    rt += alcani
    ax.barh(y, Ofr[6][0], xerr=[[Ofr[6][1]],[Ofr[6][2]]], left=rt, color='yellow', label='C')
    rt += Ofr[6][0]

    if rt < 1:
        ax.barh(y, 1-rt, left=rt, color='dodgerblue', label='O excess')
    return


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
steady_pb = pb()
late1350_pb = pb()

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
    
    early_err = [[early_lo/BE[d.z]], [early_hi/BE[d.z]]]
    a.errorbar(dcount-0.05, early_ZSi/BE[d.z], yerr=early_err, ms=4, fmt='go')
    early_pb.add_elmntsamp(d.z, early_samp)
    early_pb.add_elmnt(d.z, early_ZSi*Si.nabund, early_hi*Si.nabund)
    # error bars are same size in all phases - could just show on one point type?
    #print('{} :  {:.2e} + {:.2e} - {:.2e}'.format(d.z, early_ZSi, early_hi, early_lo))

    #steady_ZSi, steady_lo, steady_hi, steady_samp = d.nab/Si.nabund * Si.tz/d.tz
    #steady_err = [[steady_lo/BE[d.z]],[steady_lo/BE[d.z]]]
    #steady_pb.add_elmnt(d.z, steady_ZSi * Si.nabund)
    #a.scatter(dcount, steady_ZSi/BE[d.z], 8, c='b', marker='s')
    #a.errorbar(dcount, steady_ZSi/BE[d.z], yerr=steady_err, ms=4, fmt='rs')
    
    #late1350_ZSi, late1350_lo, late_1350_hi, late_samp = d.nab/Si.nabund * np.exp(1350*(1/d.tz - 1/Si.tz))
    #late1350_pb.add_elmnt(d.z, late1350_ZSi*Si.nabund)
    #a.scatter(dcount+.05, late1350_ZSi/BE[d.z], 10, c='hotpink', marker='*')

plt.legend()
plt.xticks(range(0, len([d.el for d in detections])), [d.el for d in detections])
plt.show(block=False)


early_pb.parent_comp()
#steady_pb.parent_comp()
#late1350_pb.parent_comp()

plt.figure()
a = plt.gca()
a.set_xlabel('Oxygen source')
a.set_ylabel('Accretion Phase')

plot_O(a, early_pb.composition, 0)
#plot_O(a, steady_pb.composition, 1)
#plot_O(a, late1350_pb.composition, 2)
plt.axvline(1, color='grey')
plt.yticks(range(0, 3), ['Early\nPhase', 'Steady\nState', 'Late\nPhase\n(1350 yr)'])
a.set_xlim(0, 1.4)
xvals=a.get_xticks()
a.set_xticklabels(['{:.0%}'.format(x) for x in xvals if x <=1])
handles, labels = a.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show(block=False)

""""""
early_pb.calco()

plt.figure()
a = plt.gca()
a.set_xlabel('Oxygen source')
a.set_ylabel('Accretion Phase')

oxs_plot(a, early_pb.Ofrac, 0)
#plot_O(a, steady_pb.composition, 1)
#plot_O(a, late1350_pb.composition, 2)
plt.axvline(1, color='grey')
plt.yticks(range(0, 3), ['Early\nPhase', 'Steady\nState', 'Late\nPhase\n(1350 yr)'])
a.set_xlim(0, 1.4)
xvals=a.get_xticks()
a.set_xticklabels(['{:.0%}'.format(x) for x in xvals if x <=1])
handles, labels = a.get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.show(block=False)



print('New Method:')
early_pb.cal()
