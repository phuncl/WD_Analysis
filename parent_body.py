""""""

import numpy as np
import matplotlib.pyplot as plt
from photometry import reddenings as rd

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


def get_dk(fname, showplot=False):
    with open(fname, 'r') as f:
        d = f.readlines()[52:]
    d2 = [x.strip().split() for x in d] # make list of lists
    d3 = np.asarray([[float(y) for y in x] for x in d2]) # convert to floats
    
    if showplot:
        plt.figure()
        plt.plot(d3[:,0], d3[:,1], c = 'r')
        plt.show()

    return d3


##########################################

#10.17 mas +/- 0.08 mas
prlx = 0.01017
errprlx = 0.00008
errfrac = errprlx/prlx
d = 1/prlx
errd=d*errfrac

# read abundances from file?
#fname = 'models/dab-v5.dk'

# q, Mwd, t_sink from Koester 2018 priv. comm.
q = 10**(-10.3288) # mass fraction of convection zone
Mwd = 0.673 * 1.989e33
AMU = 1.67377e-24
Mcvz = q * Mwd


##########################################
# elements general data
# BE data from Allegre et al 2001
# t_sink from Koester 2018 priv comm
# Z:[name, A, BE mass frac, err, t_sink, CI mass ppm (lodders2003 table3) , Solar (lodders2003 table1)]
# Solar (lodders2003) is numerical abundance per 10^6 Si I.E. cosmochemical abundance
elements = {1:['H', 1, 0, 0, 0, 21015, 2.884e10],
            2:['He', 4, 0, 0, 0, 9.17e-3, 2.288e9],
            6:['C', 12, 1.6e-3, 0.1e-3, 2.8349, 35180, 7.079e6],
            7:['N', 14, 1.27e-6, 1.0e-6, 2.887, 2940, 1.950e6],
            8:['O', 16, 0.32436, 1.0e-4, 2.9354, 458200, 1.413e7],
            #11:['Na', 23, 0.00187, 0.00015, 0, 5060, 5.751e4],
            12:['Mg', 24, 0.158, 0.001, 2.7541, 95870, 1.020e6],
            13:['Al', 27, 0.01507, 0.0001, 2.6896, 8500, 8.410e4],
            14:['Si', 28, 0.171, 0.002, 2.6388, 106500, 1.0e6],
            15:['P', 31, 0.69e-3, 0.01e-3, 2.5455, 920, 8373],
            16:['S', 32, 0.0046, 0.0015, 2.5954, 54100, 4.449e5],
            20:['Ca', 40, 0.0162, 0.0002, 2.6397, 9070, 6.287e4],
            26:['Fe', 56, 0.288, 0.004, 2.4563, 182800, 8.380e5],
            28:['Ni', 59, 0.0169, 0.0003, 2.4322, 10640, 4.780e4]} # CURRENTLY USING FE TIMESCALE FOR NI


# measured abundances from model fitting
# z: [phot abund, err phot abund]
abundances = {1:[-0.75, 0.25],
              6:[-6.0, 0.1],
              7:[-5.8, 0.],
              8:[-4.7, 0.1],
              12:[-5.5, 0.1],
              13:[-6.2, 0.], # upper limit until optical
              14:[-5.6, 0.1],
              15:[-7.0, 0.1],
              16:[-6.6, 0.1],
              20:[-6.3, 0.], # upper limit until optical
              26:[-5.6, 0.1],
              28:[-6.7, 0.1]}

acc_els = []
for k in abundances.keys():
    acc_els.append(elements[k][0])

# nHe in cvz for pure helium atmosphere
nHepure = Mcvz/(4*AMU)#
# nHe, including H, = Mcvz / (MHe + nH/nHe*MH)
nHe = Mcvz / (AMU * (4 + 1*np.power(10, abundances[1][0]))) # nHe*AMU*4 + abund[1][0]*nHe*AMU*1 = Mcvz

# append abundance dictionary with n(Z)/n(He)
for i in abundances:
    logZHe = abundances[i][0]
    errlogZHe = abundances[i][1]
    
    nZHe = 10**logZHe
    errnZHe = 2.303*nZHe*errlogZHe
    
    abundances[i].append(nZHe)
    abundances[i].append(errnZHe)
    
    #if i[1] != -12 and i[1] != 0:
        #abund[int(i[0]/100)] = [i[1], 10**i[1]]
        #ERRORS

##########################################

# calculate abundances of accreted material (ie material in photosphere)
accreted_abs = {}
#accreted_abs_uplim = {} # REDUNDANT
accreted_Mabs = {}
#accreted_Mabs_uplim = {} # REDUNDANT
for z in abundances:
    if z!=2:
        nZ = abundances[z][2] * nHe # number of Zatoms in cvz
        errnZ = abundances[z][3] * nHe
        symb = abundances[z][0]
        
        MZ = nZ * AMU * z # mass in cvz
        errMZ = errnZ * AMU * z
        
        accreted_abs[z] = [nZ, errnZ]
        accreted_Mabs[z] = [MZ, errMZ]
        """ denote upper limits somehow
        if abundances[z][2]:
            accreted_abs[z] = nZ
            accreted_Mabs[z] = MZ
        else:
            accreted_abs_uplims = nZ
            accreted_Mabs_uplim = MZ
        """

# Silicon numbers  -for normalisation
nSi = accreted_abs[14][0]
mSi = accreted_Mabs[14][0]
nSippm = elements[14][5]/(10**6) / 28 # Si ppm by num in CI chonds
Sippm = elements[14][5]/(10**6) # mass Si ppm in CI chondrites
solnSi = elements[14][6] # Si abundance is solar, = 1e6 by definition of cosmochemical scale
solmSi = elements[14][6] * 28 # Si mass abundance in solar 

# calculate numerical/mass abundances for BE and CI composition
BE_ZSi, mBE_ZSi = [], []
CI_ZSi, mCI_ZSi = [], []
sol_ZSi, msol_ZSi = [], []

for z in abundances:
    if z!=2:
        # all this is calculated from literature values
        BE_ZSi.append((elements[z][2] / elements[z][1]) / (elements[14][2] / elements[14][1]))
        #errBE_ZSi.append()
        mBE_ZSi.append(elements[z][2] / elements[14][2])
        
        nppm = elements[z][5]/(10**6) / elements[z][1]
        ppm = elements[z][5]/(10**6)
        CI_ZSi.append(nppm/nSippm)
        mCI_ZSi.append(ppm/Sippm)
        
        sol_ZSi.append(elements[z][6]/solnSi) # solar number abundance relative to Si
        msol_ZSi.append(elements[z][6]*elements[z][1]/solmSi) # solar mass abundance relative to Si


# create abundance profiles relative to BE for CI and sol
BE_ZSi = np.asarray(BE_ZSi)
mBE_ZSi = np.asarray(mBE_ZSi)

CI_ZSi = np.asarray(CI_ZSi)
mCI_ZSi = np.asarray(mCI_ZSi)
CIBE_ZSi = CI_ZSi/BE_ZSi
mCIBE_ZSi = mCI_ZSi/mBE_ZSi

sol_ZSi = np.asarray(sol_ZSi)
msol_ZSi = np.asarray(msol_ZSi)
solBE_ZSi = sol_ZSi/BE_ZSi
msolBE_ZSi = msol_ZSi/mBE_ZSi

# FOLD IN Si ERROR HERE TOO?
acc_absSi = np.asarray([accreted_abs[x][0] for x in abundances.keys()])/nSi / BE_ZSi
# err dz = dx/x + dy/y
#erracc_absSi = np.asarray([accreted_abs[x][1]/accreted_abs[x][0] for x in abundances.keys()]) + accreted_abs[14][1]/accreted_abs[14][0] # include Si - not needed?
erracc_absSi = np.asarray([accreted_abs[x][1] for x in abundances.keys()])/nSi / BE_ZSi # purely error on Z, not Si too

# mark out upper limits
upperlims = np.logical_or(erracc_absSi==0, erracc_absSi==np.inf)
# hold onto upper limits as 40%
erracc_absSi[np.argwhere(np.asarray([accreted_abs[x][1] for x in abundances.keys()])==0)] = 0.4*acc_absSi[np.argwhere(np.asarray([accreted_abs[x][1] for x in abundances.keys()])==0)]

#acc_absSi = np.asarray(list(accreted_abs.values()))/nSi / BE_ZSi
#acc_mabs = np.asarray(list(accreted_Mabs.values()))/mSi / mBE_ZSi


# CALCULATE STEADY STATE ABUNDANCES
""" BROKEN BY ERRORS
mSiSS = accreted_Mabs[14]/np.power(10, elements[14][4]) # MSi / tSi
mOSS = accreted_Mabs[8]/np.power(10, elements[8][4]) # MO/tO

SS_mabs = {} # abundances in steady state relative to Si
SS_absO = {}
for z in abundances:
    if elements[z][4] != 0: # ie if I have a timescale
        # PB abundance relative to Si ~= MZ/MSi * tSi/tZ
        mZSS = accreted_Mabs[z]/np.power(10,elements[z][4]) # MZ/tZ
        # relative mass abundances in PB
        SS_mabs[z] = mZSS/mSiSS # = MZ/MSi * tSi/tZ (Jura Young 2017)
        
        SS_absO[z] = mZSS/mOSS * elements[8][1]/elements[z][1] # = nZ/nO * tO/tZ = num abund/O
        # ie num abundance in pb relative to O
    else:
        SS_mabs[z] = -1
        print('No timescale for {}'.format(elements[z][0]))

acc_mabsSS = np.asarray(list(SS_mabs.values()))/ mBE_ZSi
acc_mabsSS[acc_mabsSS<0] = None
"""


# plot ___ relative to bulk earth
fig = plt.figure()
#fig, [a1, a2] = plt.subplots(2, sharex=True)
plt.title('Parent Body Number Abundances')
a1 = plt.gca()
a1.axhline(1, color='grey', linestyle='--', label='Bulk Earth')
a1.plot(range(len(acc_els)), CIBE_ZSi, label='CI Chondrite', linestyle='--', color='orange')
a1.plot(range(len(acc_els)), solBE_ZSi, label='Solar', linestyle='--', color='red')

#abnd_errs

# show upper limit measurements here
# skip H, not in BE

a1.errorbar(range(1, len(acc_els)), acc_absSi[1:], erracc_absSi[1:], uplims=upperlims[1:], ls='', color='black', marker='_', label='Early State')


#a1.scatter(range(len(acc_els)), acc_absSi, color='black', marker='o', label='Early State')
#a1.scatter(range(len(acc_els)), acc_mabsSS, color='green', marker='o', label='Steady State') # Minimally different, worse
a1.set_xlim(auto=False)
a1.semilogy()
plt.xlabel('Element')
plt.xticks(range(1, len(acc_els)), acc_els[1:])
a1.set_ylabel('Z/Si [APASS 2047] / Z/Si [BE]')
#a1.set_ylim(3e-1, 4e1)
a1.legend()
plt.show(block=False)


# Abundance plot in early phase normalised to Iron
"""
nFe = accreted_abs['Fe']
BE_ZFe = []
CI_ZFe = []
nFeppm = elements[26][5]/(10**6) / 56
for z in abundances:
    if z != 2:
        #BE number abundance fraction
        BE_ZFe.append((elements[z][2] / elements[z][1]) / (elements[26][2] / elements[26][1]))
        nppm = elements[z][5]/(10**6) / elements[z][1]
        CI_ZFe.append(nppm/nFeppm)

CI_ZFe = np.asarray(CI_ZFe)
BE_ZFe = np.asarray(BE_ZFe)/CI_ZFe
acc_absFe = np.asarray(list(accreted_abs.values()))/nFe / CI_ZFe

f = plt.figure()
plt.title('Abundances Relative to Iron')
a1 = plt.gca()
a1.axhline(1, color='grey', linestyle='--', label='CI Chondrite')
a1.scatter(range(len(acc_els)), acc_absFe, color='black', label='Early State')
plt.xticks(range(len(acc_els)), acc_els)
a1.semilogy()
a1.set_ylabel('Z/Fe [APASS 2047] / Z/Fe [CI]')
a1.legend()
plt.show()
"""

acc_absSi_CI = np.asarray(list(accreted_abs.values()))/nSi / CI_ZSi
BECI_ZSi = BE_ZSi / CI_ZSi
solCI_ZSi = sol_ZSi / CI_ZSi

# plot with baseline of CI Chondrite composition

fig = plt.figure()
plt.title('Parent Body Number Abundances')
a1 = plt.gca()
a1.axhline(1, color='orange', linestyle='--', label='CI Chondrite')
a1.plot(range(len(acc_els)), BECI_ZSi, label='Bulk Earth', linestyle='--', color='grey')
a1.plot(range(len(acc_els)), solCI_ZSi, label='Solar', linestyle='--', color='red')
a1.scatter(range(len(acc_els)), acc_absSi_CI, color='black', label='Early Phase')
plt.xticks(range(len(acc_els)), acc_els)
a1.semilogy()
plt.xlabel('Element')
a1.set_ylabel('Z/Si [APASS 2047] / Z/Si [CI]')
a1.legend()
plt.show(block=False)


pb_acc = {}
for z in abundances:
    if z>2:
        # number and mass of Z in CVZ
        nZ = abundances[z][2] * nHe
        mZ = nZ * elements[z][1] * AMU
        pb_acc[z] = [nZ, mZ] # [numb abundance, mass abundance] in cvz so in pb (min), early phase


# simple total mass, from sum of observed metals (no H)
totmass = np.sum(np.asarray(list(pb_acc.values()))[:,1])
print('Total Mass in photosphere = {:.3e}g'.format(totmass))

# mass given apparent Chondrite composition?

# O xs = nO - 1*nMg
#Oexcess = pb_abb[8][0] - 1*pb_acc[12][0] #...

nO = pb_acc[8][0]
OMgFrac = 1 * pb_acc[12][0]/nO # MgO
OSiFrac = 2 * pb_acc[14][0]/nO # SiO2
OFeFrac = 1  *pb_acc[26][0]/nO # FeO
OCaFrac = 1 * pb_acc[20][0]/nO # CaO
ONiFrac = 1 * pb_acc[28][0]/nO # NiO
OAlFrac = 1.5 * pb_acc[15][0]/nO # Al2O3

OxsFrac = max(1 - (OMgFrac + OSiFrac + OFeFrac + OCaFrac + ONiFrac + OAlFrac), 0)
Oxs = OxsFrac*nO
mH2O = Oxs * 18 * AMU
print('Early state max water mass from O excess = {:.3e} g'.format(mH2O))

# Steady State
SS_OMgFrac = 1 * SS_absO[12]
SS_OSiFrac = 2 * SS_absO[14]
SS_OFeFrac = 1 * SS_absO[26]
SS_OCaFrac = 1 * SS_absO[20]
SS_ONiFrac = 1 * SS_absO[28]
SS_OAlFrac = 1.5 * SS_absO[13]

SS_OxsFrac = max(1 - (SS_OMgFrac + SS_OSiFrac + SS_OFeFrac + SS_OCaFrac + SS_ONiFrac + SS_OAlFrac), 0)
SS_Oxs = SS_OxsFrac*nO
SS_mH2O = SS_Oxs * 18 * AMU
print('Steady state max water mass from O excess = {:.3e} g'.format(SS_mH2O))

plt.figure()
plt.bar(1, OSiFrac, bottom=0, color='red', width=0.6, label='SiO$_2$')
plt.bar(1, OFeFrac, bottom=(OSiFrac), color='orange', width=0.6, label='FeO')
plt.bar(1, OMgFrac, bottom=(OSiFrac+OFeFrac), color='yellow', width=0.6, label='MgO')
plt.bar(1, (ONiFrac+OCaFrac+OAlFrac), bottom=(OSiFrac+OMgFrac+OFeFrac), color='green', width=0.6, label='Others')
plt.bar(1, OxsFrac, bottom=(ONiFrac+OCaFrac+OAlFrac+OSiFrac+OMgFrac+OFeFrac), color='blue', width=0.6, label='O Excess')

plt.bar(2, SS_OSiFrac, bottom=0, width=0.6, color='red')
plt.bar(2, SS_OFeFrac, bottom=(SS_OSiFrac), width=0.6, color='orange')
plt.bar(2, SS_OMgFrac, bottom=(SS_OSiFrac+SS_OFeFrac), width=0.6, color='yellow')
plt.bar(2, (SS_OCaFrac+SS_ONiFrac+SS_OAlFrac), bottom=(SS_OSiFrac+SS_OMgFrac+SS_OFeFrac), width=0.6, color='green')
plt.bar(2, SS_OxsFrac, bottom=(SS_OCaFrac+SS_ONiFrac+SS_OAlFrac+SS_OSiFrac+SS_OMgFrac+SS_OFeFrac), width=0.6, color='blue')

#plt.text(1.45, OSiFrac/2, 'SiO$_2$', withdash=True)
#plt.text(1.45, OSiFrac + OFeFrac/2, 'FeO', withdash=True)
#plt.text(1.45, OSiFrac + OFeFrac + OMgFrac/2, 'MgO', withdash=True)
#plt.text(1.45, 1 - OxsFrac -(ONiFrac+OCaFrac+OAlFrac)/2, 'NiO, Al$_2$O$_3$, CaO', withdash=True)#'NiO, CaO, Al$_2$O$_3$')
#plt.text(1.45, 1-OxsFrac/2, 'O Excess', withdash=True )
plt.title('Oxygen Excess')
plt.xticks([1, 2], ['Early Phase', 'Steady State'])
plt.xlim(0.5,2.5)
plt.ylim(0,SS_OCaFrac+SS_ONiFrac+SS_OAlFrac+SS_OSiFrac+SS_OMgFrac+SS_OFeFrac+SS_OxsFrac)
plt.legend()
plt.show()

# Mass calculation from metals in photosphere
pbmass_extrap1 = []
for z in pb_acc:
    # metal mass * CI mass frac
    pbmass_extrap1.append(pb_acc[z][1] * (10**6)/elements[z][5]) # accZmass / Zmassfrac
extrap_totmass1 = np.average(pbmass_extrap1)
print('Average extrapolated mass = {:.3e} g'.format(extrap_totmass1))

""" # Not using P, S
pbmass_extrap2 = []
for z in pb_acc:
    if z not in [15,16]:
        # metal mass * CI mass frac
        pbmass_extrap2.append(pb_acc[z][1] * (10**6)/elements[z][5])
extrap_totmass2 = np.average(pbmass_extrap2)
print('Average extrapolated mass = {:.3e} g'.format(extrap_totmass2))
"""

##########################################
# ACCRETION RATES

# OVERSIMPLIFIED: Accretion flux = Mass diffused in 1 timescale / 1 timescale
acc_m = []
tau_sink = []
for z in pb_acc:
    if elements[z][4] !=0:
        acc_m.append(pb_acc[z][1])
        tau_sink.append(365*24*(60**2) * np.power(10, elements[z][4]))

#acc_masses = np.asarray([d[1] for d in list(pb_acc.values())])
#m_1t = acc_masses * np.exp(-1)
#timescales = 365*24*3600*10**np.asarray([d[4] for d in list(elements.values()) if d[4] != 0])

acc_rates = np.asarray(acc_m) / np.asarray(tau_sink)
sum_accrate = np.sum(acc_rates)
print('Summed accretion rate from {} elements = {:.3e}'.format(len(acc_rates), sum_accrate))
