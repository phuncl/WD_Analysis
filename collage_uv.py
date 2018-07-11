"""
Selection of basic plotting tools for WD spectra
"""

import numpy as np

from scipy import integrate as spint
from matplotlib import pyplot as plt
from matplotlib import gridspec
from photometry import reddenings as rd

from astropy.modeling import models, fitting
from astropy.convolution import convolve
from scipy import integrate as spint


def apply_redshift(beta, spec, plot=0):
    """Redshift spec by beta = v/c"""
    print("Redshifting for {} km/s".format(beta*300000))
    spc = np.copy(spec)
    wavs = np.copy(spc[:,0])
    if plot:
        plt.figure()
        plt.plot(wavs, c='red')
    # redshift
    wavs *= 1 + (np.sqrt(1+beta)/np.sqrt(1-beta) - 1)
    if plot:
        plt.plot(wavs, c='green')
        plt.show()
    spc[:,0] = wavs
    return spc


def getthefile(fn, dlim="", sh=0, uc=None):
    """"""
    try:
        return np.genfromtxt(fn, delimiter=dlim, skip_header=sh, usecols=uc)
    except OSError:
        print('File {} not found'.format(fn))


def normaliser(obs, mod): # DEFUNCT
    """normalise mod data to obs data
    THIS WILL INTEGRATE OVER Ly ALPHA - THROWS OFF
    """
    # find overlap region 
    # integrate in overlap region
    # rescale
    owmin, owmax = np.min(obs[:,0]), np.max(obs[:,0])
    mwmin, mwmax = np.min(mod[:,0]), np.max(mod[:,0])
    
    o_overlap = obs[np.where(np.logical_and(obs[:,0] > mwmin, obs[:,0] < mwmax))]
    m_overlap = mod[np.where(np.logical_and(mod[:,0] > owmin, mod[:,0] < owmax))]
    
    o_int = spint.trapz(o_overlap[:,1], o_overlap[:,0])
    m_int = spint.trapz(m_overlap[:,1], m_overlap[:,0])
    o_over_m = o_int/m_int
    
    mdlr = np.copy(mod)
    mdlr[:,1] *= o_over_m
    
    return mdlr


def cont_norm_basic(obs, mod, points, r=1):
    """Normalise by sampled points for whole continuum"""
    
    ratios = []
    for p in points:
        o_sub = obs[np.where(np.logical_and(obs[:,0]>p, obs[:,0]<p+r))]
        m_sub = mod[np.where(np.logical_and(mod[:,0]>p, mod[:,0]<p+r))]
        #print(o_sub.shape, m_sub.shape)
        
        # calculate flux density / angstrom (accounts for actual width of measured segment)
        o_int = spint.trapz(o_sub[:,1], o_sub[:,0]) / np.ptp(o_sub[:,0])
        m_int = spint.trapz(m_sub[:,1], m_sub[:,0]) / np.ptp(m_sub[:,0])
        
        ratios.append(o_int/m_int)
    
    rescale = np.mean(ratios)
    
    rescmod = np.copy(mod)
    rescmod[:,1] *= rescale
    
    print('Model rescaling applied ({:.4e})\n'.format(rescale))
    # do we want ratios?
    return rescmod, ratios


def plot_two(arr_list, lbls=None, colrs=None):
    """Plot all arrays in list"""
    print(arr_list)
    f = plt.figure()
    for i in range(len(arr_list)):
        arr = arr_list[i]
        arr.shape
        print(arr)
        if lbls and colrs: plt.plot(arr[:,0], arr[:,1], label=lbls[i], c=colrs[i])
        else: plt.plot(arr[:,0], arr[:,1])
    plt.legend()
    plt.show()


########
# DATA
cos = getthefile('uv/APASSJ204713.82-125909.5_2017-11-04.dat', uc=(0,1))
optint = getthefile('optical/J2047-1259_201505_int.dat', sh=7, dlim=' ', uc=(0,1))
optwhtb = getthefile('optical/apassj2047_wht_blue.dat', uc=(0,1))

# OLD MODELS
#dkmodv2 = getthefile('models/dab-150318-lsf_red.dk')
#dkmodv3 = getthefile('models/metal-18500/dab-v3-n68-lsf_red.dk')
#dkmodv3b = getthefile('models/metal-18500/dab-v3-n80-lsf_red.dk')
#dkmodv4 = getthefile('models/metal-18500/dab-v4-lsf_rds.dk')
#dkmodv5 = getthefile('models/metal-18500/dab-v5-lsf_rds.dk')
#dkmodv5n = getthefile('models/metal-18500/dab-v5-noise-lsf_rds.dk')
dkmodv6 = getthefile('models/metal-18500/dab-v6-lsf_rds.dk')
#dkmodAl = getthefile('models/metal-18500/dab-Al-lsf.dk')

# OLD OPTICAL 
#dkbasic = getthefile('models/DAB/t18500_g850_h-0.75.dk', sh=52)
#dkoptv5 = getthefile('models/metal-18500/dab-v5.dk', sh=52)
#dkoptv6 = getthefile('models/metal-18500/dab-v6.dk', sh=52)


# NEW MODELS
dkmod18000 = getthefile('models/dabz_18000_825_-1_redlsf.dk')
dkmod18000b = getthefile('models/dabz_18000_825_-1_lsfrds.dk')
dkopt18000 = getthefile('models/dabz_18000_825_-1.dk')

norm_pts = [1137, 1150, 1161, 1171, 1180, 1188, 1199, 1210, 1222, 1238,
            1245, 1269, 1291, 1315, 1326, 1341, 1357, 1372, 1386, 1396,
            1414, 1422]

#modv2, rescaling_values = cont_norm_basic(cos, dkmodv2, norm_pts)
#modv3, rescaling_values = cont_norm_basic(cos, dkmodv3, norm_pts)
#modv3b, rescaling_values = cont_norm_basic(cos, dkmodv3b, norm_pts)
#modv4, rescaling_values = cont_norm_basic(cos, dkmodv4, norm_pts)
#modv5, rescaling_values = cont_norm_basic(cos, dkmodv5, norm_pts)
modv6, rescaling_6 = cont_norm_basic(cos, dkmodv6, norm_pts)
mod18000, resc_18000 = cont_norm_basic(cos, dkmod18000, norm_pts)
mod18000b, resc_18000b = cont_norm_basic(cos, dkmod18000, norm_pts)

#modv5n, r = cont_norm_basic(cos, dkmodv5n, norm_pts)
#modAl, rescaling_Al = cont_norm_basic(cos, dkmodAl, norm_pts)
#modAl = apply_redshift(37/3e5, modAl)



#plot_two((cos, mod), ('COS', 'Model'))

f = plt.figure()
plt.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
#plt.plot(modv2[:,0], modv2[:,1], c='red', label='Model v2')
#plt.plot(modv3[:,0], modv3[:,1], c='green', label='Model v3 high N')
#plt.plot(modv3b[:,0], modv3b[:,1], c='blue', label='Model v3 low N')
#plt.plot(modv4[:,0], modv4[:,1], c='green', label='Model v4')
#plt.plot(modv5[:,0], modv5[:,1], c='red', label='Model v5')
plt.plot(modv6[:,0], modv6[:,1], c='green', label='Model v6')
plt.plot(mod18000[:,0], mod18000[:,1], c='orange', label='18000/8.25/-1.0 lsf')
plt.plot(mod18000b[:,0], mod18000b[:,1], c='red', lw=0.5, label='18000/8.25/-1.0 lsf/mjh')
#plt.plot(modAl[:,0], modAl[:,1], c='purple', label='Model Al')
plt.legend()
plt.show()

"""
# Collage plot
mod = modv5

bigfig = plt.figure(figsize=[8,8])
gs = gridspec.GridSpec(3,2)
a11 = plt.subplot(gs[0,0])
a12 = plt.subplot(gs[0,1])
a22 = plt.subplot(gs[1,1])
a21 = plt.subplot(gs[1,0])
auv = plt.subplot(gs[2,:])
a11.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
a21.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
a22.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
a12.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
auv.plot(cos[:,0], cos[:,1], c='black', lw=0.5)
a11.plot(mod[:,0], mod[:,1], c='red')
a21.plot(mod[:,0], mod[:,1], c='red')
a22.plot(mod[:,0], mod[:,1], c='red')
a12.plot(mod[:,0], mod[:,1], c='red')

auv.plot(mod[:,0], mod[:,1], c='red')
# ADD SOME LINE LABELS
# WANT TO HAVE A FOCUS ON VOLATILES IN AT LEAST ONE PLOT
a11.text(1148.4, 3.8e-14, '|FeII')
a11.text(1150.07, 3.8e-14, '|PII') # see Gaensicke2012
a11.text(1151.3, 4.4e-14, '|FeII')
a11.text(1152.25, 3.8e-14, '|OI')
a11.text(1152.95, 3.8e-14, '|PII')
a11.text(1154.15, 3.8e-14, '|PII')
a11.set_xlim(1147.8,1154.5)
a11.set_ylim(0,5e-14)

# rethink the regions used in this plot

a12.text(1248.56, 2.2e-14, '|SiII')
a12.text(1249.96, 2.2e-14, '|PII')
a12.text(1250.22, 2.2e-14, '|SiII')
a12.text(1250.56, 1.9e-14, '|SiII')
a12.text(1250.71, 2.2e-14, '|SII')
a12.text(1251.30, 2.2e-14, '|CII')

a21.text(1259.53, 2.8e-14, '|SII')
a21.text(1260.42, 2.8e-14, '|SiII')
a21.text(1260.83, 3.2e-14, '|FeII')
a21.text(1261.62, 3.0e-14, '|CII')
a21.text(1262.16, 3.5e-14, '|FeII')
a21.text(1264.74, 3.0e-14, '|SiII')
a21.text(1265.01, 3.5e-14, '|SiII')

a22.text(1296.73, 4.7e-14, '|SiIII')
a22.text(1298.94, 4.7e-14, '|SiIII')
a22.text(1302.18, 4.7e-14, '|OI')
a22.text(1303.33, 4.7e-14, '|SiII')
a22.text(1304.87, 5.2e-14, '|OI')
a22.text(1306.04, 4.7e-14, '|SiII')
a22.text(1309.29, 4.7e-14, '|SiII')

a22.text(137.31, 4.7e-14, '|FeII')

#a11.text.TextWithDash(1251.79, 2.5e-14, '|PII', horizontalalignment='center', dashdirection=1)

bigfig.text(0.5, 0.05, 'Wavelength $\AA$', ha='center')
bigfig.text(0.05, 0.5, '$F_\lambda [erg cm^{-2} s^{-1} \AA^{-1}]$', rotation='vertical')
plt.show()
"""


"""
whtb_pts = [4050, 4150, 4300, 4600, 4950,]#  5500, 6100, 6500, 6800]
mod_opt, rescaling_opt = cont_norm_basic(optwhtb, dkoptv6, whtb_pts, r=25)
red_modopt = apply_redshift(37/3e5, mod_opt)
mod_int, r = cont_norm_basic(optint, dkbasic, whtb_pts, r=25)
red_modint = apply_redshift(37/3e5, mod_int)

plt.figure()
plt.plot(optwhtb[:,0], optwhtb[:,1], c='black', lw=0.5)
plt.plot(red_modopt[:,0], red_modopt[:,1], c='red')
plt.show()

# CONVOLVE OPTICAL MODEL TO OPTICAL RES
x = np.arange(-1,1,0.001)
xw = int(len(x)/2)
g = models.Gaussian1D(amplitude=1, mean=0, stddev=0.003)
optgauss = g(x)
optgauss /= (spint.trapz(optgauss, x))
cvopt = np.copy(red_modopt)
cvopt[:,1] = np.convolve(cvopt[:,1], optgauss)[xw-1: -1*xw]

convopt, c = cont_norm_basic(optwhtb, cvopt, whtb_pts, r=25)


bigfig2 = plt.figure(figsize=[8,7])
gs = gridspec.GridSpec(2, 3)
awht = plt.subplot(gs[0,:-1])
awhtCa = plt.subplot(gs[0,-1])
aint = plt.subplot(gs[1,:])
awht.plot(optwhtb[:,0], optwhtb[:,1], c='black', lw=0.5)
awht.plot(convopt[:,0], convopt[:,1], c='red')
awhtCa.plot(optwhtb[:,0], optwhtb[:,1], c='black', lw=0.5)
awhtCa.plot(convopt[:,0], convopt[:,1]-1.0e-16, c='red')
#awhtCa.set_xlims([3875,3975])
aint.plot(optint[:,0], optint[:,1], c='black', lw=0.5)
aint.plot(red_modint[:,0], red_modint[:,1], c='red')
bigfig2.text(0.5, 0.05, 'Wavelength $\AA$', ha='center')
bigfig2.text(0.05, 0.5, '$F_\lambda [erg cm^{-2} s^{-1} \AA^{-1}]$', rotation='vertical')
plt.show()

"""
