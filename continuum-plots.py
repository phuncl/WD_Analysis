""""""

import csv
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt

from astLib import astSED
from scipy import interpolate
from scipy import integrate as spint
from photometry import reddenings as rd

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

PBD = rd.psb_dict

####    FUNCTIONS


def photometrics():
    """"""
    with open('photometric_fluxes.dat', 'r') as fin:
        fread = csv.reader(fin, delimiter=',')
        dat = [[float(x) for x in line[:-1]] + [line[-1]] for line in fread]
    return dat



def read_massradius(file_loc):
    """
    Read mass-radius grid file, to interpolate onto later
    """
    dat = np.genfromtxt(file_loc, names=True)
    t = dat['Teff']
    log = np.array([7.0,7.5,8.0,8.5,9.0,9.5])
    
    massgrid_data = np.stack((dat['m7_0'], dat['m7_5'], dat['m8_0'], dat['m8_5'], dat['m9_0'], dat['m9_5']), axis=1).T
    massgrid = interpolate.interp2d(t, log, massgrid_data, kind='cubic')
    
    radgrid_data = np.stack((dat['r7_0'], dat['r7_5'], dat['r8_0'], dat['r8_5'], dat['r9_0'], dat['r9_5']), axis=1).T
    radgrid = interpolate.interp2d(t, log, radgrid_data, kind='cubic')
    
    return massgrid, radgrid


def rescale_optspectrum(spec_data, filters, mags=None, errmags=None, fluxs=None, errfluxs=None):
    """Procedure to rescale a spectrum by the photometry in the wavelength range of spectrum"""
    """
    if (not mags) and (not fluxs):
        print('No input magnitudes or fluxes given for photometry.')
        print('Returning unscaled spectrum')
        return spec_data
    if (not fluxs):
        # convert mags to fluxs
        # TO BE WRITTEN
        pass
    """    
    min_wav, max_wav = min(spec_data[:,0]), max(spec_data[:,0])
    spec_sed = astSED.SED(spec_data[:,0], spec_data[:,0])
    
    rescalers = []
    for i, f in enumerate(filters):
        pband = PBD[f]
        pbandmin = min([x[0] for x in pband.asList() if x[1]>0.00005])
        pbandmax = max([x[0] for x in pband.asList() if x[1]>0.00005])
        if (pbandmin > min_wav) and (pbandmax < max_wav):
            spec_flux = spec_sed.calcFlux(pband)
            if fluxs[0]:
                resc = fluxs[i]/spec_flux
                errresc = errfluxs[i]/spec_flux
            elif mags[0]:
                f, errf = astSED.flux2Mag(mags[i], errmags[i])
                resc = f/spec_flux
                errresc = errf/spec_flux
            else:
                print('Something strange has happened. Returning unscaled spectrum.')
            rescalers.append([resc, errresc])
    f_ratios = np.asarray(rescalers)
    mean_numer = np.sum(f_ratios[:,0] / np.square(f_ratios[:,1]))
    mean_denom = np.sum(1 / np.square(f_ratios[:,1]))
    resc_fac = mean_numer / mean_denom
    resc_facerr= np.sqrt(1 / mean_denom)
    
    spec_output = np.copy(spec_data)
    spec_output[:,1] *= resc_fac
    spec_output[:,2] = spec_data[:,1] * np.sqrt((spec_data[:,2]*resc_fac/spec_data[:,1])**2 + (spec_data[:,1]*resc_facerr/resc_fac)**2) 
    
    return spec_output


def redshift_wavelengths(data_array, beta):
    """Apply redshift to column one of an array"""
    if len(data_array.shape) > 1:
        wavs = np.copy(data_array[:,0])
    else:
        wavs = np.copy(data_array)
    
    new_wavs = wavs * np.sqrt((1 + beta)/(1 - beta))
    data_array[:,0] = new_wavs
    return data_array


def plot_phot(phot_data, axis, markershape='x', col='k'):
    w = np.asarray([x[0] for x in phot_data])
    f = np.asarray([x[1] for x in phot_data])
    ef = np.asarray([x[2] for x in phot_data])
    
    upperlims = ef==0
    ef[np.argwhere(ef==0)] = f[np.argwhere(ef==0)]*0.6
    axis.errorbar(w, f, ef, None, fmt=markershape, uplims=upperlims, marker='x', color=col)
    

####    MAIN

BETA = 35000/3e8
prlx = 0.01017
errprlx = 0.00008
D = 1/prlx
errD = D * errprlx/prlx
pc2cm = 3.086e18 # 1 pc in cgs
Rsun = 6.96e10 # Solar radius in cgs
G = 6.674e-08 # cgs

mgrid, rgrid = read_massradius('mass_radius/mr.dat') # mass-radius grids

photometry = photometrics() # all photometry
phot_filters = [p[-1] for p in photometry]
phot_wavs = np.asarray([p[0] for p in photometry])
phot_fluxes = np.asarray([p[1] for p in photometry])
phot_errfluxes = np.asarray([p[2] for p in photometry])
phot_uplims = phot_errfluxes == 0

apass = []
twomass = []
galex = []
panstarrs = []
gaia = []
for p in photometry:
    if 'apass' in p[-1]: apass.append(p)
    if '2mass' in p[-1]: twomass.append(p)
    if 'galex' in p[-1]: galex.append(p)
    if 'panstarrs' in p[-1]: panstarrs.append(p)
    if 'gaia' in p[-1]: gaia.append(p)

spec_uv = np.genfromtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat', usecols=(0,1,2)) # uv spectrum
spec_uv = ma.masked_inside(spec_uv, 1214, 1217)

spec_int = np.genfromtxt('optical/J2047-1259_201505_int.dat')
spec_int_resc = rescale_optspectrum(spec_int, phot_filters, fluxs=phot_fluxes, errfluxs=phot_errfluxes)
spec_whtb = np.genfromtxt('optical/apassj2047_wht_blue.dat')
spec_whtb_resc = rescale_optspectrum(spec_whtb, phot_filters, fluxs=phot_fluxes, errfluxs=phot_errfluxes)

spec_phot = np.genfromtxt('optical/int_photrescaled.dat')

spec_xsh_uvb = np.genfromtxt('')

spec_model = np.genfromtxt('models/subgrid2018/DBA_18150.0_8.1H-1.1.dk')
spec_model[:,1] *= 0.25e-8
spec_model, Av = rd.deredden_fix(redshift_wavelengths(spec_model, BETA), -0.01)
model_r = rgrid(18150, 8.1)
s_val = 4*np.pi * model_r**2 / (D*pc2cm)**2
#errs_val = 2*4*np.pi * model_r**2 * errD/D
spec_model[:,1] *= s_val
    

f, a = plt.subplots(3, 1)

a[0].plot(spec_uv[:,0], spec_uv[:,1], 'k')
a[0].set_xlim(auto=False)
a[0].set_ylim(auto=False)
a[0].plot(spec_model[:,0], spec_model[:,1], 'r')

#a[1].plot(spec_int[:,0], spec_int[:,1], 'k')
a[1].plot(spec_phot[:,0], spec_phot[:,1], 'k')
a[1].set_xlim(auto=False)
a[1].set_ylim(auto=False)
a[1].plot(spec_model[:,0], spec_model[:,1], 'r')
a[1].set_ylabel('Flux [erg/cm$^{-2}$/s/\AA]')

a[2].set_xlim(auto=False)
a[2].set_ylim(auto=False)
a[2].plot(spec_model[:,0], spec_model[:,1], 'r')
a[2].set_xlim(auto=True)
a[2].set_ylim(auto=True)
a[2].set_yscale('log')
plot_phot(apass, a[2], col='orange', label='APASS')
plot_phot(twomass, a[2], col='darkmagenta', label='2MASS')
plot_phot(galex, a[2], col='blue', label='GALEX')
plot_phot(panstarrs, a[2], col='green', label='Pan-STARRS')
plot_phot(gaia, a[2], col='cyan', label='Gaia')
#a[2].errorbar(phot_wavs, phot_fluxes, phot_errfluxes, None, 'k.')
a[2].set_xlabel('Wavelength [\AA]')
plt.tight_layout()
plt.show()
