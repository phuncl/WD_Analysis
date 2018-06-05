"""Calculate stellar radius
Use input model, * 0.25e-8 to convert to eddington flux
rescale to hst continuum / photometry
scaling factor = S = 4 pi r^2/d^2

"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint
from photometry import phot
from photometry import reddenings as rd
from astLib import astSED


PBD = rd.psb_dict

prlx = 0.01017
errprlx = 0.00008
D = 1/prlx
errD = d * errprlx/prlx

pc2cm = 3.086e18 # 1 pc in cm
Rsun = 6.96e10 # Solar radius in cm
G = 6.67e-08

samples = 10000

##########################################


def getthefile(fn, dlim="", sh=0, uc=None):
    """"""
    try:
        return np.genfromtxt(fn, delimiter=dlim, skip_header=sh, usecols=uc)
    except OSError:
        print('File {} not found'.format(fn))
        quit()


def get_lsf_lp3(fname, dlim=''):
    """Open and read lsf for lifetime position 3"""
    # could allow reading of any lifetime pos by variable input - from fits file
    d_array = np.transpose(np.genfromtxt(fname, delimiter=dlim))
    w = np.copy(d_array[:,0]) # wavelengths given in lsf file
    r = np.copy(d_array[:,1:]) # response function 
    # by pixel (row is same pixel) for each wavelength (column is same wavelength)
    return w, r


def integrate(spec):
    """Integrate a given two column (x,y) spectrum array"""
    i = spint.trapz(spec[:,1], spec[:,0])
    return i


def photometrics():
    with open('photometric_fluxes.dat', 'r') as fin:
        fread = csv.reader(fin, delimiter=',')
        dat = [[float(x) for x in line[:-1]] + [line[-1]] for line in fread]
    return dat


##########################################

modl = getthefile('models/dab-v5-lsf_rds-tru.dk')
modl[:,1]*= 0.25e-8 # conversion to normal flux units from whatever detlev uses

spec = np.genfromtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat', usecols=(0,1,2))

# define region of spectrum and integrate
lims = np.logical_and(modl[:,0]>1300, modl[:,0]<1400)

modl_sub = modl[np.where(lims)]
modl_intg = spint.trapz(modl_sub[:,1], modl_sub[:,0])

# true wavelength range should include 2 half widths of wavelength bins!
# small but should include anyway

# DEREDDEN THIS DATA?


spec_sub = spec[np.where(lims)]
#spec_intg = spint.trapz(spec_sub[:,1], spec_sub[:,0]) # direct integration
# error calculation
fluxes, errors = spec_sub[:,1].squeeze(), spec_sub[:,2].squeeze()

# create samples
spec_subsample = np.random.normal(fluxes, errors, (samples, len(fluxes)))

# integrate over 1300 - 1400 wavelength region
spec_intgsample = np.trapz(spec_subsample, spec_sub[:,0], axis=1)
spec_intg = np.mean(spec_intgsample)
spec_intgerr = np.std(spec_intgsample)

S = spec_intg/modl_intg
errS = spec_intgerr/modl_intg

R = np.sqrt(S * (D * pc2cm)**2 / (4*np.pi))
r_err1 = 0.5 * np.power(S, -0.5) * D * pc2cm * errS / np.sqrt(4*np.pi)
r_err2 = np.power(S, 0.5) * errD * pc2cm / np.sqrt(4*np.pi)
errR =  np.sqrt(r_err1**2 + r_err2**2) # dr = ((dr/dx dy)**2 + (dr/dy dx)**2)**0.5

print('UV spectrum:')
print('R = {:.3e} +/- {:.2e}cm = ({:.3e} +/- {:.2e}) Rsun'.format(R, errR, R/Rsun, errR/Rsun))

##########################################

modl_full = getthefile('models/dab-v5.dk')
# redshift
modl_full[:,1] *= 0.25e-8 # conversion to normal flux units from whatever detlev uses

# dab-v5 is similar values to dab-v5-lsf..., but less sampled?!

phot = photometrics()
panstarrs = [x for x in phot if 'panstarrs' in x[-1]]

modl_sed = astSED.SED(modl_full[:,0], modl_full[:,1])

modl_flux = []
phot_flux = []
errphot_flux = []

"""
for filt_data in panstarrs:
    passbnd = PBD[filt_data[-1]]
    #synth_flux[filt_data[-1]] = modl_sed.calcFlux(passbnd)
    
    # WANT TO INTRODUCE ERROR ON THIS MEASUREMENT
    modl_flux.append(modl_sed.calcFlux(passbnd))
    
    # CAN INCLUDE ERROR ON THIS MEASUREMENT
    phot_flux.append(filt_data[1])
    
    #errp_flux.append(filt_data[2])

# MC SAMPLING OF FLUX NEEDED
"""

filter_fluxes = []

for filt_data in panstarrs:
    passbnd = PBD[filt_data[-1]]
    modl_flux.append(modl_sed.calcFlux(passbnd))
    # photometric flux and error
    phot_flux.append(filt_data[1])
    errphot_flux.append(filt_data[2])
    # 10,000 fluxes as array
    filter_fluxes.append(np.random.normal(filt_data[1], filt_data[2], (samples, 1)))

# filter_fluxes is a set of arrays of sampled flux
sample_fluxes = np.stack(filter_fluxes).squeeze() # extra dimension appears from stacking

modflx = np.asarray(modl_flux)
modflx.shape = (len(modl_flux),1)

sample_S2 = np.mean(sample_fluxes, axis=1) / modflx
sample_errS2 = np.std(sample_fluxes, axis=1) / modflx

S2 = np.mean(sample_S2)
errS2 = np.sqrt(np.sum(np.square(sample_errS2)))/len(sample_errS2)
#errS2 = np.std(sample_S2)

#print('S2:')
#print(S2, errS2)

# CAN WE IMPLEMENT INTO A FITTING PROCEDURE THAT THE RESCALING FACTORS SHOULD ALL BE THE SAME?

R2 = np.sqrt(S2 * (D * pc2cm)**2 / (4*np.pi))
r2_err1 = 0.5 * np.power(S2, -0.5) * D * pc2cm * errS2 / np.sqrt(4*np.pi)
r2_err2 = np.power(S2, 0.5) * errD * pc2cm / np.sqrt(4*np.pi)
# dr = ((dr/dx dy)**2 + (dr/dy dx)**2)**0.5
errR2 =  np.sqrt(r2_err1**2 + r2_err2**2)


# NEED ERRORS ON THIS!!!

print('Panstarrs Photometry:')
print('R = {:.3e} +/- {:.2e}cm = {:.2e} Rsun'.format(R2, errR2, R2/Rsun))
    
    


