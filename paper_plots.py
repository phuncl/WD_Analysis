""""""

import csv
import numpy as np
from photometry import reddenings as rd
from photometry import phot
import matplotlib.pyplot as plt
from astLib import astSED
from scipy import integrate as spint

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


PBD = rd.psb_dict

MASK_RANGES = [(1214.4, 1216.7), # Ly-alpha core
               (1198.5, 1201.6), # N  I atmospheric line
               (1301.2, 1302.7), # O  I atmospheric line
               (1259.1, 1261.9), # Si II
               (1263.8, 1266.5), # Si II
# other significant looking lines
               #(1152.1, 1152.5), 
               (1190.1, 1191.0), # Si II
               (1193.0, 1193.7), # Si II, C  I ?
               (1194.3, 1195.0), # Si II
               (1249.9, 1250.7), # S  II ?
               (1251.0, 1251.5), # Si II
               (1253.7, 1254.1), # S  II
               (1298.9, 1299.5),
               (1304.2, 1306.4), # O  I (doublet)
               (1309.0, 1310.1), # Si II
               (1334.2, 1334.9), # C  II
               (1335.6, 1336.2), # C  II
               (1346.8, 1347.3),
               (1348.5, 1348.9),
               (1349.9, 1351.0), # Si II
               (1352.5, 1353.1),
               (1353.6, 1354.1)]
              # Cl I at 1334.7, 1347.7 ??

INTG_REGIONS = [[1155., 1190.], # for rescaling
                [1235., 1260.],
                [1355., 1425.]]


EBV = 0.01
    
def open_uvarray(fname, msk=0):
    """Manager function for opening and adjusting uv flux spectrum
    
    Returns prepared uv flux spectrum"""
    # read in wavl, flux, errflux from file and adjust to vacuum wavelength
    #array = vac2air(np.genfromtxt(fname, delimiter=' ')[:,:3]) # SKIPPING AS OFFSETS LY ALPHA CORE
    array = np.genfromtxt(fname, delimiter = ' ')[:,:3]
    # mask as required
    
    masked = mask_wavelengths(array, mask=msk)
    # rebin spectrum
    rebin = resample_uv(masked, 5)

    return rebin, array


def mask_wavelengths(spec_array, mask=0):
    """For each mask range in masks, apply to wavelength column of array
    Then mask all rows with masked wavelengths
    
    Return masked array"""
    if not mask:
        return spec_array
    else:
        print("Masking input array in these ranges:")
        for rangepair in MASK_RANGES:
            rmin, rmax = rangepair
            print(rmin, "-", rmax)
            spec_array = ma.masked_inside(spec_array, rmin, rmax)
            # technically this could mask flux or errors, but in practise will not as ~10**-15
        spec_marray = ma.mask_rows(spec_array)
        print('\n')
        return spec_marray


def resample_uv(uv_array, binsize=5):
    """Rebin spectrum by given size
    Excess data points trimmed from end of array
    uv_array = three-column array of wavelength/flux/error
    """
    print("Resampling spectrum with binsize of {}\n".format(binsize))
    # trim uv_array to multiple of bin_size
    uv_len = len(uv_array)
    cut = uv_len%binsize
    new_len = uv_len//binsize 
    uv_cut = uv_array[:(-1*cut)] # Losing up to binsize-1 lines of data.
    
    old_wavs = uv_cut[:,0]
    old_flux = uv_cut[:,1]
    old_errs = uv_cut[:,2]
    
    uv_smoothed = np.zeros((new_len,3))
    uv_smoothed[:,0] = np.asarray([np.mean(b, 0) for b in np.split(old_wavs, new_len)])
    uv_smoothed[:,1] = np.asarray([np.mean(b, 0) for b in np.split(old_flux, new_len)])
    #err = rms of errors
    uv_smoothed[:,2] = np.asarray([np.sqrt(np.sum(b**2))/binsize for b in np.split(old_errs, new_len)])
    
    return uv_smoothed


def rescaling_manager(model):
    """Manage overall operation of rescaling of model
    using photometry from panstarrs, galex
    and integrated flux from 3 regions of the observed spectrum
    """
    # create synthetic fluxes from unscaled model
    model_synth_flux = synthetic_fluxes(model, phot_names)
    
    flux_pts = []
    for i, line in enumerate(model_synth_flux):
        synth_pts = [model_synth_flux[i]] + phot_values[i][1:] # synthetic points
        flux_pts.append(synth_pts)

    # calculate rescalings for model and for data
    
    # integrate flux in chosen regions
    # for spectrum
    #print("Integrating parts of COS spectrum")
    spec_integrals = flux_integrator(spec_uv, calc_errors=1)
    # for model (no errors)
    #print("Integrating parts of model")
    model_integrals = flux_integrator(model, calc_errors=0)
    
    # format inputs to rescaling as model/observed/error
    integral_pts = []
    for j, lin in enumerate(model_integrals):
        intpts = model_integrals[j] + spec_integrals[j]
        integral_pts.append(intpts)
    
    #rescaling_inputs = np.asarray(flux_pts + integral_pts)
    rescaling_inputs = np.asarray(flux_pts)  # just using photometry
    phot_wavs = [float(x[0]) for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
    #integral_wavs = [float(np.mean(y)) for y in INTG_REGIONS]
    #central_wavs = np.asarray(phot_wavs + integral_wavs)
    central_wavs = np.asarray(phot_wavs)
    
    # calculate weighted mean scaling value andd error
    wm, errwm = weighted_mean(rescaling_inputs, central_wavs)
    
    model[:,1] *= wm
    #model_errors = np.copy(model) * errwm/wm
    #rescaled_model = np.concatenate((model, model_errors), axis=1) # add error column to rescaled spectrum
    
    return model, (wm, errwm)


def flux_integrator(spectrum, calc_errors=0):
    """Manage integration of a spectrum for ranges given by INTG_REGIONS
    
    Return list of lists of integrals and errors
    """
    wavelengths = spectrum[:,0]
    fluxes = spectrum[:,1]
    if calc_errors:
        err_fluxes = spectrum[:,2]
    all_integrals = []
    for minlim, maxlim in INTG_REGIONS:
        # separate region to integrate
        in_limits = np.logical_and(wavelengths > minlim, wavelengths < maxlim)
        sub_wavelengths = wavelengths[np.where(in_limits)]
        
        sub_fluxes = fluxes[np.where(in_limits)]
        if calc_errors: sub_err_fluxes = err_fluxes[np.where(in_limits)]

        # integrate spectrum, and errors if desired
        integral_val = intgrate(np.stack((sub_wavelengths, sub_fluxes), axis = -1))
        if calc_errors:
            integral_err = intgrate(np.stack((sub_wavelengths, sub_err_fluxes), axis = -1))
            # append values to all_integrals
            all_integrals.append([integral_val, integral_err])
        else:
            all_integrals.append([integral_val])
    #print()
    return all_integrals


def weighted_mean(dataset, wavs = None):
    """Calculate weighted mean from data array looking like a/x/errx
    
    Return error weighted mean of set
    """
    values = dataset[:,1] / dataset[:,0]
    errors = dataset[:,2] / dataset[:,0]
    #for i,v in enumerate(values):
        #print(v, '+/-', errors[i])
    #if wavs.any():
        #plt.errorbar(wavs, values, errors, c='red', fmt='o')
        #plt.semilogx(basex=10)
        #plt.show()
    
    mean_numer = np.sum(values / np.square(errors))
    mean_denom = np.sum(1 / np.square(errors))
    
    weighted_mean = mean_numer / mean_denom
    err_weighted_mean = np.sqrt(1 / mean_denom)
    
    return weighted_mean, err_weighted_mean
    

def redu_chi_sq(mdlspec, nam):
    """Calculate chi-sq value for model, relative to observation
    Create linear interpolation data for each obs x
    Calc chi-sq_i, take mean
    
    Return chi squared per point"""
    # set observed spec to smoothed uv fluxes
    obsf = spec_uv[:,1]
    err_obs = spec_uv[:,2] # AND ERROR FROM MODEL GOES WHERE
    mdlf = np.interp(spec_uv[:,0], mdlspec[:,0], mdlspec[:,1]) # create linear interpolated points in model
    resid = obsf - mdlf # calc residual values
    resid_err = err_obs + mdlf*frac_error
    
    onechi = np.square(resid) / np.square(resid_err)
    lennotnan = len(onechi[~np.isnan(onechi)]) # no of non-nan values in onechi
    chisq = np.nansum(onechi) # sum of chi squared values for spectrum
    #plt.plot(spec_uv[:,0], onechi, 'o')

    # now calculate same for panstarrs photometry FLUXES
    phot_wavl = np.asarray([x[0] for x in phot_values]) # is a wavelength, not a passband
    phot_flux = np.asarray([y[1] for y in phot_values])
    phot_errf = np.asarray([z[2] for z in phot_values])

    model_interp_flux = np.interp(phot_wavl, mdlspec[:,0], mdlspec[:,1])
    phot_resid = phot_flux - model_interp_flux
    phot_resid_err = phot_errf + model_interp_flux*frac_error
    
    phot_onechi = np.square(phot_resid) / np.square(phot_resid_err)
    phot_lnn = len(phot_onechi)
    phot_chisq = np.sum(phot_onechi)
    #plt.plot(phot_wavl, phot_onechi, 'o')
    #plt.show()
    #print('Spec chisq = {}, Phot chisq = {}'.format(chisq, phot_chisq))
    #input('Press Enter')
    # returns chi squared per point that is not nan, including values from photometry
    rchisq = (chisq + phot_chisq)/(lennotnan + phot_lnn)
    
    galex = phot_finder('galex')
    apass = phot_finder('apass')
    panstarrs = phot_finder('panstarrs')
    tmass = phot_finder('2mass')
    uplims = (np.asarray([x[0] for x in phot_master if x[2]==0]),
              np.asarray([x[1] for x in phot_master if x[2]==0]))
    
    if rchisq:
        #savnam = 'uv/plots/pht_newtest_{}pdf'.format(nam.split('/')[-1][:-2])
        f = plt.figure(figsize=(5,6))
        a1 = plt.subplot(311)
        a1.plot(spec_unchanged[:,0], spec_unchanged[:,1], color='grey', alpha=0.3)#, label='COS (oversampled)')
        a1.plot(spec_uv[:,0], spec_uv[:,1], color='black')#, label='COS')
        a1.plot(spec_uv[:,0], mdlf, color='red')#, label='Model')
        a1.set_ylim([0, 6e-14])
        a1.set_xlim(1130, 1430)
        #plt.title(model_params_txt)
        #plt.text(1300, 1.5e-14, "Reduced $\chi^2$ = {:.5e}".format(rchisq))
        
        a2=plt.subplot(312)
        #a2.set_yscale("log", nonposy='clip')
        plt.plot(intspec[:,0], intspec[:,1], color='black')
        plt.plot(model_scaled[:,0], model_scaled[:,1], color='red')
        a2.set_xlim(3600, 6900)
        a2.set_ylim(1e-16, 0.5e-14)
        
        a3=plt.subplot(313)
        #a3.set_yscale("log", nonposy='clip')
        plt.plot(model_scaled[:,0], model_scaled[:,1], color='red')
        #plt.errorbar(phot_wavl, phot_flux, phot_errf, marker='.', capsize=4, color='black', linestyle='None')
        #plt.plot(phot_wavl, phot_flux, marker='.', color='black', linestyle='None')
        plt.errorbar(galex[0], galex[1], galex[2], marker='o', markersize=4, color='black', linestyle='None', label='GALEX')
        plt.errorbar(apass[0], apass[1], apass[2], marker='^', markersize=4, color='black', linestyle='None', label='APASS')
        plt.errorbar(panstarrs[0], panstarrs[1], panstarrs[2], markersize=4, marker='s', color='black', linestyle='None', label='Pan-STARRS')
        plt.errorbar(tmass[0], tmass[1], tmass[2], marker='*', markersize=4, color='black', linestyle='None', label='2MASS')
        plt.errorbar(uplims[0], uplims[1], [0.15e-14]*3, marker='_', markersize=4, color='black', uplims=(1, 1, 1), linestyle='None', label='Upper Limits')
        #plt.legend()
        a3.set_xlim(1300, 22000)
        a3.set_ylim(-0.3e-14, 3e-14)
        #plt.xlabel(r'Wavelength [\AA]')
        #plt.ylabel(r'Flux [erg/cm$^2$/s/\AA]')
        f.text(0.5, 0.01, r'Wavelength \AA', ha='center')
        f.text(0., 0.5, r'Flux [erg/cm$^2$/s/\AA]', va='center', rotation='vertical')
        #plt.legend(loc=4)
        plt.tight_layout()
        #plt.savefig(savnam)
        plt.show()
    return rchisq


def photometrics():
    with open('photometric_fluxes.dat', 'r') as fin:
        fread = csv.reader(fin, delimiter=',')
        dat = [[float(x) for x in line[:-1]] + [line[-1]] for line in fread]
    return dat


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


def synthetic_fluxes(spectrum, filters):
    """Calculate sythetic fluxes from spectrum for all filters given
    
    filters = list of filter info
        each line should be formatted as str(filter_survey)
    
    Create SED object from spectrum
    
    Returns list of synthetic fluxes
    """
    #print("Creating synthetic fluxes for...")
    sed = astSED.SED(spectrum[:, 0], spectrum[:, 1])
    synths = []
    for f in filters:
        #print(f)
        pband = PBD[str(f)]
        synthflx = sed.calcFlux(pband)
        synths.append(synthflx)
    #print()
    synths = np.asarray([float(x) for x in synths])
    return synths


def intgrate(spec):
    """Integrate a given two column (x,y) spectrum array"""
    i = spint.trapz(spec[:,1], spec[:,0])
    return i


def phot_finder(surv):
    wavs = np.asarray([x[0] for x in phot_master if surv in x[-1]])# and x[2] !=0])
    flux = np.asarray([x[1] for x in phot_master if surv in x[-1]])# and x[2] !=0])
    errs = np.asarray([x[2] for x in phot_master if surv in x[-1]])# and x[2] !=0])
    return(wavs, flux, errs)

###################################
# DATA

# optical
intspec = np.genfromtxt('optical/int_rescaled.csv', delimiter=',')
# uv
spec_uv, spec_unchanged = open_uvarray('uv/APASSJ204713.82-125909.5_2017-11-04.dat')
#np.savetxt('uv/APASSJ2047_resampled.dat', spec_uv, delimiter=',')
# photometry
phot_master = photometrics()


#phot_values = [x[0:3] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
phot_values = [x[0:3] for x in phot_master if x[2]!=0.0] # any photometry with error bar
#phot_names = [x[-1] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
phot_names = [x[-1] for x in phot_master if x[2]!=0.0] # any photometry with error bar

###################################
# MODEL
m='models/dab-atmos-2013.dk'
model_data = rd.deredden_fix(get_dk(m), -1*EBV)[0]
model_scaled, rescalings = rescaling_manager(np.copy(model_data))

###################################
# ATMOSPHERE UV + PHOTOMETRY + OPTICAL 

# read observation data from files
spec_uv, spec_unchanged = open_uvarray('uv/APASSJ204713.82-125909.5_2017-11-04.dat')
#np.savetxt('uv/APASSJ2047_resampled.dat', spec_uv, delimiter=',')
phot_master = photometrics()

#phot_values = [x[0:3] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
phot_values = [x[0:3] for x in phot_master if x[2]!=0.0] # any photometry with error bar
#phot_names = [x[-1] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
phot_names = [x[-1] for x in phot_master if x[2]!=0.0] # any photometry with error bar

m='models/dab-atmos-2013.dk'
model_data = rd.deredden_fix(get_dk(m), -1*EBV)[0]
model_scaled, rescalings = rescaling_manager(np.copy(model_data))
frac_error = rescalings[1]/rescalings[0]
rxsq = redu_chi_sq(model_scaled, m)





