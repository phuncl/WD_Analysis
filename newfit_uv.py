"""Test rescaling method applied to models
Want to make use of integrated flux and of rescaling values"""

import numpy as np
import numpy.ma as ma
import csv
from photometry import phot
from photometry import reddenings as rd
import matplotlib.pyplot as plt
import glob as g
from astLib import astSED
from scipy import integrate as spint


#### DEFINITIONS ####

PBD = rd.psb_dict

MASK_RANGES = [(1214.4, 1216.7), # Ly-alpha core
               (1198.5, 1201.6), # N  I atmospheric line
               (1301.2, 1302.7), # O  I atmospheric line
               (1259.1, 1261.9), # Si II
               (1263.8, 1266.5), # Si II
# other significant looking lines
               (1152.1, 1152.5), 
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

EBV = 0.01 # approx, from PANSTARRS maps for d <= 0.15 kpc

def vac2air(vac_spec):
    """Adjust to air wavelengths"""
    v = vac_spec[:,0]
    c1 = (1.0 + 2.735182 * np.power(10.0, -4.0))
    c2 = 131.4182
    c3 = 2.76249 * np.power(10.0, 8.0)
    a = v / (c1 + c2/(v**2) + c3/(v**4))
    vac_spec[:,0] = a
    return vac_spec

    
def open_uvarray(fname):
    """Manager function for opening and adjusting uv flux spectrum
    
    Returns prepared uv flux spectrum"""
    # read in wavl, flux, errflux from file and adjust to vacuum wavelength
    #array = vac2air(np.genfromtxt(fname, delimiter=' ')[:,:3]) # SKIPPING AS OFFSETS LY ALPHA CORE
    array = np.genfromtxt(fname, delimiter = ' ')[:,:3]
    # mask as required
    masked = mask_wavelengths(array, mask=1)
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
        integral_val = integrate(np.stack((sub_wavelengths, sub_fluxes), axis = -1))
        if calc_errors:
            integral_err = integrate(np.stack((sub_wavelengths, sub_err_fluxes), axis = -1))
            # append values to all_integrals
            all_integrals.append([integral_val, integral_err])
        else:
            all_integrals.append([integral_val])
    #print()
    return all_integrals


def integrate(spec):
    """Integrate a given two column (x,y) spectrum array"""
    i = spint.trapz(spec[:,1], spec[:,0])
    return i


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
    rescaling_inputs = np.asarray(flux_pts)
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
    phot_flux = np.asarray([y[0] for y in phot_values])
    phot_errf = np.asarray([z[0] for z in phot_values])

    model_interp_flux = np.interp(phot_wavl, mdlspec[:,0], mdlspec[:,1])
    phot_resid = phot_flux - model_interp_flux
    phot_resid_err = phot_errf + model_interp_flux*frac_error
    

    phot_onechi = np.square(phot_resid) / np.square(phot_resid_err)
    phot_lnn = len(phot_onechi)
    phot_chisq = np.sum(phot_onechi)
    #plt.plot(phot_wavl, phot_onechi, 'o')
    #plt.show()
    
    # returns chi squared per point that is not nan, including values from photometry
    rchisq = (chisq + phot_chisq)/(lennotnan + phot_lnn)
    
    if rchisq < 3:
        savnam = 'uv/plots/pht_' + nam.split('/')[-1][:-2] + 'png'
        plt.plot(spec_unchanged[:,0], spec_unchanged[:,1], color='grey', label='COS (oversampled)', alpha=0.5)
        plt.plot(spec_uv[:,0], spec_uv[:,1], color='blue', label='COS')
        plt.plot(spec_uv[:,0], mdlf, color='green', label='Model')
        axes = plt.gca()
        axes.set_ylim([0, 5.5e-14])
        plt.title(model_params_txt)
        plt.text(1300, 1.5e-14, "Reduced $\chi^2$ = {:.5e}".format(rchisq))
        plt.legend(loc=4)
        plt.savefig(savnam)
        #if rchisq < 2: plt.show()
        plt.close()

    return rchisq


def contour_abc(a, b, c, d, savenam = False, xlbl=None,
                ylbl=None, ttl=None, bounds=None,
                showit=False, src=None, best_fit=None):
    """For discrete parameters a,b,c,
    contour plot b vs c (coloured by d) for each value of a
    save contour plots to file
    """
    # unique values
    ua = np.unique(a)
    ub = np.unique(b)
    uc = np.unique(c)
    # loop over values - HORRIBLE, OPTIMISE? Possible?
    for aval in ua:
        plot_BFV=0
        plot_vals = np.zeros((len(ub), len(uc)))
        a_inds = np.argwhere(a==aval)
        for bval in ub:
            b_inds = np.argwhere(b==bval)
            for cval in uc:
                c_inds = np.argwhere(c==cval)
                xy = (np.argwhere(ub==bval), np.argwhere(uc==cval))
                # get index of line in master array which points to the combination three 'unique' values
                val_indx = np.intersect1d(a_inds, np.intersect1d(b_inds, c_inds))
                try:
                    plot_vals[xy] = d[val_indx]
                except:
                    print("Model with params {}, {}, {} not in this subset.".format(aval, bval, cval))
                    plot_vals[xy] = np.nan
                    continue
                if best_fit.any() and aval in best_fit and bval in best_fit and cval in best_fit:
                    plot_BFV = 1
                    BFVx = cval
                    BFVy = bval
        plt.contourf(uc, ub, plot_vals, extend='max', cmap='gnuplot_r', levels = bounds)
        if plot_BFV:
            plt.plot(BFVx, BFVy, 'xg', mew = 2, label='Best fit value')
            plt.legend(loc=3)
        clbr = plt.colorbar()
        clbr.set_label('Reduced $\chi^2$')
        clbr.set_ticks(bounds)
        plt.xlabel(xlbl)
        plt.ylabel(ylbl)
        plt.title(ttl + ' = {:.2f}'.format(aval))
        if savenam:
            plot_name = 'uv/plots/pht_{}_{}_{}.png'.format(src, savenam, aval)
            plt.savefig(plot_name)
        if showit:
            plt.show()
        plt.close()
        print("{} = {} complete".format(savenam, aval))


#### MAIN ####

# read observation data from files
spec_uv, spec_unchanged = open_uvarray('uv/APASSJ204713.82-125909.5_2017-11-04.dat')
np.savetxt('uv/APASSJ2047_resampled.dat', spec_uv, delimiter=',')
phot_master = photometrics()

phot_values = [x[0:3] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]
phot_names = [x[-1] for x in phot_master if 'panstarrs' in x[-1] or 'galex' in x[-1]]

#phot_values = [x[0:3] for x in phot_master if x[-2]]
#phot_nams = [x[-1] for x in phot_master if x[-1]]

# test on 'best fit' model -> t18500 g8.50 h-1.0, including reddening
#model_spec = rd.deredden_fix(get_dk('models/DAB/t18500_g850_h-1.0.dk'), -1*EBV)[0]
#mtest, wtest = rescaling_manager(np.copy(model_spec))
#frac_error = wtest[1]/wtest[0]

#rxsq = redu_chi_sq(mtest, 't18500_g850_h-1.0.dk')
# rescale model - include uncertainty - HOW TO DO??
    # RESCALE SPECTRUM INSTEAD?
    # INCLUDE ERRORS ON MODEL PTS?
        # FOR CHI-SQR, ERROR = SUM OF ERRORS ON OBS AND ON MDL PTS tick

model_filelist = g.glob('models/DAB/*.dk')
mlen = len(model_filelist)
mcount = 0.
rxsq_output = []
for m in model_filelist:
    model_params_txt = m[:-3].split('/')[-1].split('_')
    model_teff = float(model_params_txt[0][1:])
    model_logg = float(model_params_txt[1][1:])/100
    model_hyhe = float(model_params_txt[2][1:])
    
    model_data = rd.deredden_fix(get_dk(m), -1*EBV)[0]
    model_scaled, rescalings = rescaling_manager(np.copy(model_data))

    frac_error = rescalings[1]/rescalings[0]
    
    rxsq = redu_chi_sq(model_scaled, m)
    
    rxsq_output.append([model_teff, model_logg, model_hyhe, rxsq, rescalings[0]])
    mcount += 1
    print("{:.2f} % complete".format(100*mcount/mlen))

rxsq_array = np.asarray(rxsq_output)

np.savetxt('uv/chi_sqr/pht_rxsq_params.dat', rxsq_array, delimiter=',', header='T_eff, logg, H/He, reduced chi^2, rescaling')

# make contour plots in h/he

tf = np.copy(rxsq_array[:,0])
lg = np.copy(rxsq_array[:,1])
hh = np.copy(rxsq_array[:,2])
x2 = np.copy(rxsq_array[:,3])
minxs = np.nanmin(rxsq_array[:, 3])
BFV = np.copy(rxsq_array[np.argwhere(rxsq_array[:, 3]==minxs), :])[0][0] # not sure why nested twice but it is
boundaries = [1.,1.5,2.,2.5,3,4,5,7,9]
contour_abc(hh, lg, tf, x2, 'fxdhhe','$T_{eff}$', '$log(g)$', 'Fixed [H/He]',
            boundaries, src='COS', best_fit=BFV)
contour_abc(lg, hh, tf, x2, 'fxdlg', '$T_{eff}$', 'Fixed [H/He]', '$log(g)$',
            boundaries, src='COS', best_fit=BFV)

# MAKE PLOT (data - model)/errors for goodness of fit
# BUT WITH WHAT PARAMETERS
