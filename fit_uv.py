"""Fit COS UV spectrum and #WHICH# photometry with best few models from optical fitting"""

# imports
import numpy as np
import numpy.ma as ma
import csv
from photometry import phot
from photometry import reddenings as rd
import matplotlib.pyplot as plt
import glob as g
from astLib import astSED

# constants
EMPTY = -999.0 # empty photometry value
#EBV = rd.sfd_web(204713.82, -125909.5, 'equ 2000')[0] # extinction from online lookup
EBV = 0.01 # approx, from PANSTARRS maps for d <= 0.15 kpc
PBD = rd.psb_dict
#RXSQ = np.genfromtxt('optical/chi_sqr/int_best_rchisq_params.dat', delimiter=',') # optical fit results
MASK_RANGES = [(1214.0, 1217.2), # Ly-alpha core
               (1198.5, 1201.5), # N<?> line
               (1301.2, 1303.2), # OI line
               (1152.1, 1152.5), # other significant looking lines
               (1190.2, 1190.7), #
               (1193.2, 1193.6), #
               (1260.0, 1260.9), #
               (1264.2, 1265.6), #
               (1334.3, 1334.8), #
               (1335.6, 1336.0), #
              ]

# classes and functions

# ADD WAVELENGTH TRANSFORMATION?

class photometry_source:
    """Generic class of photometry source
    
    Takes filters, sources, line of photometry data and survey name.
    
    survey = suvey from which data were taken
    photometry = stored photometric data from survey
    fluxes = (flux, flux_err, central_wavelength) for each used filter
    filters = filters with data
    
    """
    def __init__(self, filters, phot_line, surv):
        self.survey = surv
        self.filters = filters
        self.photometry = {}
        self.fluxes = {}
        self.magnitudes = {}
        # phot line contains mag, errmag, ...(repeated)
        # calculate flux data and store in dict
        for i, val in enumerate(phot_line):
            if not i%2:
                if val != EMPTY:
                    if phot_line[i+1] != EMPTY: # Some consideration for EMPTY error values?
                        self.magnitudes[self.filters[i]] = [val, phot_line[i+1]]
                        self.photometry[self.filters[i]] = [val, phot_line[i+1]]
                        print("Calculating flux data from following data:")
                        print(phot_line[i], phot_line[i+1], self.filters[i], surv, "\n")

                        self.fluxes[self.filters[i]] = phot.return_fluxes(val,
                                                                          phot_line[i+1],
                                                                          self.filters[i],
                                                                          surv)
                    elif phot_line[i+1] == EMPTY:
                        self.magnitudes[self.filters[i]] = [val, 0]
                        self.photometry[self.filters[i]] = [val, 0]
                        #print("Calculating flux data from following data:")
                        #print(val, '<No error value>', self.filters[i], surv, "\n")
                        self.fluxes[self.filters[i]] = phot.return_fluxes(val,
                                                                          0,
                                                                          self.filters[i],
                                                                          surv)                     
                elif val == EMPTY:
                    pass
                else:
                    print("Something really weird has happened with value {}!".format(val))


def get_dk(fname, showplot=False):
    with open(fname, 'r') as f:
        d = f.readlines()[52:]
    d2 = [x.strip().split() for x in d] # make list of lists
    d3 = np.asarray([[float(y) for y in x] for x in d2])
    
    if showplot:
        plt.figure()
        plt.plot(d3[:,0], d3[:,1], c = 'r')
        plt.show()
    return d3


def read_photometry(fname):
    """Return photometry data from file."""
    with open('photometry.csv', 'r') as f:
        frd = csv.reader(f, delimiter=',')
        headers = next(frd)[1:]
        dat = [x for x in frd]
        sources = [x[0] for x in dat]
        # separate values and put into np array
        for i, line in enumerate(dat):
            dat[i] = [EMPTY if x=='' else float(x) for x in line[1:]]
        dat = np.asarray(dat)
    return headers, sources, dat


def flux_in_wav_range(wavmin, wavmax, flux_dict):
    """For dictionary of filter:(flux,err,wavelength)
    For wavelength in range wavmin < wavelength < wavmax add to list
    Return full list of [wavelength, flux, err, filter]] 
    """
    selected_fluxes = []
    for k in flux_dict:
        src_flux = flux_dict[k].fluxes
        src_mag = flux_dict[k].magnitudes
        for key in src_flux.keys():
            try:
                if wavmin <= src_flux[key][2] <= wavmax:
                    selected_fluxes.append([format(src_flux[key][2], '.4f'),
                                          format(src_flux[key][0], '.4e'),
                                          format(src_flux[key][1], '.4e'),
                                          format(src_mag[key][0], '.4f'),
                                          format(src_mag[key][1], '.4f'),
                                          str(key)+'_'+str(k)])
            except TypeError:
                pass
    return selected_fluxes


def plot_spectrum(spectrum, c=None, l=None, style='-', a=0.7):
    plt.plot(spectrum[:,0], spectrum[:,1], color=c, label=l, alpha=a, ls=style)


def plot_fluxpoints(fluxpoints, c=None):
    for line in fluxpoints:
        if line[2]:
            if c: plt.errorbar(line[0], line[1], line[2], label=line[-1], fmt='o', color=c, ms=3)
            else: plt.errorbar(line[0], line[1], line[2], label=line[-1], fmt='o', ms=3)
        else:
            if c: plt.errorbar(line[0], line[1], label=line[-1], fmt='s', color=c, ms=3)
            else: plt.errorbar(line[0], line[1], label=line[-1], fmt='s', ms=3)


def minmax(dat):
    """Find min,max value of 1d dat array -+5% (for plots))"""
    return [min(dat)*.95, max(dat)*1.05]


def photometry_fluxes():
    """Open photometry file, generate fluxes"""
    phot_filter, phot_source, phot_data = read_photometry('photometry.csv')
    # generate flux data
    flux_data = {}
    for i,s in enumerate(phot_source):
        flux_data[s] = photometry_source(phot_filter, phot_data[i], s)
    all_phot_flux = flux_in_wav_range(0,1e6,flux_data)
    return phot_filter, phot_source, phot_data, flux_data, all_phot_flux


def resample_uv(uv_array, binsize=5):
    """Rebin spectrum by given size
    Excess data points trimmed from end of array
    uv_array = three-column array of wavelength/flux/error
    """
    # trim uv_array to multiple of bin_size
    uv_len = len(uv_array)
    cut = uv_len%binsize
    new_len = uv_len//binsize 
    uv_cut = uv_array[:(-1*cut)] # Losing some data... up to binsize-1 points
    
    old_wavs = uv_cut[:,0]
    old_flux = uv_cut[:,1]
    old_errs = uv_cut[:,2]
    
    uv_smoothed = np.zeros((new_len,3))
    uv_smoothed[:,0] = np.asarray([np.mean(b, 0) for b in np.split(old_wavs, new_len)])
    uv_smoothed[:,1] = np.asarray([np.mean(b, 0) for b in np.split(old_flux, new_len)])
    #err = rms of errors
    uv_smoothed[:,2] = np.asarray([np.sqrt(np.sum(b**2))/binsize for b in np.split(old_errs, new_len)])
    
    return uv_smoothed


def show_spec_rebin():
    # plot rebinned spectrum
    plt.figure()
    plot_spectrum(uv_spec, l='COS raw')
    plot_spectrum(uv_smth, l='COS rebinned')
    axes = plt.gca()
    axes.set_xlim([1130, 1430])
    axes.set_ylim([-1e-15, 5.8e-14])
    plt.xlabel("Wavelength / $\AA$ ")
    plt.ylabel("Flux / erg/cm^2/s/A")
    plt.legend()
    plt.savefig('uv/plots/spec_smoothed.png', dpi=192)
    #plt.show()
    plt.close()


def show_spec_dered(model_spectrum):
    # plot dereddened/reddened COS/model spectra (resp)
    plt.figure()
    #plt.subplot('211')
    plot_spectrum(uv_smth, l='COS (rebinned)', c='blue')
    #plot_spectrum(uv_drd, l='COS (dereddened)', c='green')
    plot_spectrum(model_spectrum, l='Model (reddened)', c='orange')
    axes = plt.gca()
    axes.set_xlim([1130,1430])
    axes.set_ylim([0,6e-14])
    plt.legend()
    plt.show()


def figure_model_comparison(model, nam):
    """Create plot of model, uv_spec and photometry"""
    teff = nam[0][1:]
    logg = nam[1][1:]
    h_he = nam[2][1:]
    modelname = 't{}_g{}_h{}'.format(teff, logg, h_he)
    print(modelname)
    plt.figure()
    plt.title("T_eff = {} log(g) = {} [H/He] = {}".format(teff, logg, h_he))
    plt.subplot('211')
    plot_spectrum(uv_smth, l='COS (rebinned)')  
    plot_spectrum(model, l='Model', c='orange')
    axes = plt.gca()
    axes.set_xlim([1130,1430])
    axes.set_ylim([-1e-15,6e-14])
    plt.legend()
    # and photometry
    plt.subplot('212')
    plot_spectrum(model, c='orange')
    plot_fluxpoints(f_apass)
    plot_fluxpoints(f_panstarrs)
    #plt.text(4500, 1e-15,
    #        "$T_(EFF)=${}\nlog(g)={:.2d}\n[H/He]={:.2d}".format(model_name.split('_')))
        
    axes = plt.gca()
    axes.set_xlim([4250,9700])
    axes.set_ylim([0,3e-15])
    plt.legend(fontsize='small', ncol=2)
    #plt.show()
    plt.savefig('uv/plots/{}.png'.format(modelname),
                dpi = 128,
                format='png')
    plt.close()
    

def redu_chi_sq(mdlspec, modl_flux, nam):
    """Calculate chi-sq value for model, relative to observation
    Create linear interpolation data for each obs x
    Calc chi-sq_i, take mean
    
    Return chi squared per point"""
    # set observed spec to smoothed uv fluxes
    obsf = uv_smth[:,1]
    err_obs = uv_smth[:,2]
    mdlf = np.interp(uv_smth[:,0], mdlspec[:,0], mdlspec[:,1]) # create linear interpolated points in model
    resid = obsf - mdlf # calc residual values
    
    onechi = np.square(resid) / np.square(err_obs)
    lennotnan = len(onechi[~np.isnan(onechi)]) # no of non-nan values in onechi
    chisq = np.nansum(onechi)

    # now calculate same for panstarrs photometry FLUXES
    
    
    phot_data = np.asarray([x[:3] for x in f_panstarrs])
    print(phot_data)
    phot_pbnd = phot_data[:,0] # is a wavelength, not a passband
    phot_flux = phot_data[:,1]
    phot_errf = phot_data[:,2]

    #print([[x[0], x[-1]] for x in f_panstarrs])
    #print("Comparing {} and {}".format(phot_pbnd, modl_pbnd))
    #print("\n\n")
    #print(phot_flux)
    #print(modl_flux)
    #phot_resid = phot_flux - modl_flux
    #phot_mdlf = np.interp(phot_wavl, mdlspec[:,0], mdlspec[:,1])
    #phot_resid = phot_flux - phot_mdlf
    phot_onechi = np.square(phot_resid) / np.square(phot_errf)
    phot_lnn = len(phot_onechi)
    phot_chisq = np.sum(phot_onechi)
    
    print('{} vs {}'.format(chisq, phot_chisq))
    
    rchisq = (chisq + phot_chisq)/(lennotnan + phot_lnn)
    # returns chi squared per point that is not nan, including values from photometry
    return rchisq, chisq, phot_chisq


def synthetic_fluxes(spectrum, filters):
    """Calculate sythetic fluxes from spectrum for all filters given
    
    filters = list of filter info
        each line should be formatted as str(filter_survey)
    
    Create SED object from spectrum
    
    Returns list of 
    """
    vega_pb = ['b','v','U','B','V']
    ab_pb = ['g','r','i','z','y']
    print("Creating synthetic fluxes...")
    sed = astSED.SED(spectrum[:,0], spectrum[:,1])
    synths = []
    for f in filters:
        pband = PBD[str(f)]
        synthflx = sed.calcFlux(pband)
        #if f[0] in vega_pb:
            #synthflx = sed.calcMag(pband, addDistanceModulus=False, magType="Vega")
        #elif f[0] in ab_pb:
            #synthflx = sed.calcMag(pband, addDistanceModulus=False, magType="AB")
        synths.append(synthflx)
        
    synths = np.asarray([float(x) for x in synths])
    return synths


def manage_model(model_name, rescal, filts):
    """Overall manager function for model reddening, rescaling and comparison to uv spec
    Also needs to reject models that are not in 'shortlist' to run
    """
    model_red = rd.deredden_fix(get_dk(model_name), -1*EBV)[0] # Reddens model, also returns Av
    model_red[:,1] *= rescal # rescale model flux by given value AFTER REDDENING
    
    # create synthetic fluxes from model spectrum, for panstarrs filters
    model_fluxes = synthetic_fluxes(model_red, filts)
    #print(model_fluxes)
    
    rx2, s_vals, p_vals = redu_chi_sq(model_red, model_fluxes, model_name)
    figure_model_comparison(model_red, model_name.split('/')[-1][:-3].split('_'))
    # return details of fit
    return rx2, s_vals, p_vals


def RXSQ_lookup(temp, logg, hhe):
    """Look up unique rescaling value from 3 parameters"""
    ind_t = np.argwhere(RXSQ[:,0] == temp)
    ind_g = np.argwhere(RXSQ[:,1] == logg)
    int_h = np.argwhere(RXSQ[:,2] == hhe)
    rescaling = RXSQ_vals[np.intersect1d(int_t, np.intersect1d(int_g, int_h))][0][-1]
    return rescaling


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
        plot_BFV = 0
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
                if aval in best_fit and bval in best_fit and cval in best_fit:
                    plot_BFV = 1
                    BFVx = cval
                    BFVy = bval
        #print(plot_vals)
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
            plot_name = 'uv/plots/{}_{}_{}.png'.format(src, savenam, aval)
            plt.savefig(plot_name)
        if showit:
            plt.show()
        plt.close('all')
        print("{} = {} complete".format(savenam, aval))


def get_best_rxsqs(num=None):
    """Obtain best x values from file of reduced chi square values
    
    Return sub-array containing those values"""
    rxsq_array = np.genfromtxt('optical/chi_sqr/int_rchisq_params.dat', delimiter=',')
    ranked_inds = np.argsort(rxsq_array[:,3])
    ranked_rxsq = rxsq_array[ranked_inds]
    #print("RANKED \n", ranked_rxsq[:num])
    return ranked_rxsq[:num]


def mask_wavelengths(spec_array, masks):
    """For each mask range in masks, apply to wavelength column of array
    Then mask all rows with masked wavelengths
    
    Return masked array"""
    print("Masking input array in these ranges:")
    for rangepair in masks:
        rmin, rmax = rangepair
        spec_array[:,0] = ma.masked_inside(spec_array[:,0], rmin, rmax)
    spec_marray = ma.mask_rows(spec_array)
    return spec_marray


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
    array = vac2air(np.genfromtxt(fname, delimiter=' ')[:,:3])
    # mask as required
    masked = mask_wavelengths(array, MASK_RANGES)
    return masked


####    MAIN    ####

# CURRENTLY IMPORTING RESCALING VALUE FROM OPTICAL SPEC FIT
# REWRITE SO THAT RESCALED TO UV(+phot?) ONLY
# FIT FOR ALL MODELS, NOT ONLY BEST FROM OPTICAL SPEC FIT

uv_spec = open_uvarray('uv/APASSJ204713.82-125909.5_2017-11-04.dat')

p_flt, p_src, p_dat, f_dat, f_all = photometry_fluxes() # photometry filters, sources, data, flux data, flux all

f_all = [[float(x) for x in line[:-1]] + [line[-1]] for line in f_all] # reformat
f_apass = [[x for x in line[:-1]] + [line[-1]] for line in f_all if 'apass' in line[-1]] # separate apass flux data
f_panstarrs = [[x for x in line[:-1]] + [line[-1]] for line in f_all if 'panstarrs' in line[-1]] # separate panstarrs flux data
# save fluxes to dat file
with open('photometric_fluxes.dat', 'w') as fl_out:
    fl_wrt = csv.writer(fl_out, delimiter=',')
    fl_wrt.writerows(f_all)

# get array of best fit models from int
RXSQ = get_best_rxsqs(num=50)
# EDIT TO RUN OVER ALL VALUES

# create smoothed uV spectrum by resampling HST-COS spec
uv_smth = resample_uv(uv_spec, 5)

# create target list for new fit details
fit_vals = []

for line in RXSQ:
    mod_params = line[:3] # separate params from other data
    rescale = line[-1]
    # generate modelname
    mod_name = 'models/DAB/t{}_g{}_h{}.dk'.format(int(mod_params[0]),
                                                  int(mod_params[1]*100),
                                                  mod_params[2])
    # create output line
    rxsq_val, spec_vals, phot_vals = manage_model(mod_name, rescale, [x[-1] for x in f_panstarrs])
    fit_line = [y for y in mod_params] + [rxsq_val]
    fit_vals.append(fit_line)

# save rxsq results to file
fit_output = np.asarray(fit_vals)
np.savetxt('uv/chi_sqr/rxsq_params.dat',
           fit_output,
           delimiter=',',
           header='Teff,logg,[H/He],Red. chi-sq.',
           fmt=['%i', '%.2f', '%.2f', '%.6e'])

# produce contour plots
tf = fit_output[:,0]
lg = fit_output[:,1]
hh = fit_output[:,2]
xs = fit_output[:,3]
minxs = np.nanmin(fit_output[:, 3])
BFV = fit_output[np.argwhere(fit_output[:, 3]==minxs), :].copy()[0][0] # not sure why nested twice but it is
contour_bounds = None
contour_abc(lg, tf, hh, xs,
           'fxd_logg', 'T_eff [K]', '[H/He]', 'Fixed logg',
           contour_bounds, False, 'HST-COS', BFV)



