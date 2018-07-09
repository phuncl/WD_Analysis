"""
New chi squared fitting procedure for uv spectrum
Includes Gaia distance to calculate radius as part of fit
"""

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import glob as g
import csv

from scipy import interpolate
from scipy import integrate as spint
from photometry import reddenings as rd

from astLib import astSED
from photometry import phot

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


integration_regions = (1155, 1325, 1360, 1395, 1420) #REDUNDANT

# most of these masked regions are the stronger metal lines, not well fit by the generic model
# their in/exclusion increases across all models
# however the same models comprise the top 10 fits in each case
masking_regions = ([1130.0, 1138.0], 
                   [1140.4, 1140.8],
                   [1142.2, 1142.8],
                   [1143.0, 1143.6],
                   [1144.8, 1145.3],
                   [1154.0, 1154.2],
                   [1214.0, 1217.2], # Ly alpha
                   [1253.7, 1254.1],
                   [1260.0, 1261.2],
                   [1263.9, 1266.2],
                   [1302.0, 1302.5], #sky
                   [1304.2, 1306.5], #sky
                   [1309.0, 1310.1], #sky
                   [1334.3, 1336.3],
                   [1346.6, 1347.1],
                   [1349.7, 1350.9],
                   [1352.5, 1352.9])

#masking_regions = ([1214.0, 1217.2],)



def photometrics():
    """"""
    with open('photometric_fluxes.dat', 'r') as fin:
        fread = csv.reader(fin, delimiter=',')
        dat = [[float(x) for x in line[:-1]] + [line[-1]] for line in fread]
    return dat


def array_inrange(arr, lo, hi):
    return arr[np.where(np.logical_and(arr[:,0]>lo, arr[:,0]<hi))]
    

def redshift_wavelengths(data_array, beta):
    """Apply redshift to column one of an array"""
    if len(data_array.shape) > 1:
        wavs = np.copy(data_array[:,0])
    else:
        wavs = np.copy(data_array)
    
    new_wavs = wavs * np.sqrt((1 + beta)/(1 - beta))
    data_array[:,0] = new_wavs
    return data_array


def lsf_conv(modl_array):
    """"""
    lsf_wavs, lsf_spreads = getlsf('COS_LSF/fuv_G130M_1291_lsf.dat', 3) # LSF FILENAME

    disp = 0.00997
    startw = lsf_wavs[0]
    endw = lsf_wavs[-1]
    startw*2
    #print(startw, endw, disp)
    wavelengths = np.arange(startw, endw, disp)
    fluxes = np.interp(wavelengths, modl_array[:,0], modl_array[:,1])[np.newaxis].T
    #print(fluxes.shape)

    # set up pixel dimension for convolution
    pix = np.arange(lsf_spreads.shape[1])
    # create 2d interpolation grid for LSF
    lsf_interp = interpolate.interp2d(pix, lsf_wavs, lsf_spreads)
    # interpolate onto grid of observed wavelength * pixel
    lsf_vals = lsf_interp(pix, wavelengths)
    # integrate each lsf and normalise
    lsf_i = np.trapz(lsf_vals, pix, axis=1).reshape(len(lsf_vals), 1)
    lsf_vals /= lsf_i
    #return wavelengths, lsf_vals
    
    # convolve
    #print('lsf_vals')
    #print(lsf_vals.shape)
    conv_spec = model_x_lsf(fluxes, lsf_vals)
    
    output_spec = np.stack((wavelengths, conv_spec), axis=1)
    return output_spec


def getlsf(lsf_filename, lp):
    """Read lsf file (lsfname), which is in format of lp2 onwards
    
    Returns:
    wavelengths  (shape n,)
    spread_functions (shape n,pix)
    """
    
    try:
        lsf_file = np.transpose(np.genfromtxt(lsf_filename, delimiter=''))
    except OSError:
        print('File {} not found! Exiting...'.format(lsf_filename))
    wavls = np.copy(lsf_file[:,0])
    spread_funcs = np.copy(lsf_file[:,1:])
    return wavls, spread_funcs


def fold_tails(flux_toolong, lsf_size):
    """Fold flux_toolong, to reflect tails into edge data
    
    flux_toolong = unfolded array
    lsf_size = size of line spread function ie number of pixels
    (should be the same for all lsfs if multiple used)
    
    Returns:
    flux_justright = tail-folded 1d flux array"""
    fl = lsf_size//2
    # separate tails of conv procedure
    lead_flux = flux_toolong[:fl]
    follow_flux = flux_toolong[-1*fl:]
    #follow_flux[:] = 0
    flux_justright = flux_toolong[fl:-1*fl]
    # add flipped tails
    flux_justright[:fl] += np.flip(lead_flux, axis=0)
    flux_justright[-1*fl:] += np.flip(follow_flux, axis=0)
    
    return flux_justright


def model_x_lsf(specflx, lsf):
    """Perform convolution of model with lsf profiles
    
    mflx = model flux array (1d)
    lsf = line spread functions (2d array; 'stack' of lsfs interpolated from file)"""
    
    if lsf.shape[0] != specflx.shape[0]:
        if lsf.shape[1] == specflx.shape[0]:
            lsf = np.transpose(lsf)
        else:
            print("LSF array length not equal to flux array length! Exiting...")
    
    # define lengths required, and lsf holder array
    lwavs, lpix = lsf.shape
    if lpix%2 == 0:
        print("Detected pixel width of LSF is not odd!\nExiting...")
    holder = np.zeros((lwavs, lwavs+lpix-1))
    
    # multiply each lsf by relevant flux
    fluxed_lsf = specflx * lsf
    # write into holder, along lead diagonal
    for l in range(lwavs):
        holder[l,l:l+lpix] = fluxed_lsf[l]

    # and sum holder along pix dimension, giving a 1d flux array with tails
    wide_flux = np.sum(holder, axis=0)
    
    #return wide_flux, lpix
    # fold tails in, to account for flux loss
    convolved_flux = fold_tails(wide_flux, lpix)
    
    return convolved_flux


def rescale_valcalc(m, s, plotit=False): # REDUNDANT
    """
    m = model array
    s = (observed) spectrum array
    """
    r = []
    for i in integration_regions:
        sub_m = array_inrange(m, i, i+5)
        m_flux = spint.trapz(sub_m[:,1], sub_m[:,0])#/(sub_m[-1,0] - sub_m[0,0])
        
        # mc sample flux
        sub_s = array_inrange(s, i, i+5)
        samples_s = np.random.normal(sub_s[:,1], sub_s[:,2], (10000, len(sub_s[:,1])))
        samples_int = spint.trapz(samples_s, sub_s[:,1], axis=1)
        s_flux = np.mean(samples_s)
        s_errflux = np.std(samples_s)
        
        #s_flux = spint.trapz(sub_s[:,1], sub_s[:,0])
        r.append(s_flux/m_flux)
        #r.append([s_flux/m_flux, s_errflux/s_flux])
    r = np.asarray(r)
    print('Rescaling values')
    print(r)
    if plotit:
        plt.figure()
        plt.plot(integration_regions, r[:,0])
        plt.show()
    
    resc = np.mean(r)
    errresc = np.std(r)
    
    #meanr = np.sum(r[:,0]/(r[:,1]**2))/np.sum(1/(r[:,1]**2))
    #errmeanr = np.sqrt(1/np.sum(1/(r[:,1]**2)))
    #return meanr, errmeanr
    return resc, errresc


def mask_wavelengths(spec_array, masks):
    """For each mask range in masks, apply to wavelength column of array
    Then mask all rows with masked wavelengths
    
    Return masked array"""
    #print("Masking input array in these ranges:")
    spec_array = ma.asarray(spec_array)
    for rangepair in masks:
        spec_array[:,0] = ma.masked_inside(spec_array[:,0], rangepair[0], rangepair[1])
    spec_marray = ma.mask_rows(spec_array)
    return spec_marray


def rx2_spec(mod_uv, uv_dat, mod_opt, opt_dat, mnam, masking=True, makeplot=False):
    """
    Calc reduced chi squared statistic for model interpolated to data wavelengths
    """
    # UV
    #spec_masked = ma.masked_inside(spec_dat, 1214.0, 1217.2) # mask Ly-alpha core
    if masking:
        #print('spectral masking applied')
        spec_masked = mask_wavelengths(uv_dat, masking_regions)
    else:
        spec_masked = uv_dat
    wavs = spec_masked[:,0]
    f_obs = spec_masked[:,1]
    errf_obs = spec_masked[:,2]
    f_mod = np.interp(wavs, mod_uv[:,0], mod_uv[:,1])
    
    residuls = f_obs - f_mod
    chivals = np.square(residuls) / np.square(errf_obs)
    
    rchisq = np.sum(chivals) / ma.count(chivals) # divide by unmasked points only!
    
    contribution_data = (residuls*abs(residuls)/np.square(errf_obs))
    contribution_mean = np.mean(contribution_data)

    
    # OPTICAL
    # any optical masking - blue end (<3850) ???
    opt_f_mod = np.interp(opt_dat[:,0], mod_opt[:,0], mod_opt[:,1])
    opt_resids = opt_dat[:,1] - opt_f_mod
    opt_chivals = np.square(opt_resids) / np.square(opt_dat[:,2])
    opt_rchisq = np.sum(opt_chivals) / len(opt_chivals)
    
    big_rchisq = (np.sum(chivals) + np.sum(opt_chivals))/(ma.count(chivals) + len(opt_chivals))
    
    opt_contribs = (opt_resids*abs(opt_resids)/np.square(opt_dat[:,2]))
    opt_contrib_mean = np.mean(opt_contribs)
    
    # PLOT
    if makeplot:
        f, ax = plt.subplots(2,2, figsize=(16,6), sharex='col')
        ax[0][0].plot(wavs, f_obs, 'k', lw=0.5)
        ax[0][0].plot(wavs, f_mod, 'r', lw=1)
        ax[0][0].set_title('UV $\chi^2$ = {:.3f}'.format(rchisq))
        ax[1][0].plot(wavs, contribution_data, color='black', lw=0.5)
        ax[1][0].axhline(0, color='red', lw=1)
        ax[1][0].set_title('Residuals * Residuals / $\sigma^2$, mean = {:.3f}'.format(contribution_mean))
        #ax[1].text(1190, 50 *contribution_mean, 'Mean = {:.3f}'.format(contribution_mean))
        
        #f, ax = plt.subplots(2,1, figsize=(10,6), sharex=True)
        ax[0][1].plot(opt_dat[:,0], opt_dat[:,1], 'k', lw=0.5)
        ax[0][1].plot(opt_dat[:,0], opt_f_mod, 'r', lw=1)
        ax[0][1].set_title('Optical $\chi^2$ = {:.3f}'.format(opt_rchisq))
        ax[1][1].plot(opt_dat[:,0], opt_contribs, color='black', lw=0.5)
        ax[1][1].axhline(0, color='red', lw=1)
        
        ax[1][1].set_title('Residuals * Residuals / $\sigma^2$, mean = {:.3f}'.format(opt_contrib_mean))
        #ax[1][1].text(3600, 50*opt_contrib_mean, '')
        
        plt.suptitle('Combined $\chi^2$={:.5f} for {}'.format(big_rchisq, mnam.replace('_', ' ')))
        #plt.show()
        plt.savefig('chisqr-results/plots/sgrid2018/uv+opt_{}.pdf'.format(mnam), dpi = 128, format='pdf')
        plt.close('all')
    
    return rchisq, opt_rchisq, big_rchisq


def model_to_data(model_name, uv, optical):
    """Fit model to data
    Calculate a rescaling value and a chi squared"""
    # open model
    modl = np.genfromtxt(model_name)    
    modl[:,1] *= 0.25e-8 # translate to Eddington flux
    
    # lsf convolution
    modl_hst = lsf_conv(modl) # uv
    # any optical convolution?!
    
    # redden by 0.01
    modl_hst, Av = rd.deredden_fix(redshift_wavelengths(modl_hst, BETA), -0.01)
    modl_opt, Av = rd.deredden_fix(redshift_wavelengths(modl, BETA), -0.01)
    
    # model details
    mdets = model_name.split('/')[-1][4:-3] # skip pre-/suffix
    teff = float(mdets.split('_')[0])
    logg = float(mdets.split('_')[1].split('H')[0])
    hyhe = float(mdets.split('H')[-1])
    outnam = 'chi2_t{}_g{}_h{}'.format(teff, logg, hyhe)
    #print(teff, logg, hyhe)
    
    # get model-specific mass/radius, calculate rescaling from radius
    model_m = mgrid(teff, logg)
    model_r = rgrid(teff, logg)
    #print('Radius = {}'.format(model_r[0]))
    
    # calculate rescaling value
    s_val = 4*np.pi * model_r**2 / (D*pc2cm)**2
    errs_val = 2*4*np.pi * model_r**2 * errD/D
    # WHAT TO DO WITH THIS ERROR
    
    # rescale model
    modl_hst[:,1] *= s_val
    modl_opt[:,1] *= s_val
    
    #plt.plot(spectrum[:,0], spectrum[:,1])
    #plt.plot(modl_scaled[:,0], modl_scaled[:,1])
    
    # calculate a chi square value for the fit
    rx2 = rx2_spec(modl_hst, uv, modl_opt, optical, outnam, makeplot=True)
    return [teff, logg, hyhe, rx2[0], rx2[1], rx2[2]]
    

def read_massradius(file_loc):
    """
    Read mass-radius grid file, to interpolate onto later
    """
    dat = np.genfromtxt(file_loc, names=True)
    t = dat['Teff']
    log = np.array([7.0,7.5,8.0,8.5,9.0,9.5])
    
    #massgrida = np.stack((dat[:,1], dat[:,3], dat[:,5], dat[:,7], dat[:,9], dat[:,11]), axis=1)
        # transposed massgrid_data because of input format for inter2d
    massgrid_data = np.stack((dat['m7_0'], dat['m7_5'], dat['m8_0'], dat['m8_5'], dat['m9_0'], dat['m9_5']), axis=1).T
    massgrid = interpolate.interp2d(t, log, massgrid_data, kind='cubic')
    
    radgrid_data = np.stack((dat['r7_0'], dat['r7_5'], dat['r8_0'], dat['r8_5'], dat['r9_0'], dat['r9_5']), axis=1).T
    radgrid = interpolate.interp2d(t, log, radgrid_data, kind='cubic')
    
    return massgrid, radgrid
    

####    CONSTANTS

prlx = 0.01017
errprlx = 0.00008
D = 1/prlx
errD = D * errprlx/prlx
D += errD * 0

pc2cm = 3.086e18 # 1 pc in cgs
Rsun = 6.96e10 # Solar radius in cgs
G = 6.674e-08 # cgs

RV = 35000 # km/s away
RV += 0
BETA = RV/3e8
EBV = 0.01

PBD = rd.psb_dict # passband file locations


####    MAIN

spec_uv = np.genfromtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat', usecols=(0,1,2))
photometry = photometrics()


####    OPTICAL
int_raw = np.genfromtxt('optical/J2047-1259_201505_int.dat')
int_min, int_max = min(int_raw[:,0]), max(int_raw[:,0])

# need to rescale int spectrum to photometry 
# only filters encompassed by spectrum
int_filters = []
for i in photometry:
    f = i[-1]
    pband = PBD[f]
    pbandmin = min([x[0] for x in pband.asList() if x[1]>0.00005])
    pbandmax = max([x[0] for x in pband.asList() if x[1]>0.00005])
    if (pbandmin > int_min) and (pbandmax<int_max): int_filters.append(i)

int_sed = astSED.SED(int_raw[:,0], int_raw[:,1])
f_ratios = []
for filt in int_filters:
    pband = PBD[filt[-1]]
    flx = filt[1]
    flxerr = filt[2]
    
    int_flx = int_sed.calcFlux(pband)
    flx_r = flx/int_flx
    flx_rerr = flxerr/int_flx
    f_ratios.append([flx_r, flx_rerr])

f_ratios = np.asarray(f_ratios)
   
mean_numer = np.sum(f_ratios[:,0] / np.square(f_ratios[:,1]))
mean_denom = np.sum(1 / np.square(f_ratios[:,1]))
resc_fac = mean_numer / mean_denom
resc_facerr= np.sqrt(1 / mean_denom)

int_spec = np.copy(int_raw)
int_spec[:,1] *= resc_fac
# rescaling includes error
int_spec[:,2] = int_raw[:,1] * np.sqrt((int_raw[:,2]*resc_fac/int_raw[:,1])**2 + (int_raw[:,1]*resc_facerr/resc_fac)**2) 
int_bot, int_top = min(int_spec[:,1]), max(int_spec[:,1])
np.savetxt('optical/int_photrescaled.dat', int_spec)


# plot optical rescaling result
"""
colrs = ('b', 'g', 'c', 'm', 'y', 'r')
f, (a1, a2) = plt.subplots(2,1, sharex=True)
# show int spec, rescaled, and photometry points used, and in panel below show filter profiles
a1.plot(int_raw[:,0], int_raw[:,1], 'grey', lw=1) # int spec
a1.plot(int_spec[:,0], int_spec[:,1], 'k', lw=1)# int spec rescaled by phot
a1.plot(test_modl[:,0], test_modl[:,1], 'r') # model, rescaled by radius/distance factor
for pos, f in enumerate(int_filters):
    farray = np.asarray(PBD[f[-1]].asList())
    #a1.errorbar(f[0], f[1], f[2], fmt='.', color=colrs[pos], label=f[-1].replace('_', ' '))
    a1.plot(f[0], f[1], 'o', ls='-', color=colrs[pos], label=f[-1].replace('_', ' '))
    a2.plot(farray[:,0], farray[:,1], color=colrs[pos], label=f[-1].replace('_', ' '))
    a1.set_xlim(3500, 7200)
    a1.set_ylim(0, 5e-15)
    a2.set_ylim(0, 0.00135)

# SHOW LEGEND
a1.legend()
#a2.legend()
plt.subplots_adjust(hspace=0)
a1.tick_params(which='both', direction='in', top=True, right=True)
a2.tick_params(which='both', direction='in', top=True, right=True)
plt.show()
"""

mgrid, rgrid = read_massradius('mass_radius/mr.dat')

gridpath = 'models/subgrid2018/'
modl_grid = g.glob(gridpath+'*.dk')


#rx2_data = np.empty((len(modl_grid), 4))
rx2_data = []
i=1
for m in modl_grid:
    rx2_data.append(model_to_data(m, spec_uv, int_spec))
    print('{:.2f}%'.format(100*i/len(modl_grid)))
    i+=1
    
# save rx2_data to file
save_dest = 'chisqr-results/sgrid2018-fitvals_uv+opt_d{:.3f}_b{}.dat'.format(D, int(RV))
print("Saving output data to {}".format(save_dest))
#input("Press Ctrl + C to abort this process, or Return to continue.".format(save_dest))

np.savetxt(save_dest, rx2_data, delimiter=',',
           header='Teff,logg,H/He,UV_Rchisq,OPT_Rchisq,TOTAL_Rchisq',
           fmt=['%i', '%.2f', '%.2f', '%.6e', '%.6e', '%.6e'])


"""
####    TEST MODEL
test_modl_name = 'models/grid2018/DBA_18100.0_8.1H-1.0.dk' # (best fit)
test_modl = np.genfromtxt(test_modl_name)
test_modl[:,1] *= 0.25e-8 # translate to Eddington flux
#test_modl = lsf_conv(test_modl)
test_modl, Av = rd.deredden_fix(redshift_wavelengths(test_modl, BETA), -0.01)


# rescale model as appropriate by distance
modl_m = mgrid(18100, 8.1)
modl_r = rgrid(18100, 8.1)
s = 4*np.pi * modl_r**2 / (D*pc2cm)**2
errs = 2*4*np.pi * modl_r**2 * errD/D
test_modl[:,1] *= s

#x, y = lsf_conv(test_modl)
t = model_to_data(test_modl_name, np.copy(spec_uv), int_spec)
"""
