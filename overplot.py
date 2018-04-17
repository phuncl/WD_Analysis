"""Plot a series of spectra, for use in comparing line strengths"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate as spint

INTG_REGS = [(1290.0, 1292.5),
             (1313.5, 1316.0),
             (1325.5, 1328.0),
             (1354.5, 1357.0),
             (1371.0, 1373.5),
             (1388.0, 1390.5),
             (1417.8, 1420.3)]


def plot01(dat, labl=None, colour=None, lstyle=None, mark=None, a=1):
    """Streamlined plt.plot function for first two columns of array"""
    return plt.plot(dat[:,0], dat[:,1], label=labl, c=colour, ls=lstyle, marker=mark, alpha=a)


def subplot01(axisname, dat, labl=None, colour=None, lstyle=None, mark=None, a=1):
    return axisname.plot(dat[:,0], dat[:,1], label=labl, c=colour, ls=lstyle, marker=mark, alpha=a)


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
    #old_errs = uv_cut[:,2]
    
    uv_smoothed = np.zeros((new_len,2))
    uv_smoothed[:,0] = np.asarray([np.mean(b, 0) for b in np.split(old_wavs, new_len)])
    uv_smoothed[:,1] = np.asarray([np.mean(b, 0) for b in np.split(old_flux, new_len)])
    #err = rms of errors
    #uv_smoothed[:,2] = np.asarray([np.sqrt(np.sum(b**2))/binsize for b in np.split(old_errs, new_len)])
    
    return uv_smoothed


def vac2air(vac_spec):
    """Adjust to air wavelengths"""
    air_spec = np.copy(vac_spec)
    v = vac_spec[:,0]
    c1 = 1.0002735182
    c2 = 131.4182
    c3 = 2.76249 * np.power(10.0, 8.0)
    a = v / (c1 + c2/(v**2) + c3/(v**4))
    air_spec[:,0] = a
    return air_spec


def normalisation_manager(spectrum):
    """"""
    norm_pts = region_integrator(spectrum)

    # returns m, c coefficients of linear fit
    lin_fit = np.polyfit(norm_pts[:,0], norm_pts[:,1], 1)
    
    x=np.arange(1280, 1460, 10)
    #plot01(spectrum, labl='Original spectrum', a=0.2)
    #plot01(norm_pts, mark='o', lstyle='', labl='Normalisation points')
    #plt.plot(x, x*lin_fit[0] + lin_fit[1], label='Linear fit')
    # rescale spectrum by this fit
    #normalised = spectrum[np.where(np.logical_and(spectrum[:,0]>1280, spectrum[:,1]<1460))]
    normalised = np.copy(spectrum)
    normalised[:,1] /= (normalised[:,0]*lin_fit[0] + lin_fit[1]) # y=mx+c
    #plot01(new_spectrum, labl='Adjusted spectrum')
    #plt.legend()
    #plt.show()
    
    return normalised


def region_integrator(spec):
    """"""
    spectrum = np.copy(spec)
    flux_pts = []
    for wmin, wmax in INTG_REGS:
        wmid = (wmin+wmax)/2
        conds = np.logical_and(spectrum[:,0] > wmin, spectrum[:,0] < wmax)
        sub_spec = spectrum[np.where(conds)]
        flux_dens = integrate(sub_spec)
        flux_pts.append([wmid, flux_dens])
    return np.asarray(flux_pts)


def integrate(s):
    """Integrate a given two column (x,y) spectrum array"""
    spec_width = np.max(s[:,0]) - np.min(s[:,0])
    i = spint.trapz(s[:,1], s[:,0])
    fd = i/spec_width
    return fd


# open spectra files
apassj2047 = resample_uv(np.genfromtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat'))
wd0843 = resample_uv(np.genfromtxt('comparison_spectra/WD0843+516_2011-05-13.dat', delimiter=' ', usecols=(0,1)))
wd1015 = resample_uv(np.genfromtxt('comparison_spectra/WD1015+161_2011-04-01.dat', delimiter=' ', usecols=(0,1)))
wd1226 = resample_uv(np.genfromtxt('comparison_spectra/SDSS1228+1040_cos.dat', delimiter=' ', usecols=(0,1)))
wd1929 = resample_uv(np.genfromtxt('comparison_spectra/WD1929+011_2011-04-03.dat', delimiter=' ', usecols=(0,1)))

# create rescaled spectra
apassj2047 = normalisation_manager(apassj2047)
wd0843 = normalisation_manager(wd0843)
wd1015 = normalisation_manager(wd1015)
wd1226 = normalisation_manager(wd1226)
wd1929 = normalisation_manager(wd1929)


# Do we want to plot all at same time? Probably.
cont_lvl = np.asarray([[1100,1],[1500,1]]) # can replace this with axhline per axis

fig = plt.figure()
a1 = plt.subplot('221')
subplot01(a1, apassj2047, 'APASSJ2047', 'red', a=1)
subplot01(a1, wd0843, 'WD0843', 'green', a=.5)
subplot01(a1, cont_lvl, colour='grey') # axhline instead here
#alims = plt.gca()
#alims.set_xlim(1280, 1440)
#alims.set_ylim(0,2)
#plt.legend()

a2 = plt.subplot('222', sharex=a1, sharey=a1)
subplot01(a2, apassj2047, 'APASSJ2047', 'red', a=1)
subplot01(a2, wd1015, 'WD1015', 'blue', a=.5)
subplot01(a2, cont_lvl, colour='grey')
#plt.legend()

a3 = plt.subplot('223', sharex=a1, sharey=a1)
subplot01(a3, apassj2047, 'APASSJ2047', 'red', a=1)
subplot01(a3, wd1226, 'WD1226', 'purple', a=.5)
subplot01(a3, cont_lvl, colour='grey')
#plt.legend()

a4 = plt.subplot('224', sharex=a1, sharey=a1)
subplot01(a4, apassj2047, 'APASSJ2047', 'red', a=1)
subplot01(a4, wd1929, 'WD1929', 'cyan', a=.5)
subplot01(a4, cont_lvl, colour='grey')
a4.set_xlim(1280, 1440)
a4.set_ylim(0,2)
#plt.legend()

fig.text(0.5, 0.01, 'Wavelength $\AA$', ha='center')
fig.text(0., 0.5, 'Arbitrary Flux', va='center', rotation='vertical')
fig.tight_layout()
plt.show()
