"""Spectral plotting tool.

This version is designed for one object and will later be generalised.
"""

import numpy as np
import os
import glob as g
import csv
import matplotlib.pyplot as plt

from astLib import astSED
from photometry import reddenings as rd
from photometry import phot


# constants
EMPTY = -999		# value for unmeasured photometry CAREFUL
PBD = rd.psb_dict

class photometry_source:
	"""Generic class of photometry source
	
	Takes line of photometry data and survey name.
	
	survey = suvey from which data were taken
	photometry = stored photometric data from survey
	fluxes = (flux, flux_err, central_wavelength) for each used filter
	filters = filters with data
	
	"""
	def __init__(self, phot_line, surv):
		self.survey = surv
		self.photometry = {}
		self.fluxes = {}
		self.magnitudes = {}
		# phot line contains mag, errmag, ...(repeated)
		# calculate flux data and store in dict
		for i, val in enumerate(phot_line):
			if not i%2:
				if val != EMPTY:
					self.magnitudes[phot_FILTER[i]] = [phot_line[i], phot_line[i+1]]
					self.photometry[phot_FILTER[i]] = [val, phot_line[i+1]]
					print("Calculating flux data from following data:")
					print(phot_line[i], phot_line[i+1], phot_FILTER[i], surv, "\n")
					# Some consideration for EMPTY values?
					self.fluxes[phot_FILTER[i]] = phot.return_fluxes(phot_line[i],
																	 phot_line[i+1],
																	 phot_FILTER[i],
																	 surv)
				elif val == EMPTY:
					pass
				else:
					print("Something really weird has happened with value {}!".format(val))


def get_datfiles(target_dir):
	"""Return a list of .dat files from target_dir."""
	if not os.path.isdir(target_dir):
		print("{} not found.".format(target_dir))
		return None
	flist = g.glob('{}/*.dat'.format(target_dir))
	print("Collected {0} files from {1}\n.".format(len(flist), target_dir))
	return flist


def get_spectra(spec_list):
	"""Return a dictionary of spectra data arrays keyed by filename."""
	if not spec_list:
		return None
	spec_dict = {}
	for spec in spec_list:
		spec_dict[spec] = np.genfromtxt(spec)
	return spec_dict


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
					selected_fluxes.append([src_flux[key][2],
										  src_flux[key][0],
										  src_flux[key][1],
										  src_mag[key][0],
										  src_mag[key][1],
										  str(key)+'_'+str(k)])
			except TypeError:
				pass
	return selected_fluxes


def minmaxandflux(spectral_data, photometric_data):
	"""From a spectrum and a photometry set determine relevant photometry
	
	Return min wavelength, max wavelength, seleceted photometry"""
	minw = spectral_data[:,0].min()
	maxw = spectral_data[:,0].max()
	selec_phot = flux_in_wav_range(minw, maxw, photometric_data)
	return minw, maxw, selec_phot


def plot_spectrum(spectrum, c=None, l=None, style='-'):
	plt.plot(spectrum[:,0], spectrum[:,1], color=c, label=l, alpha=0.6, ls=style)


def plot_fluxpoints(fluxpoints):
	for line in fluxpoints:
		plt.errorbar(line[0], line[1], line[2], label = line[-1], fmt = 'o')


def synthetic_mags(spectrum, filters):
	"""Calculate sythetic magnitudes from spectrum for all filters given
	
	filters = list of filter info
		each line should be formatted as str(filter_survey)
	
	Create SED object from spectrum
	"""
	vega_pb = ['b','v','U','B','V']
	ab_pb = ['g','r','i','z','y']
	print("Creating synthetic magnitudes...")
	sed = astSED.SED(spectrum[:,0], spectrum[:,1])
	synths = []
	for f in filters:
		pband = PBD[str(f)]
		if f[0] in vega_pb:
			synthmag = sed.calcMag(pband, addDistanceModulus=False, magType="Vega")
		elif f[0] in ab_pb:
			synthmag = sed.calcMag(pband, addDistanceModulus=False, magType="AB")
		synths.append([synthmag, f])
	return synths


def compare_mags(Msynthetic, Mphotometric):
	"""Compare set of synthetic mags to set of photometric mags
	Compute rescaling factor for each pair
	# Check wavelength dependence?
	
	Return avg rescaling factor
	"""
	# CURRENTLY NO ERRORS
	print("Calculating magnitude scaling.")
	mag_dict = {}
	del_m = []
	for l in Msynthetic:
		mag_dict[l[-1]] = [l[0]] # initialise mag_dict with synthetic mag value
	for m in Mphotometric:
		mag_dict[m[-1]].append(m[0]) # append photometric magnitude
	for v in mag_dict.values():
		del_m.append(v[0] - v[1])
	dm = np.average(np.asarray(del_m))
	print("Average magnitude difference = {}".format(dm))	
	# f2/f1 = 100**(dm/5)
	return 100**(dm/5)


def rescale_spectrum(spectrum_dat, scaling):
	"""Rescale spectral flux (and errors) by given scaling factor (and errors)
	
	Return new_spectrum (same shape as spectrum_dat)
	"""
	new_spectrum = np.zeros(np.shape(spectrum_dat))		# copy original array
	new_spectrum[:,0] = spectrum_dat[:,0]						# want to keep wavelengths!
	new_spectrum[:,1] = spectrum_dat[:,1] * scaling		# rescale flux
	new_spectrum[:,2] = new_spectrum[:,1] * abs((spectrum_dat[:,2]/spectrum_dat[:,1]))
	return(new_spectrum)
	

##########  MAIN  ##########

# Get spectra
uv_spec = get_spectra(get_datfiles('uv/'))
opt_spec = get_spectra(get_datfiles('optical/'))

# Get photometric data and convert to flux
phot_FILTER, phot_source, phot_data= read_photometry('photometry.csv')

# generate flux data
flux_data = {}
for i,s in enumerate(phot_source):
	flux_data[s] = photometry_source(phot_data[i], s)
all_phot_flux = flux_in_wav_range(0,1e10,flux_data)

# spectral data for plotting
cos_dat = uv_spec['uv/APASSJ204713.82-125909.5_2017-11-04.dat']
ntt_dat = opt_spec['optical/J2047-1259_20140613_ntt.dat'] 
int_dat = opt_spec['optical/J2047-1259_201505_int.dat']

# create flux sets (for plotting and for rescaling process)
uvmin, uvmax, uv_fluxes = minmaxandflux(cos_dat, flux_data)
nttmin, nttmax, ntt_fluxes = minmaxandflux(ntt_dat, flux_data)
intmin, intmax, int_fluxes = minmaxandflux(int_dat, flux_data)
all_fluxes = flux_in_wav_range(uvmin, nttmax, flux_data)

# plot UV
plt.figure('Flux data')
plt.subplot('211')
plot_spectrum(cos_dat, 'b', 'COS')
plot_fluxpoints(uv_fluxes)
axes = plt.gca()
axes.set_ylim([-1e-15,5.8e-14])
plt.xlabel("Wavelength / $\AA$ ")
plt.ylabel("Flux / erg/cm^2/s/A")
plt.legend()

# plot Optical
plt.subplot('212')
plot_spectrum(int_dat, 'r', 'INT')
plot_spectrum(ntt_dat, 'g', 'NNT')
plot_fluxpoints(ntt_fluxes)
plt.xlabel("Wavelength / $\AA$ ")
plt.ylabel("Flux / erg/cm^2/s/A")
plt.legend()
# show plot
plt.show()

# create ntt synthetic magnitudes, compare to photometry, rescale
ntt_synth = synthetic_mags(ntt_dat, [x[-1] for x in ntt_fluxes])
ntt_scaling = compare_mags(ntt_synth, [x[3:] for x in ntt_fluxes]) # passing synth mags + photometry
ntt_rescaled = rescale_spectrum(ntt_dat, ntt_scaling)
print("NTT scaling factor = {}".format(ntt_scaling))
# same for int
int_synth = synthetic_mags(int_dat, [x[-1] for x in int_fluxes])
int_scaling = compare_mags(int_synth, [x[3:] for x in int_fluxes]) # passing synth mags + photometry
int_rescaled = rescale_spectrum(int_dat, int_scaling)
print("INT scaling factor = {}".format(int_scaling))

# plot both rescaled spectra
plt.figure('Optical Rescaling')
plot_spectrum(ntt_dat, 'g', 'NTT original')
plot_spectrum(ntt_rescaled, 'g', 'NTT rescaled', '--')
plot_spectrum(int_dat, 'r', 'INT original')
plot_spectrum(int_rescaled, 'r', 'INT rescaled', '--')
plot_fluxpoints(ntt_fluxes)
plt.xlabel("Wavelength / $\AA$ ")
plt.ylabel("Flux / erg/cm^2/s/A")
plt.legend()
plt.savefig("Optical_rescaling.png",bbox_inches='tight')
plt.show()

# fit H/He from normalised absorption line regions of optical spectra - HOW TO WEIGHT POINTS
# first need to look at model data


# fit Teff, logg from uv spec + photometry
# plot Teff logg for each H/He 

