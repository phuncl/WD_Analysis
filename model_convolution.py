"""Convolve a COS-LSF with a model spectrum

More description
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
from astropy.modeling import models, fitting
from scipy import integrate as spint

# possible convolution tools
from astropy.convolution import convolve
from scipy import ndimage, interpolate


INTG_REGIONS = [[1155., 1190.], # for rescaling continuum
                [1235., 1260.],
                [1355., 1425.]]


def get_dk(fname, showplot=False):
    """Utility - open .dk model file """
    d = np.genfromtxt(fname, skip_header=52)
    if showplot:
        plt.figure()
        plot01(dat, colour='r')
        plt.show()
    return d


def get_lsf_lp3(fname, dlim=''):
    """Open and read lsf for lifetime position 3"""
    # could allow reading of any lifetime pos by variable input - from fits file
    d_array = np.transpose(np.genfromtxt(fname, delimiter=dlim))
    w = np.copy(d_array[:,0]) # wavelengths given in lsf file
    r = np.copy(d_array[:,1:]) # response function 
    # by pixel (row is same pixel) for each wavelength (column is same wavelength)
    return w, r


def fold_edges(flux_toolong, lsf_size):
    """'Fold' array to reflect tails into edge data
    
    flux_toolong = unfolded array
    lsf_size = size of line spread function ie a number of pixels
    (should be the same for all lsfs if multiple used)"""
    fl = lsf_size//2
    lead_flux = np.flip(flux_toolong[:,:fl], axis=1)
    follow_flux = np.flip(flux_toolong[:,-1*fl:], axis=1)
    flux_justright = flux_toolong[:,fl:-1*fl]
    flux_justright[:,:fl] += lead_flux
    flux_justright[:,-1*fl:] += follow_flux
    
    return flux_justright # because of plotting format


def rescale_continuum(wavs, obs_f, mod_f):
    """Rescale mod_f to match obs_f based on 3 integration regions (0th order rescaling)"""
    obs_f = obs_f.flatten()
    mod_f = mod_f.flatten()
    rescales = []
    for l,u in INTG_REGIONS:
        conds = np.where(np.logical_and(wavs>=l, wavs<=u))
        obs_i = spint.trapz(obs_f[conds])
        mod_i = spint.trapz(mod_f[conds])
        rescales.append(mod_i/obs_i)
    r = np.mean(np.array(rescales))
    return np.copy(mod_f)/r


def apply_redshift(beta, wavs, plot=0):
    """Redshift spec by beta = v/c"""
    if plot:
        plt.figure()
        plt.plot(wavs, c='red')
    wavs += wavs * (np.sqrt(1+beta)/np.sqrt(1-beta) - 1)
    if plot:
        plt.plot(wavs, c='green')
        plt.show()
    return wavs


def model_x_lsf(mflx, lsf, plots=False):
    """Perform convolution of model with lsf profiles
    
    mflx = model flux array (1d)
    lsf = line spread functions (2d array; 'stack' of lsfs)"""
    # reshape mlfx and transpose lsf if needed
    if len(mflx.shape)==1:
        mflx.shape = (mflx.shape[0],1)
    
    if lsf.shape[0] != mflx.shape[0]:
        if lsf.shape[1] == mflx.shape[0]: lsf = np.transpose(lsf)
        else: print("WARNING\nLSF array length is not equal to model flux array length!\n")
    
    # define lengths required, and holder array
    lwavs, lpix = lsf.shape
    if lpix%2 == 0: print("WARNING\nDetected pixel width of LSF is not odd!\n") # so...
    holder = np.zeros((lwavs, lwavs+lpix-1))
    holder_response = np.zeros((lwavs, lwavs+lpix-1))
    
    # 'spread' fluxes across lsfs
    fluxed_lsf = mflx * lsf
    
    # write into holder and sum
    for l in range(lwavs):
        holder[l,l:l+lpix] = fluxed_lsf[l]
        holder_response[l,l:l+lpix] = lsf[l]
    wider_flux = np.sum(holder, axis=0) # wider_flux is FLAT (1D) 
    resp_flux = np.sum(holder_response, axis=0)
    
    lf = np.flip(resp_flux[:lpix//2 ], axis=0)
    ff = np.flip(resp_flux[lpix//2 *-1 :], axis=0)
    
    nf_0 = resp_flux[lpix//2 : lpix//2 *-1]
    nf = np.copy(nf_0)
    nf[:lpix//2 ] += lf
    nf[lpix//2 *-1 :] += ff
    
    if plots and 0==1:
        fig, [ax1, ax2] = plt.subplots(2,1, figsize=(10,8))
        ax1.plot(mdlw, nf_0, label='Unfolded response fn')
        ax1.plot(mdlw, nf, label='Folded response fn')
        plt.legend()
        #a.axvline(lsf_wavs.min(), c='grey') # mark min and max wavelength given in lsf grid
        #a.axvline(lsf_wavs.max(), c='grey')
        fig.suptitle('Response Function of LSFs (Linear Interpolation)')
        #ax2.plot(mdlw, nf_0, '.')
        ax2.plot(mdlw, nf, '+', mew=0.5)
        ax2.set_xlim(1260, 1290)
        ax2.set_ylim(0.9975, 1.0005)
        plt.show(block=False)

    # fold in lsf wings
    lead_flux = np.flip(wider_flux[:lpix//2], axis=0)
    follow_flux = np.flip(wider_flux[lpix//2 *-1:], axis=0)
    new_flux = wider_flux[lpix//2:lpix//2 *-1]

    new_flux[:lpix//2] += lead_flux
    new_flux[lpix//2 *-1:] += follow_flux
    if plots:
        f, a = plt.subplots()
        a.plot(cos_wavs, mflx, lw=2, c='r', label='Pre-convolution')
        a.plot(cos_wavs, new_flux, lw=1, c='g', label='Post-convolution')
        #plt.legend()
        #plt.title('Spectrum before/after LSF convolution')
        a.set_xlabel('Wavelength $\AA$')
        a.set_ylabel('Relative Intensity')
        a.set_xlim(1162.5,1181)
        a.set_ylim(0.25,1.25)
        a2 = plt.axes([0.57,0.58,0.3,0.25])
        a2.plot(cos_wavs, mflx, lw=2, c='r')
        a2.plot(cos_wavs, new_flux, lw=1, c='g')
        a2.set_xlim(1164.4,1164.9)
        a2.set_ylim(0.65,1.05)
        a2.set_xticks([])
        a2.set_yticks([])
        f.show()
    
    return new_flux, [nf, nf_0, resp_flux]


def save_model(fdata):
    """"""
    print('Now saving your file.')
    sn = input('Give path/to/filename of save file, (leave blank to skip saving):\n')
    if sn == '':
        print('Skipping file save process.')
        return
    np.savetxt(sn, fdata, delimiter=' ')
    print('File saved at {}'.format(sn))


####   MAIN    ####
fname = sys.argv[1]

lsf_wavs, lsf_resps = get_lsf_lp3('COS_LSF/fuv_G130M_1291_lsf.dat')

test_model = get_dk(fname)
test_model[:,1] *= 1.0e-17 # rescale to ~1

test_model[:,0] = apply_redshift(35/3e5, np.copy(test_model[:,0])) # reshift model wavelengths

cos_spec = np.genfromtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat', usecols=(0,1))
cos_flx = np.copy(cos_spec[:,1]) *1.0e15 # rescale numbers to ~1
cos_wavs = np.copy(cos_spec[:,0]) # individual wavelengths measured ie separate pixels

# redshift model, before interpolationg (no flux stretching)
#mdlw = apply_redshift(37e3/3e8, np.copy(cos_wavs)) # redshift so modelled wav = measured wav
mdlw = np.copy(cos_wavs)

mdlf = np.interp(mdlw, test_model[:,0], test_model[:,1]) # interpolate flux vals at cos wavelengths
pix = np.arange(0, len(lsf_resps[0])) # pixel dimension

lsf_interp = interpolate.interp2d(pix, lsf_wavs, lsf_resps, kind='linear')
# NOTE outside range this is flat - probably fine as only edges of spectrum and lsf quite smooth

lsf_vals = lsf_interp(pix, cos_wavs)
lsf_i = np.trapz(lsf_vals, pix, axis=1).reshape(len(lsf_vals), 1)
lsf_vals /= lsf_i # normalised linespread functions (row = fixed wavelength)

convol_mdl, tst = model_x_lsf(mdlf, lsf_vals, plots=1)

output_model = np.stack((mdlw, convol_mdl), axis=1)

#save_model(output_model)


########

#tflx = np.ones(len(mdlw))
#tgauss = models.Gaussian1D(amplitude=1, mean=50, stddev=5)
#tpix = np.arange(101)
#tgrid = np.tile(tgauss(tpix), (len(mdlw),1))
#tgrid = tgrid/(np.trapz(tgrid, tpix).reshape(len(mdlw),1))
#outf, outresp = model_x_lsf(tflx, tgrid, plots=1)

#mdlw = np.arange(1000,2000)
#tflx = np.ones(1000)
#tgauss = models.Gaussian1D(amplitude=1, mean=50, stddev=5)
#tpix = np.arange(101)
#tgrid = np.tile(tgauss(tpix), (1000,1))
#tgrid = tgrid/(np.trapz(tgrid, tpix).reshape(1000,1))
#outf, outresp = model_x_lsf(tflx, tgrid, plots=0)

#tgauss2 = models.Gaussian1D(amplitude=1, mean=50, stddev=6)
#tgauss3 = models.Gaussian1D(amplitude=1, mean=50, stddev=3)
#tgauss4 = models.Gaussian1D(amplitude=1, mean=50, stddev=1.5)
#tgauss5 = models.Gaussian1D(amplitude=1, mean=50, stddev=0.8)
#tgauss6 = models.Gaussian1D(amplitude=1, mean=50, stddev=0.3)
#tgauss7 = models.Gaussian1D(amplitude=1, mean=50, stddev=0.1)
#tinterp = interpolate.interp2d(tpix, np.array([1100, 1250, 1400, 1500, 1600, 1750, 1900]), np.array([tgauss(tpix), tgauss2(tpix), tgauss3(tpix), tgauss4(tpix), tgauss5(tpix), tgauss6(tpix), tgauss7(tpix)]), kind='linear')
#tinterpc = interpolate.interp2d(tpix, np.array([1100, 1250, 1400, 1500, 1600, 1750, 1900]), np.array([tgauss(tpix), tgauss2(tpix), tgauss3(tpix), tgauss4(tpix), tgauss5(tpix), tgauss6(tpix), tgauss7(tpix)]), kind='cubic')
#tinterpq = interpolate.interp2d(tpix, np.array([1100, 1250, 1400, 1500, 1600, 1750, 1900]), np.array([tgauss(tpix), tgauss2(tpix), tgauss3(tpix), tgauss4(tpix), tgauss5(tpix), tgauss6(tpix), tgauss7(tpix)]), kind='quintic')
#tgrid2 = tinterp(tpix, mdlw)
#tgrid2 = tgrid2/(np.trapz(tgrid2, tpix).reshape(1000,1))
#outf2, outresp2 = model_x_lsf(tflx, tgrid2, plots=0)
#tgrid3 = tinterpc(tpix, mdlw)
#tgrid3 = tgrid3/(np.trapz(tgrid3, tpix).reshape(1000,1))
#outf3, outresp3 = model_x_lsf(tflx, tgrid3, plots=0)
#tgrid4 = tinterpq(tpix, mdlw)
#tgrid4 = tgrid4/(np.trapz(tgrid4, tpix).reshape(1000,1))
#outf4, outresp4 = model_x_lsf(tflx, tgrid4, plots=0)

# still need to rescale mdlf, convol_mdl to continuum level of cos
"""
plt.figure()
plt.plot(cos_wavs, cos_flx, c='grey', alpha=0.5, label='COS')
plt.plot(mdlw, mdlf, c='red', label='Model')
plt.plot(mdlw, convol_mdl, c='green', label='Model * LSF')
ax=plt.gca()
ax.set_ylim(-2, 60)
plt.legend()
plt.show()


#############
# create plot of every 20th LSF, stacked, to see any evolution

#plt.figure()
#for i in range(0, len(lsf_resps), 20):
    #plt.plot(pix[140:181], lsf_resps[i,140:181] + i/20, label='{}$\AA$'.format(lsf_wavs[i]))
#plt.legend()
#plt.axvline(x=160)
#plt.show(block=False)    

#############
# test cases

# BASIC CONVOLUTION
# fit gaussian to a sample LSF, so values are approx right
eglsf = np.copy(lsf_resps[0]) # data for singular gaussian fit over relative pixel space

g_i = models.Gaussian1D(amplitude=1, mean=160, stddev=3)
fit_g = fitting.LevMarLSQFitter()
g_f = fit_g(g_i, pix, eglsf)

gaussian = g_f(pix)
# SHOULD NORMALISE THIS. SIMILAR FOR LSF
G = spint.trapz(gaussian, pix)
gaussian /= G

conv_model = np.convolve(mdlf.reshape(28434,), gaussian) # THIS INCLUDES TAILS from convolution (ie becomes longer)
cm = fold_edges(np.copy(conv_model.reshape((1, conv_model.shape[0]))), gaussian.shape[0])

#plt.figure()
#plt.plot(cos_wavs, mdlf, label='Raw Model')
#plt.plot(cos_wavs, conv_model[160:-160], label='Convolved model')
#plt.plot(cos_wavs, cm, label='Tail-folded model')
#plt.legend()
#a1 = plt.gca()
#a1.set_xlabel('Wavelength $\AA$')
#a1.set_ylabel('Relative Intensity')
#plt.title('Basic Gauss-Model convolution')
#plt.show(block=False)

##############

gauss_grid = np.tile(gaussian, (len(cos_wavs),1))

# multiplying this by flux array gives flux distribution by pixel
# BUT
# lose wavelength information, sort of

# try this as convolution input
#test2 = convolve(mdlf, gauss_grid)
# DOESN'T WORK

lg = len(gaussian)
lc = len(cos_wavs)

# this loop inserts lsf profiles into huge_array with a diagonal offset - YEAH, LOOPS!
huge_array = np.zeros((lc, lc+lg-1))
for c in range(lc):
    huge_array[c, c:c+lg] = gauss_grid[c]
# COULD REPLACE THIS BY NP.FLIPUD AND THEN NP.TRACE

mdlf.shape = (mdlf.shape[0], 1)
flux_dist = mdlf * huge_array
flux_result = np.sum(flux_dist, axis=0)
# gives a 1d array of flux value, with tails and depletion near edges
# want to 'fold in' edges of array to simulate continuum contribution outside wavelength range

plt.figure()
plt.plot(cos_wavs, mdlf, c='blue', lw=3, label='Model')
plt.plot(cos_wavs, conv_model[160:-160], c='r', lw=2, label='Simple convolution')
#plt.plot(cos_wavs, mdlf + 0.5, c='blue')
plt.plot(cos_wavs, flux_result[160:-160], c='orange', label='Manual convolution')
plt.legend()
plt.title('Comparison of numpy and manual convolutions')
a2 = plt.gca()
a2.set_xlabel
plt.show(block=False)

# difference is <=1e-9 at all points - YAY

############

# ADD COMPLEXITY

wg = models.Gaussian1D(amplitude=1, mean=160, stddev=10)
wide_gauss = wg(pix)
WG = spint.trapz(wide_gauss, pix)
wide_gauss /= WG
wide_grid = np.tile(wide_gauss, (len(cos_wavs)-15000, 1))
# smaller grid to apply to red end of 

gauss_grid2 = np.copy(gauss_grid)
gauss_grid2[15000:,:] = wide_grid

huge_array2 = np.zeros((lc, lc+lg-1))
for c in range(lc):
    huge_array2[c, c:c+lg] = gauss_grid2[c]

flux_dist2 = np.copy(mdlf) * huge_array2
flux_result2 = np.sum(flux_dist2, axis=0)

#plt.figure()
#plt.plot(cos_wavs, mdlf, c='blue', label='Model')
#plt.plot(cos_wavs, flux_result[160:-160], c='orange', label='Uniform Gauss')
#plt.plot(cos_wavs, mdlf + 0.5, c='blue')
#plt.plot(cos_wavs, flux_result2[160:-160]+0.5, c='red', label='Linearly Scaled Gauss')
#plt.legend()
#plt.axvline(cos_wavs[15000], color='grey')
#plt.title('Split width Gauss')
#plt.show(block=False)

del g_i, fit_g, g_f, G
del wg, wide_gauss, WG, wide_grid
del flux_dist, flux_dist2, flux_result, flux_result2
del huge_array, huge_array2

############

# THE PROPER ONE
# lsf_resps response function at fixed wavelength on rows

# NEED TO MAKE AN INTERPOLATION GRID FOR lsf by pixel by wavelength
# scipy.interpolate.interp2d makes a function which interpolates

#lsf = np.interp(cos_wavs, lsf_wavs, lsf_resps)

lsf_interp = interpolate.interp2d(lsf_wavs, pix, np.transpose(lsf_resps)) # uses linear interpolation

# LSF IS NOT NORMALISED

lsf_vals = lsf_interp(cos_wavs, pix)
lsf_i = np.trapz(lsf_vals, pix, axis=0)
lsf_vals2 = np.copy(lsf_vals) # 2 means non-normalised lsf distributions
lsf_vals /= lsf_i # normalise lsf by integral value - DO WE WANT TO DO THIS NOW!!! LOSE RELATIVE SIZE??
lsf_grid = np.zeros((cos_wavs.shape[0], cos_wavs.shape[0]+pix.shape[0]-1))
lsf_grid2 = np.zeros((cos_wavs.shape[0], cos_wavs.shape[0]+pix.shape[0]-1))
for j in range(cos_wavs.shape[0]):
    lsf_grid[j, j:j+pix.shape[0]] = lsf_vals[:,j]
    lsf_grid2[j, j:j+pix.shape[0]] = lsf_vals2[:,j]
lsf_flux = np.sum(fold_edges(mdlf*lsf_grid, pix.shape[0]), axis=0)
lsf_flux2 = np.sum(fold_edges(mdlf*lsf_grid2, pix.shape[0]), axis=0)

#plt.figure()
#plt.plot(cos_wavs, mdlf, label='Pure model')
#plt.plot(cos_wavs, lsf_flux, label='LSF convolved model') # tail effects
#plt.legend()
#plt.show(block=False)

#qc1 = np.sum(lsf_grid, 0)[160:-159] # NO EDGE FOLDING (tails clipped)
#qc2 = np.sum(fold_edges(lsf_grid, pix.shape[0]), 0) # Includes edge folding 
#plt.figure()
#plt.plot(qc1, label='No edge folding')
#plt.plot(qc2, label='Reflected edge folding')
#plt.legend()
#plt.title('Response change due to edge-folding')
#plt.show(block=False)

# NEED TO rescale models to fit data
mdlf_scaled = rescale_continuum(cos_wavs, cos_spec[:,1], np.copy(mdlf))
lsff_scaled = rescale_continuum(cos_wavs, cos_spec[:,1], np.copy(lsf_flux))
lsff2_scaled = rescale_continuum(cos_wavs, cos_spec[:,1], np.copy(lsf_flux2))

#f, (a1, a2) = plt.subplots(2, 1, sharex=True, sharey=True)
#a1.plot(cos_wavs, cos_spec[:,1], color='grey', alpha=0.5, label='COS spectrum')
#a2.plot(cos_wavs, cos_spec[:,1], color='grey', alpha=0.5, label='COS spectrum')
#a1.plot(cos_wavs, mdlf_scaled, color='blue', label='Model')
#a2.plot(cos_wavs, lsff_scaled, color='green', label='Model * LSF')
#a1.set_ylim(-5,70)
#a1.legend()
#plt.suptitle('LSF effect on model')
#a2.legend()
#f.subplots_adjust(hspace=0)
#plt.setp(a1.get_xticklabels(),visible=False)
#plt.show(block=False)

# test of normalising lsf before convolution
#plt.figure()
#plt.plot(cos_wavs, cos_spec[:,1], color='grey', alpha=0.5, label='COS spectrum')
#plt.plot(cos_wavs, mdlf_scaled, color='blue', label='Model')
#plt.plot(cos_wavs, lsff_scaled, color='green', label='Model * LSF')
#plt.plot(cos_wavs, lsff2_scaled, color='purple', label='Model * LSF 2')
#a0 = plt.gca()
#a0.set_ylim(-5,60)
#plt.title('LSF effect on model')
#plt.legend()
#plt.show(block=False)

model_wavs = apply_redshift(37000/3e8, np.copy(cos_wavs))

plt.figure()
plt.plot(cos_wavs, cos_spec[:,1], color='grey', alpha=0.5, lw=1, label='COS spectrum')
plt.plot(cos_wavs, lsff_scaled, color='green', label='Model * LSF')
plt.plot(model_wavs, lsff_scaled, color='red', label='Redshifted Model * LSF')
a0 = plt.gca()
a0.set_ylim(-5,60)
plt.legend()
plt.show(block=False)


artificial_gauss = models.Gaussian1D(amplitude=1, mean=160, stddev=20)
artificial_i = spint.trapz(artificial_gauss(pix), pix)
a_gauss = artificial_gauss(pix)/artificial_i # normalise

a_flux = fold_edges(np.convolve(np.copy(mdlf.reshape(28434,)), a_gauss).reshape(1,28754), 321)

#a_grid = np.tile(a_gauss, (len(cos_wavs),1))
#a_array = np.zeros((lc, lc+lg-1))
#for c in range(lc):
    #a_array[c, c:c+lg] = a_grid[c]
#a_flux = np.sum(fold_edges(mdlf * a_array, 321), axis=0)

a_flux_scaled = rescale_continuum(cos_wavs, cos_spec[:,1], a_flux)

plt.figure()
plt.plot(pix, gaussian, c='red')
plt.plot(pix, a_gauss, 'green')
plt.show(block=False)


plt.figure()
plt.plot(cos_wavs, cos_spec[:,1], lw=0.5, color='grey', alpha=0.5, label='COS')
plt.plot(model_wavs, lsff_scaled, color='green', lw=2, label='Model * LSF')
plt.plot(model_wavs, mdlf_scaled, color='red', lw=1, label='Model')
plt.plot(model_wavs, a_flux_scaled, color='blue', label='Wide Gaussian LSF')
plt.legend()
this_a = plt.gca()
this_a.set_ylim(-5, 60)
plt.show()

#######
"""
