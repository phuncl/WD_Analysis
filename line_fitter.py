"""Line fitter program"""

import numpy as np
import matplotlib.pyplot as plt

from astropy.modeling import models, fitting
from matplotlib.widgets import SpanSelector
from matplotlib import gridspec


def apply_redshift(beta, spec):
    """Redshift spec by beta = v/c"""
    spec[:,0] += spec[:,0] * (np.sqrt(1+beta)/np.sqrt(1-beta) - 1)
    return spec


####################
#   LINE FITTING   #
####################


def model_the_line(contw, contf, linew, linef, n=2):
    """Use cont_data - line_data
    to model continuum as order n polynomial.
    
    Then fit a gaussian to the line,
    calculating initial guesses from data
    """
    global collected_fits
    # create error array for cont
    in_rng = np.logical_and(coswav[:]>=contw[0], coswav[:]<=contw[-1])
    conte = coserr[np.where(in_rng)]
    
    # find indices of cont that are data within line
    line_in_cont = np.logical_and(contw>=linew[0], contw<=linew[-1])
    contw_only = contw[~line_in_cont] # data not in line region
    contf_only = contf[~line_in_cont]
    conte_only = conte[~line_in_cont]
    
    coefs = np.polyfit(contw_only, contf_only, n, w=1/conte_only) # weight by w=1/error**2 or suchlike?
    contf_fit = np.polyval(coefs, contw)
    contf_reduced = contf - np.polyval(coefs, contw)
    
    # Guess parameters of gaussian from line data selection
    g_fwhm = (linew[-1] - linew[0])/4 # FWHM ~ 1/4 of data width
    g_amp = np.min(linef) - np.polyval(coefs, linew[np.where(linef==np.min(linef))]) # minimum flux value in line region1
    g_mdpt = np.mean(linew) # mid point of line region
    
    # fit gaussian to data
    g_init = models.Gaussian1D(amplitude=g_amp, mean=g_mdpt, stddev=g_fwhm)
    fit_g = fitting.LevMarLSQFitter()
    g_result = fit_g(g_init, contw, contf_reduced, weights=1/conte)
    final_gauss = g_result(contw) + contf_fit
    
    fitfig, (a_fit, a_res) = plt.subplots(2, sharex=True, figsize=(10.,6.))
    
    a_fit.plot(contw, contf,
    color='black', marker='o', linestyle='')
    a_fit.plot(contw, contf_fit,
    color='grey', linestyle='-', alpha=0.5, label='Continuum fit')
    a_fit.plot(contw, g_init(contw) + contf_fit,
    color='orange', linestyle='-', label='Estimated Gaussian')
    a_fit.plot(contw, final_gauss,
    color='red', linestyle='-', label='Fit Gaussian')
    
    a_fit.set_title('Fit to line')
    a_fit.xaxis.set_visible(False) # set tick labels invisible
    a_fit.legend()
    a_fit.set_ylabel('Flux [units]')
    
    a_res.axhline(c='r', lw=0.5)
    a_res.errorbar(contw, contf-final_gauss, conte,
    fmt='k.')
    a_res.set_xlabel('Wavelength [$\AA$]')
    a_res.set_ylabel('Relative flux')
    
    resid_mean = np.mean(contf - final_gauss)
    print('Mean of residuals = {:.3e}'.format(resid_mean))
    # rescale y axis to something meaningful?
    fitfig.subplots_adjust(hspace=0)
    # Measure fit quality from stddev of residuals?
    print('Fit parameters')
    fit_rms = np.sqrt(np.mean(np.square(final_gauss - contf)))
    fit_params = [g_result.mean.value, g_result.stddev.value, g_result.amplitude.value, fit_rms]
    print(g_result)
    print('RMS = {:.4e}'.format(fit_rms))
    plt.show()
    collected_fits.append(fit_params)


####################
# REGION SELECTION #
####################


def controls():
    print("""Control buttons for data selection:
    1:  Switch to continuum selection mode
    2:  Switch to absorption line selection mode
    3:  Switch to navigation mode (use pan/zoom tools in hotbar)
    9:  Fit continuum and line from selected data
    r:  Reset plots
    h:  Display control information
    q:  Quit the program
    """)


def redraw():
    """"""
    global fig, a_pick, a_cont, a_line, cont_region, line_region
    
    # reset pick plot, maintaining previous limits of view
    a_pick_tmpx = a_pick.get_xlim()
    a_pick_tmpy = a_pick.get_ylim()
    a_pick.clear()
    a_pick.plot(coswav, cosflx)
    a_pick.set_xlim(a_pick_tmpx)
    a_pick.set_ylim(a_pick_tmpy)

    c1, c2 = cont_region.get_data()
    a_cont.clear()
    cont_region, = a_cont.plot(c1, c2)
    a_cont.autoscale_view()
    if c1[0] == coswav[0] and c1[-1] == coswav[-1]:
        # no effective selection
        print('No continuum sub-selection')
        print('Min = {:.3f}, Max = {:.3f}'.format(c1[0], c1[-1]))
    else:
        print('Continuum has been sub-selected')
        cont_box = a_pick.axvspan(c1[0], c1[-1],
                                  facecolor='orange', alpha=0.3)
    
    l1, l2 = line_region.get_data()
    a_line.clear()
    line_region, = a_line.plot(l1, l2)
    a_line.autoscale_view()
    if l1[0] == coswav[0] and l1[-1] == coswav[-1]:
        print('No line sub-selection')
        print('Min = {:.3f}, Max = {:.3f}'.format(l1[0], l1[-1]))
    else:
        print('Line has been sub-selected.')
        line_box = a_pick.axvspan(l1[0], l1[-1],
                                  facecolor='red', alpha=0.3)
    
    a_pick.set_title('Selection Panel')
    a_cont.set_title('Continuum selection')
    a_line.set_title('Masking selection')
    plt.draw()


def toggler(event):
    """"""
    # ADD ANOTHER MASKING MODE FOR NONE-LINE FEATURES?
    global active_region, active_ax
    
    cx, cy = cont_region.get_data()
    lx, ly = line_region.get_data()
    print('\nContinuum range = {}-{}'.format(cx.min(), cx.max()))
    print('Line range = {}-{}'.format(lx.min(), lx.max()))
    print('\nInput detected.')
    
    if event.key.lower() == 'r':
        print('Resetting to original spectrum...')
        cont_region.set_data(np.copy(coswav), np.copy(cosflx))
        line_region.set_data(np.copy(coswav), np.copy(cosflx))
        redraw()
    
    elif event.key == '1' and active_region != cont_region:
        # Switch to continuum selection
        print('Continuum selection mode.')
        #toggler.SS.set_active(True)
        active_region = cont_region
        active_ax = a_cont
        #redraw()
    
    elif event.key == '2' and active_region != line_region:
        # Switch to line selection
        print('Line selection mode.')
        #toggler.SS.set_active(True)
        active_region = line_region
        active_ax = a_line
        #redraw()
    
    elif event.key == '3' and active_region != pick_region:
        # Turn off selection
        print('No selection mode.')
        active_region = pick_region
        active_ax = a_pick
        #toggler.SS.set_active(False)
        #redraw()
    
    elif event.key.lower() == '9':
        # model the line!
        model_the_line(cx, cy, lx, ly)
        redraw()
    
    elif event.key.lower() == 'h':
        controls()
    
    elif event.key.lower() == 'q':
        # option to save fit outputs to file here
        
        print('Goodbye')
        quit()
    
    else:
        print('Input not understood.')
        print('You may already be in the selected mode.')


def selection_call(mclick, mrelease):
    """"""
    print('\n')
    global active_ax, active_region
    
    x1, x2 = sorted((mclick, mrelease))
    print('Input x range {:.3f} - {:.3f}'.format(x1, x2))
    in_range = np.logical_and(coswav[:]>=x1, coswav[:]<=x2)
    subwav = coswav[np.where(in_range)]
    subflx = cosflx[np.where(in_range)]
    data_errors = coserr[np.where(in_range)]
    print('Data x range {:.3f} - {:.3f}'.format(subwav.min(), subwav.max()))
    active_region.set_data(subwav, subflx)
    #print(active_region.get_data())
    active_ax.relim()
    active_ax.autoscale_view()
    redraw()
    

####    MAIN    ####

print("""This program performs basic fitting of lines in spectra.

Using the interactive plot, select a section of continuum to fit,
and the subset of that region which is a spectral line.

The line will be masked while the continuum if fit with an order 3 polynomial.
A gaussian fit will the then be performed on the line.

Following a fitting, the plot will reset and a new feature may be picked.
""")

# container for fit data
# will hold midpt, stddev, depth
collected_fits = []

cos = np.loadtxt('uv/APASSJ204713.82-125909.5_2017-11-04.dat', delimiter=' ')
coswav, cosflx, coserr = cos[:,0], cos[:,1], cos[:,2]
#data_errors = cos[:,2]

fig = plt.figure(figsize=(9, 7))
gs = gridspec.GridSpec(2,2)

a_pick = plt.subplot(gs[0, :])
a_pick.set_title('Selection Panel')
a_cont = plt.subplot(gs[1, 0])
a_cont.set_title('Continuum selection')
a_line = plt.subplot(gs[1, 1])
a_line.set_title('Absorption line selection')

pick_region, = a_pick.plot(coswav, cosflx)
cont_region, = a_cont.plot(coswav, cosflx)
line_region, = a_line.plot(coswav, cosflx)

active_region = cont_region  # initially set to continuum region mode
active_ax = a_cont           # but can change mode in toggler

# create mode dictionary to set active axes abd region at start of each region selection
mode = {}

toggler.SS = SpanSelector(a_pick,
                          selection_call,
                          'horizontal',
                          useblit=True,
                          rectprops=dict(facecolor='green', alpha=0.2),
                          button=1)

controls()
print('Opening plot.\nContinuum selection mode.')
plt.connect('key_press_event', toggler)
plt.show()

quit()
