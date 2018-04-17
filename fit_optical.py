"""Fit optical spectrum with model grid


"""
# imports
import numpy as np
import glob as g
import matplotlib.pyplot as plt
from scipy import integrate as spint
from photometry import reddenings as rd

# constants
UPDATE_CHISQ = 1
UPDATE_CONTOURS = 1

# functions

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


def plot_spectrum(spectrum, c=None, l=None, a=0.7, s='-'):
    plt.plot(spectrum[:,0], spectrum[:,1], color=c, label=l, alpha=a, ls=s)


def plot_spectra(spectra, c=None, l=None, a=0.7, s='-'):
	for i, spectrum in enumerate(spectra):
		plt.plot(spectrum[:,0], spectrum[:,1], color=c[i], label=l[i], alpha=a[i], ls=s[i])
	plt.legend()
	plt.show()


def avg_flux(spec, wl): # REDUNDANT
    """Find avg flux of given spec data"""
    c = np.logical_and(spec[:,0]>wl[0], spec[:,0]<wl[1]) # take data in range given by wl
    respec = spec[np.where(c)]
    return np.average(respec[:,1])


def int_flux(spec, wl):
    c = np.logical_and(spec[:,0]>wl[0], spec[:,0]<wl[1]) # take data in range given by wl
    respec = spec[np.where(c)] # separate lines of array
    integral = spint.trapz(respec[:,1], respec[:,0])
    return integral


def normalise_flux(s1, s2, wlims):
    """Normalise flux from s2 to flux from s1 for a given wav range
    Return rescaled s1"""
    intg1 = int_flux(s1,wlims)
    intg2 = int_flux(s2,wlims)
    intg_scal = intg1/intg2
    s2[:,1] *= intg_scal
    plt.plot()
    return s2, intg_scal
    
    
def redu_chi_sq(obsspec, mdlspec):
    """Calculate chi-sq value for model, relative to observation
    
    Create linear interpolation data for each obs x
    Calc chi-sq_i and sum
    
    Do not need to reduced chi-sq for comparison between models
        for same spec, as same no of points
    BUT
        may be useful to compare int/ntt
    
    Return chi squared per point"""
    obsf = obsspec[:,1]
    err_obs = obsspec[:,2]
    mdlf = np.interp(obsspec[:,0], mdlspec[:,0], mdlspec[:,1])
    resid = (obsf - mdlf)
    
    onechi = np.square(resid) / err_obs**2    
    chisq = np.sum(onechi)
    rchisq = chisq / len(obsf)
    # returns chi squared per point
    return rchisq
    

def minmax(dat):
    """Find min,max value of 1d dat array -+10% (for plots))"""
    return [min(dat)*.9,max(dat)*1.1]


def vac2air(vac_spec):
    v = vac_spec[:,0]
    c1 = (1.0 + 2.735182 * np.power(10.0, -4.0))
    c2 = 131.4182
    c3 = 2.76249 * np.power(10.0, 8.0)
    a = v / (c1 + c2/(v**2) + c3/(v**4))
    vac_spec[:,0] = a
    return vac_spec


def contour_abc(a, b, c, d, savenam = False, xlbl=None, ylbl=None, ttl=None, bounds=None, showit=False, src=None, best_fit=None):
    """For discrete parameters a,b,c, contour plot b vs c (coloured by d) for each value of a"""
    # unique values
    ua = np.unique(a)
    ub = np.unique(b)
    uc = np.unique(c)
    # loop over values - HORRIBLE, OPTIMISE?
    for aval in ua:
        plot_BFV = 0
        plot_vals = np.zeros((len(ub), len(uc)))
        a_inds = np.argwhere(a==aval)
        if aval in best_fit:
            plot_BFV = 1
        for bval in ub:
            b_inds = np.argwhere(b==bval)
            for cval in uc:
                c_inds = np.argwhere(c==cval)
                xy = (np.argwhere(ub==bval), np.argwhere(uc==cval))
                # get index of line in master array which points to the combination three 'unique' values
                val_indx = np.intersect1d(a_inds, np.intersect1d(b_inds, c_inds))
                plot_vals[xy] = d[val_indx]
                if bval in best_fit and cval in best_fit:
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
            plot_name = 'optical/chi_sqr/plots/{}_{}_{}.png'.format(src, savenam, aval)
            plt.savefig(plot_name)
        if showit:
            plt.show()
        plt.clf()
        print("{} = {} complete".format(savenam, aval))
    
    # make lookup dictionary?
    # 3d meshgrid for a,b,c ?
    #for aval in ua:
        #for bval in ub:
            #for cval in uc:
                #pass
                

def manage_optical_fit(spec_file, spec_src):
    """Main function"""
    if UPDATE_CHISQ:
        print("Beginning reduced chi squared calculations.")
        fit_data = np.empty((len(mdl_list),5))
        # loop through models and find chisq for each
        i=0
        j=len(mdl_list)
        xlims = minmax(spec_file[:,0])
        ylims = minmax(spec_file[:,1])
        #exbv = rd.sfd_web(204713.82, -125909.5, 'equ 2000')[0] # extinction for reddening
        exbv = 0.01 # approx, from PANSTARRS maps for d <= 0.15 kpc
        for m in mdl_list:
            mdsc = m.split('/')[-1][:-3].split('_')
            teff = int(mdsc[0][1:])
            logg = float(mdsc[1][1:]) / 100
            h_he = float(mdsc[2][1:])
            outnam = "{}_t{}_g{}_h{}.png".format(spec_src, teff, logg, h_he)
            
            mod_red = rd.deredden_fix(get_dk(m), exbv)[0] # redden model
            modl, scal_fac = normalise_flux(spec_file, mod_red, (4500,6500)) # CAUTION WITH use of BLUE END LIMIT - 3750/4600/5100 TO MISS MOST LINES
            
            rxsq = redu_chi_sq(spec_file, modl)
            fit_data[i,:] = [teff, logg, h_he, rxsq, scal_fac]
    
            plot_spectrum(spec_file, l=spec_src.capitalize())
            plot_spectrum(modl, l='Model')
            plt.legend()
            plt.title("T_eff = {} log(g) = {:.2f} [H/He] = {:.2f}".format(teff, logg, h_he))
            plt.text(3400, 5e-16, "Scaling: Averaged flux\nReduced $\chi^2$ = {:.5e}".format(rxsq))
            plt.xlabel('Wavelength [$\AA$]')
            plt.ylabel('Flux [erg/cm^2/s/A]')
            axes = plt.gca()
            axes.set_xlim(xlims)
            axes.set_ylim(ylims)
            plt.savefig('optical/chi_sqr/plots/{}'.format(outnam),
                        dpi=192,
                        format='png')
            plt.clf()
            i+=1
            print("Calculation for {} complete ({:4.2f}%).".format(outnam, i*100/j))
        
        # save fit data to file
        np.savetxt('optical/chi_sqr/{}_rchisq_params.dat'.format(spec_src),
                   fit_data,
                   delimiter=',',
                   header='Teff,logg,[H/He],Red. chi-sq.,Rescaling',
                   fmt=['%i', '%.2f', '%.2f', '%.6e', '%.9e'])
    else:
        fit_data = np.genfromtxt('optical/chi_sqr/{}_rchisq_params.dat'.format(spec_src),
                                 delimiter=',')

    if UPDATE_CONTOURS:
        # separate all param cols (why)
        tf = fit_data[:,0].copy()
        lg = fit_data[:,1].copy()
        hh = fit_data[:,2].copy()
        xs = fit_data[:,3].copy()
        
        # find best fit vals
        minxs = np.nanmin(fit_data[:, 3])
        BFV = fit_data[np.argwhere(fit_data[:, 3]==minxs), :].copy()[0][0] # not sure why nested twice but it is

        contour_bounds = [0.49, 0.55, 0.60, 0.65, 0.75, 0.85, 1.00, 1.20, 1.4]
        # generate contour maps for all possible fixed variables
        contour_abc(hh, lg, tf, xs,
                    'fxd_HHe', 'T_eff [K]', 'log(g)', 'Fixed [H/He]',
                    contour_bounds, False, spec_src, BFV)
        contour_abc(tf, lg, hh, xs,
                    'fxd_Teff', '[H/He]', 'log(g)', 'Fixed T_eff',
                    contour_bounds, False, spec_src, BFV)
        contour_abc(lg, hh, tf, xs,
                    'fxd_logg', 'T_eff [K]', '[H/He]', 'Fixed log(g)',
                    contour_bounds, False, spec_src, BFV)   


####    MAIN    ####

int_spec = vac2air(np.genfromtxt('optical/int_rescaled.csv', delimiter=','))

#vac_spec = np.genfromtxt('optical/int_rescaled.csv', delimiter=',')
#plot_spectra([vac_spec, int_spec],
              #c = ['red', 'orange'],
              #l = ['Vac', 'Air'],
              #a = [0.7]*2,
              #s = ['-']*2
              #)

# define model list
mdl_list = g.glob('models/DAB/*.dk')
manage_optical_fit(int_spec, 'int')

#ntt_spec = np.genfromtxt('optical/ntt_rescaled.csv', delimiter=',') # NTT SPEC NOT USEFUL
#manage_optical_fit(ntt_spec, 'ntt') # NTT SPEC NOT USEFUL





