"""Plot histogram of atmospheric values for top <> chi-sqr fits"""

import numpy as np
import matplotlib.pyplot as plt


def plot01(dat, labl=None, colour=None, lstyle=None, mark=None, a=1):
    """Streamlined plt.plot function for first two columns of array"""
    return plt.plot(dat[:,0], dat[:,1], label=labl, c=colour, ls=lstyle, marker=mark, alpha=a)


def hist_manager(fits, max_index=None, max_rxsq=1, max_factor=2, ttl=None):
    fits_sort = fits[np.argsort(fits[:,3])]
    fit_best = fits_sort[0]
    rxsq_best = fit_best[3]
    if max_index:
        fits_best = fits_sort[:max_index]
    elif max_rxsq:
        fits_best = fits_sort[np.where(fits_sort[:,3] <= max_rxsq)]
    else:
        rxsq_max = rxsq_best * max_factor # didn't actually need to be sorted to do this...
        fits_best = fits_sort[np.where(fits_sort[:,3] <= rxsq_max)]
    inc = len(fits_best)
    # now have top <number> of fit parameters
    # create histograms
    teff = np.copy(fits_best[:,0])
    create_hist(teff, ttl+'_T_eff', inc)
    logg = np.copy(fits_best[:,1])
    create_hist(logg, ttl+'_logg', inc)
    hyhe = np.copy(fits_best[:,2])
    create_hist(hyhe, ttl+'_H_He', inc)
    

def create_hist(column, titl, num):
    """"""
    bwid = np.diff(np.unique(column)).min()
    bmin = column.min() - bwid/2
    bmax = column.max() + bwid/2
    
    plt.hist(column, np.arange(bmin, bmax+bwid, bwid))
    if titl: plt.title(titl + ' ({} points)'.format(num))
    plt.savefig('uv/plots/atmos_params/{}.png'.format(str(titl)))
    #plt.show()
    plt.clf()
    return


def plot_whist(uniques, col1, col2, xlab='_', t='_'):
    """"""
    results = []
    for val in uniques:
        w_result = np.nansum(1/(col2[np.where(col1==val)])**2)
        results.append(w_result)
    for i in range(len(uniques)):
        print(uniques[i], results[i])
    plt.plot(uniques, results, 'bs-')
    plt.xlabel(xlab)
    plt.ylabel('Weighted occurence')
    plt.title(t)
    figname = str(t) + '_' + str(xlab)
    plt.savefig('uv/plots/atmos_params/{}.png'.format(figname))
    #plt.show()
    plt.clf()


def create_weighted_hist(array, ttl):
    """"""
    d_array = np.copy(array)
    d_teff = d_array[:,0]
    d_logg = d_array[:,1]
    d_hyhe = d_array[:,2]
    d_rxsq = d_array[:,3]
    
    u_teff = np.unique(d_teff)
    u_logg = np.unique(d_logg)
    u_hyhe = np.unique(d_hyhe)
    
    plot_whist(u_teff, d_teff, d_rxsq, 'T eff', ttl)
    plot_whist(u_logg, d_logg, d_rxsq, 'log(g)', ttl)
    plot_whist(u_hyhe, d_hyhe, d_rxsq, '[H_He]', ttl)
    

####    MAIN    ####

chisqr_int = np.loadtxt('optical/chi_sqr/int_rchisq_params.dat', delimiter=',', comments='#')
chisqr_cos = np.loadtxt('uv/chi_sqr/rxsq_params.dat', delimiter=',', comments='#')
chisqr_phtcos = np.loadtxt('uv/chi_sqr/pht_rxsq_params.dat', delimiter=',', comments='#')

hist_manager(chisqr_int, 30, max_factor = 1.5, ttl='INT')
hist_manager(chisqr_cos, 30, max_factor = 4, ttl='COS')
hist_manager(chisqr_phtcos, 30, ttl='COS(photometry)')

create_weighted_hist(chisqr_int, 'INT')
create_weighted_hist(chisqr_cos, 'COS')
create_weighted_hist(chisqr_phtcos, 'COS(photometry)')



#fits = np.copy(chisqr_int)
#max_index=None
#max_factor=1.5

#fits_sort = fits[np.argsort(fits[:,3])]
#fit_best = fits_sort[0]
#rxsq_best = fit_best[3]
#if max_index:
    #fits_best = fits_sort[:max_index]
#else:
    #rxsq_max = rxsq_best * max_factor # didn't actually need to be sorted to do this...
    #fits_best = fits_sort[np.where(fits_sort[:,3] <= rxsq_max)]

## now have top <number> of fit parameters
## create histograms
#column = np.copy(fits_best[:,0])
##create_hist(teff)
#bwid = np.diff(np.unique(column)).min()
#bmin = column.min() - bwid/2
#bmax = column.max() + bwid/2


#plt.hist(column, np.arange(bmin, bmax+bwid, bwid))
#plt.show()




#SOLUTION TO WEIGHTED HISTOGRAM
#for val in rangeofvals
#np.sum(1/x[np.where(x[:,0]==val),3])
