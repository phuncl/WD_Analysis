""""""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
from scipy.interpolate import LinearNDInterpolator as LNI

from scipy import optimize



def grid_the_data(a, b, c, d):
    """using 4d data with 3 input vals, create grid of ords and result"""
    # unique values
    ua = np.unique(a)
    ub = np.unique(b)
    uc = np.unique(c)
    
    val_grid = np.zeros((ua.size, ub.size, uc.size))
    
    # loop over values to fill grid - HORRIBLE, OPTIMISE? Possible?
    for aval in ua:
        a_inds = np.argwhere(a==aval)
        for bval in ub:
            b_inds = np.argwhere(b==bval)
            for cval in uc:
                c_inds = np.argwhere(c==cval)
                # get index of line in master array which points to the combination three 'unique' values
                val_indx = np.intersect1d(a_inds, np.intersect1d(b_inds, c_inds))
                
                xyz = (np.argwhere(ua==aval), np.argwhere(ub==bval), np.argwhere(uc==cval))
                val_grid[xyz] = d[val_indx]
    val_axes = (ua, ub, uc)
    return val_grid, val_axes


def fit_a_contour():
    return


def fit_min():
    return


def chi_int(params, offset=0):
    t, l, h = params
    return chi_rgi((t, l, h)) - offset


def query_chi_lni(params, offset=0):
    t, l, h = params
    return chi_lni(t, l, h)



####    MAIN

chi_data = np.genfromtxt('chisqr-results/sgrid2018-fitvals_uv+opt_d98.328_b35000.dat', delimiter=',', comments='#', names=True)

chi_valgrid, chi_valaxes = grid_the_data(chi_data['Teff'], chi_data['logg'], chi_data['HHe'], chi_data['TOTAL_Rchisq'])

chi_ordinates = (chi_data['Teff'], chi_data['logg'], chi_data['HHe'])
chi_ordgrid = np.stack(chi_ordinates, axis=1)
chi_coordinate = chi_data['TOTAL_Rchisq']
chi_lni = LNI(chi_ordgrid, chi_coordinate)

rtest = query_chi_lni([18100, 8.1, -1.1])
rtest2 = query_chi_lni([18200, 8.0, -1.0])

r = optimize.minimize(query_chi_lni, [18150, 8.15, -1.2 ])



"""
chi_rgi = RGI(chi_valaxes, chi_valgrid, fill_value=100, bounds_error=False)
#chi_interp = lambda x, y, z: chi_rgi((x, y, z))

res = optimize.minimize(chi_int, [18200., 8.2, -1.2],)# method='SLSQP')

interpgrid = (np.arange(17900., 18300., 1.), np.arange(7.95, 8.25, 0.01), np.arange(-1.3, -0.8, 0.1))
chi_grid = griddata(chi_valaxes, chi_valgrid, (18110., 8.1, -1.1), method='linear')
"""

#chi_lni = LNI(chi_valaxes, chi_valgrid)
# take subset where h/he = -1.0?


# Plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

