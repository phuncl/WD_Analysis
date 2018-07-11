""""""

import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from functools import reduce

from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import griddata
from scipy.interpolate import interp2d
from scipy.interpolate import LinearNDInterpolator as LNI

from scipy import optimize

from skimage import measure



def grid_the_data(a, b, c, d):
    """using 4d data with 3 input vals, create grid of ords and result"""
    # unique values are from ut, ul, uh
    val_grid = np.zeros((ut.size, ul.size, uh.size))
    
    # loop over values to fill grid - HORRIBLE, OPTIMISE? Possible?
    for aval in ut:
        a_inds = np.argwhere(a==aval)
        for bval in ul:
            b_inds = np.argwhere(b==bval)
            for cval in uh:
                c_inds = np.argwhere(c==cval)
                # get index of line in master array which points to the combination three 'unique' values
                val_indx = reduce(np.intersect1d, (a_inds, b_inds, c_inds))
                #print(val_indx)
                #if not val_indx: print('val_indx', aval, bval, cval)
                xyz = (np.argwhere(ut==aval), np.argwhere(ul==bval), np.argwhere(uh==cval))
                #print(xyz)
                #if not xyz: print('xyz', aval, bval, cval)
                try:
                    val_grid[xyz] = d[val_indx]
                except ValueError:
                    pass
    return val_grid


def query_chi_lni(params):
    t, l, h = params
    return chi_lni(t, l, h)


def slice_dimension(datagrid, uniques, arrayax, x, y, plot_axis, clr='r'):
    """"""
    all_contours = []
    slabs = np.split(datagrid, len(uniques), arrayax)
    
    for ind, u in enumerate(uniques):
        #print(u)
        X, Y = np.meshgrid(x, y)
        s = slabs[ind].squeeze()
        f = plt.figure()
        try:
            c = plt.contour(X, Y, s, levels=1+chi_MIN)
        except TypeError:
            c = plt.contour(X, Y, s.T, levels=1+chi_MIN) # 'plot# contours
        
        cont_info = c.collections[0].get_paths() # all contours info
        plt.close(f)
        
        for cont in cont_info:
            # extract plotting info
            xs, ys = cont.vertices[:,0], cont.vertices[:,1]
            if len(xs) <= 1:
                # i.e. if there is not contour to plot
                continue
            # plot
            if arrayax == 0:
                dummy = [u]*len(xs)
                plot_axis.plot(dummy, xs, ys, clr)
            if arrayax == 1:
                dummy = [u]*len(xs)
                plot_axis.plot(xs, dummy, ys, clr)
            if arrayax == 2: plot_axis.plot(xs, ys, u, clr)
            all_contours.append([xs, ys])
    return all_contours
    


def hires_slice_dimension(datagrid, uniques, x, y, plot_axis, clr='r'):
    """"""
    all_contours = []
    arrayax = np.argwhere(np.array(datagrid.shape) == uniques.shape[0])[0][0]
    slabs = np.split(datagrid, len(uniques), arrayax)
    
    for ind, u in enumerate(uniques):
        #print(u)
        X, Y = np.meshgrid(x, y)
        s = slabs[ind].squeeze()
        f = plt.figure()
        try:
            c = plt.contour(X, Y, s, levels=1+chi_MIN)
        except TypeError:
            c = plt.contour(X, Y, s.T, levels=1+chi_MIN) # 'plot# contours
        
        cont_info = c.collections[0].get_paths() # all contours info
        plt.close(f)
        
        for cont in cont_info:
            # extract plotting info
            xs, ys = cont.vertices[:,0], cont.vertices[:,1]
            if len(xs) <= 1:
                # i.e. if there is not contour to plot
                continue
            # plot
            if arrayax == 0:
                dummy = [u]*len(xs)
                plot_axis.plot(dummy, xs, ys, clr)
            if arrayax == 1:
                dummy = [u]*len(xs)
                plot_axis.plot(xs, dummy, ys, clr)
            if arrayax == 2: plot_axis.plot(xs, ys, u, clr)
            all_contours.append([xs, ys])
    return all_contours
    

####    MAIN

chi_MIN = 0.8917227
chi_MIN = 1.32218
#chi_data = np.genfromtxt('chisqr-results/hybgrid2018-fitvals_uv+opt_d98.328_b35000.dat', delimiter=',', comments='#', names=True)
#chi_data = np.genfromtxt('uv/chi_sqr/grid2018-fitvals_uv+opt.dat', delimiter=',', comments='#', names=True)
chi_data = np.genfromtxt('chisqr-results/grid2018-fitvals_uv+xsh_d98.328_b35000.dat', delimiter=',', comments='#', names=True)

chi_MIN = np.min(chi_data['TOTAL_Rchisq'])

ut = np.unique(chi_data['Teff'])
ul = np.unique(chi_data['logg'])
uh = np.unique(chi_data['HHe'])

chi_valgrid = grid_the_data(chi_data['Teff'], chi_data['logg'], chi_data['HHe'], chi_data['TOTAL_Rchisq'])
chi_MIN_fit = np.where(chi_data['TOTAL_Rchisq']==chi_MIN)
print(chi_MIN_fit)

# test 2d slab contour function
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(chi_data['Teff'], chi_data['logg'], chi_data['HHe'], s=2, c='k')
#ax.scatter(18150, 8.1, -1.1, s=20, c='cyan', marker='x')
ax.set_xlabel('$T_eff$')
ax.set_ylabel('$log g$')
ax.set_zlabel('$[H/He]$')

cont_stack1 = slice_dimension(chi_valgrid, ut, 0, ul, uh, ax, 'r')
cont_stack2 = slice_dimension(chi_valgrid, ul, 1, ut, uh, ax, 'g')
cont_stack3 = slice_dimension(chi_valgrid, uh, 2, ut, ul, ax, 'b')

plt.show()

#######
"""
hrut = np.arange(ut.min(), ut.max(), 50)
hrul = np.arange(ul.min(), ul.max(), 0.01)
hruh = np.arange(uh.min(), uh.max(), 0.02)
hrx, hry, hrz = np.meshgrid(hrut, hrul, hruh)
hires_grid = griddata((chi_data['Teff'], chi_data['logg'], chi_data['HHe']), chi_data['TOTAL_Rchisq'], (hrx, hry, hrz), fill_value=20)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(chi_data['Teff'], chi_data['logg'], chi_data['HHe'], s=2, c='k')
ax.scatter(18150, 8.1, -1.1, s=20, c='cyan', marker='x')
ax.set_xlabel('$T_eff$')
ax.set_ylabel('$log g$')
ax.set_zlabel('$[H/He]$')

c_stack1 = slice_dimension(hires_grid, hrut, hruh, hrul, ax, 'r')

plt.show()
"""





"""

# DO IT IN 2D, then move to 3D
h13 = chi_valgrid[:,:,0]
h12 = chi_valgrid[:,:,1]
h11 = chi_valgrid[:,:,2]
h10 = chi_valgrid[:,:,3]
h09 = chi_valgrid[:,:,4]
h08 = chi_valgrid[:,:,5]

h_slices = [h13, h12, h11, h10, h09, h08]

T, L = np.meshgrid(ut, ul)
HHe = -1.3
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(chi_data['Teff'], chi_data['logg'], chi_data['HHe'], 'k')

for h in h_slices:
    f = plt.figure()
    c = plt.contour(T, L, h.T, levels=1+chi_MIN)
    p = c.collections[0].get_paths() # all contours
    plt.close(f)
    
    for line in p:
        v = line.vertices
        x, y = v[:,0], v[:,1]
        ax.plot(x, y, HHe, 'red')
    HHe += 0.1
plt.show()

"""



#i = np.argwhere(chi_data['HHe']==-1.0)
#x = chi_data['Teff'][i].flatten()
#y = chi_data['logg'][i].flatten()
#z = chi_coordinate[i].flatten()

#test = griddata(np.stack((x, y), axis=1), z, np.stack((hi_ut, hi_ul), axis=1), method='linear')



#flat_test = optimize.root(query_slice, np.asarray([18000, 8.]))
#flat_test2 = optimize.fsolve(query_slice, [18000, 8.])

#f_test = measure.find_contours(flat_hires, 1+chi_MIN)



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

