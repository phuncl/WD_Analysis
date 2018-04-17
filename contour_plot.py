"""Contour plotting from teff,logg,hhe, chisq data"""

import numpy as np
import matplotlib.pyplot as plt

#import scipy.ndimage


def contour_abc(a, b, c, d, savenam = False, xlbl=None, ylbl=None, ttl=None, bounds=None, showit=False):
	"""For discrete parameters a,b,c, contour plot b vs c (coloured by d) for each value of a"""
	# unique values
	ua = np.unique(a)
	ub = np.unique(b)
	uc = np.unique(c)
	# loop over values
	
	for aval in ua:
		plot_BFV = 0
		plot_vals = np.zeros((len(ub), len(uc)))
		a_inds = np.argwhere(a==aval)
		if aval in BFV:
			plot_BFV = 1
		for bval in ub:
			b_inds = np.argwhere(b==bval)
			for cval in c:
				c_inds = np.argwhere(c==cval)
				xy = (np.argwhere(ub==bval), np.argwhere(uc==cval))
				val_indx = np.intersect1d(a_inds, np.intersect1d(b_inds, c_inds))
				plot_vals[xy] = d[val_indx]
				if bval in BFV and cval in BFV:
					BFVx = cval
					BFVy = bval
		#plt.figure()
		#plot_vals = scipy.ndimage.zoom(plot_vals, 3) # interpolate for smoother contouring
		#uc = scipy.ndimage.zoom(uc,3)
		#ub = scipy.ndimage.zoom(ub,3)
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
			plot_name = 'optical/chi_sqr/plots/{}_{}.png'.format(savenam, aval)
			plt.savefig(plot_name)
		if showit:
			plt.show()
		plt.clf()
		

		print("{} = {} complete".format(savenam, aval))


def CONT_abc(data_array, col_indx, best_val): #  try to improve contour_plot
	"""Make unique value array from col_indx
	Make sub-set of data_array and plot for each unique value
		
	data_array:  2d data array
	col_indx: index of column to be sliced by unique value
	best_val = best fit value of data_array
	"""
	unq_array = np.unique(data_array[:,col_indx])
	
	for unq_val in unq_array:
		print(unq_val)
		sub_data = data_array[np.where(data_array[:,col_indx] == unq_val)]
		print("Subdata", sub_data)
		sub_data2 = np.delete(sub_data, col_indx, 1)
		print("Subdata2", sub_data2)
		# sub_data removes 
		
		plt.contourf(sub_data2)
		plt.colour_bar()
		plt.show()
		plt.clf()
		print()
	


####    MAIN    ####

DATA = np.genfromtxt('optical/chi_sqr/rchisq_params.dat',
					 delimiter = ',')
# create copy 1d arrays
tf = DATA[:,0].copy()
lg = DATA[:,1].copy()
hh = DATA[:,2].copy()
xs = DATA[:,3].copy()
# note best fit value
minxs = np.nanmin(DATA[:, 3])
BFV = DATA[np.argwhere(DATA[:, 3]==minxs), :].copy()[0][0] # not sure why nested twice but it is
print(BFV)

contour_bounds = [8.0e-18, 8.4e-18, 8.8e-18, 9.2e-18, 9.6e-18, 1.0e-17, 1.2e-17, 1.4e-17, 1.7e-17, 2.0e-17, 2.3e-17]

contour_abc(hh, lg, tf, xs, 'fxd_HHe', 'T_eff [K]', 'log(g)', 'Fixed [H/He]', contour_bounds)
contour_abc(tf, lg, hh, xs, 'fxd_Teff', '[H/He]', 'log(g)', 'Fixed T_eff', contour_bounds)
contour_abc(lg, hh, tf, xs, 'fxd_logg', 'T_eff [K]', '[H/He]', 'Fixed log(g)', contour_bounds)

#CONT_abc(DATA, 2, BFV)

