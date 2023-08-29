import numpy as np
import matplotlib.pyplot as plt
import aurora 
import torch

# using aurora module and eqdsk obj from omfit_clases to convert r/a to rho_tor=sqrt(flux_tor_normed)
def ra_to_rho_converter(ra, gfile):
    
    # note: ra must be a list or array
    new_coordinates=np.empty(len(ra))
    for idx,coord in enumerate(ra):
        new_coordinates[idx]=aurora.coords.rad_coord_transform(x=coord, name_in='r/a', name_out='rhon', geqdsk=gfile)
    
    #print("\nr/a to rhon:\n", new_coordinates)
    return new_coordinates

def rho_to_ra_converter(ra, gfile):
    
    # note: ra must be a list or array
    new_coordinates=np.empty(len(ra))
    for idx,coord in enumerate(ra):
        new_coordinates[idx]=aurora.coords.rad_coord_transform(x=coord, name_in='rhon', name_out='r/a', geqdsk=gfile)
    
    #print("\nr/a to rhon:\n", new_coordinates)
    return new_coordinates

# arg naming convention taken prior to inclusion of r/a=0.9 point
def edge_values(rho0_8, prof0_8):

    c2, c1, c0 = np.polyfit(rho0_8,prof0_8,deg=2)
    d3, d2, d1, d0 = np.polyfit(rho0_8,prof0_8,deg=3)
    e4, e3, e2, e1, e0 = np.polyfit(rho0_8,prof0_8,deg=4)
    x=np.linspace(0,1.1,1000)
    y_fit1=c2*x**2 + c1*x + c0
    y_fit2=d3*x**3 + d2*x**2 + d1*x + d0
    y_fit3 = e4*x**4 + e3*x**3 + e2*x**2 + e1*x + e0

    # toggle to compare fit lines for extrapolation to rho=1
    mk_plot=False

    if mk_plot:
        # confirm fit is exact
        plt.scatter(rho0_8, prof0_8, marker="o", label="data")
        plt.plot(x,y_fit1, color="tab:orange", ls="--", label='y(x) = '+ '{:.2f}'.format(c2)+r'$x^2$'+'{:.2f}'.format(c1)+'x'+ \
            ' + ' + '{:.2f}'.format(c0))
        plt.plot(x,y_fit2, color="tab:green", ls="--", label='y(x) = '+ '{:.2f}'.format(d3)+r'$x^3$'+'{:.2f}'.format(d2)+r'$x^2$'+ \
            ' + ' '{:.2f}'.format(d1)+'x'+ ' + ' + '{:.2f}'.format(d0))
        plt.plot(x,y_fit3,color="tab:purple", ls="--", label='y(x) = '+ '{:.2f}'.format(e4)+r'$x^4$'+'{:.2f}'.format(e3)+r'$x^3$'+\
            '{:.2f}'.format(e2)+r'$x^2$'+ ' + ' '{:.2f}'.format(e1)+'x'+ ' + ' + '{:.2f}'.format(e0))
        plt.xlabel(r'$\rho _{TOR}$', fontsize=14)
        plt.legend()
        plt.show()

    # return value at rho_n=1
    return  d3 + d2 + d1 + d0

# function obtained from PORTALS/portals/powertorch_tools/physics/CALCtools.py
def integrateGradient(x,z,z0_bound):

	# Calculate profile
	b = torch.exp(0.5*(z[:,:-1]+z[:,1:])*(x[:,1:]-x[:,:-1]))
	f1 = b / torch.cumprod(b, 1) * torch.prod(b, 1, keepdims=True)

	# Add the extra point of bounday condition
	f = torch.cat( (f1,torch.ones(z.shape[0],1).to(f1)),dim=1) * z0_bound 

	return f

'''
# [10^20 m^-4]
def normed_to_real_units_density_converter(aln, dens, a):
    return (aln * dens)/a

# [keV]
def normed_to_real_units_temperature_converter(alT, temp, a):
    return (alT * temp)/a

'''