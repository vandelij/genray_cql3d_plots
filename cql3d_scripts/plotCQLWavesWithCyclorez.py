#-----------------------------------------------
#  Author: Jacob van de Lindt, Grant Rutherford
#-----------------------------------------------

# This script plots the rays and their power deposition, as well as the cyclotron harmonic layers (nearly verticle lines)

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.collections import LineCollection
from scipy.interpolate import interp2d
import netCDF4
import os, sys

#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

#----------------------------
# import CQL3D output data
#----------------------------
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

print('WARNING: plotting from a manually entered directory.')
# cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
# cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d_krf001.nc','r')

save_number_for_scan = '1'
folder = 'scan_beam_and_RF/beam_10_scan'  #scan_bmpwr_gen_D_gen_e_longer_rays'
cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/{folder}/cql3d_rfpwr_{save_number_for_scan}.nc','r')
cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/{folder}/cql3d_krf_rfpwr_{save_number_for_scan}.nc','r')
save_folder_and_name = f'{folder}/rays_rfpwr{save_number_for_scan}.png'

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)



#---------------------------
# gfile prcoessing area
#---------------------------

# get the function to process the gfile.
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}')

#below are the variables required to work with the magnetic field
rgrid = gfileDict["rgrid"]
print(rgrid.shape)
zgrid = gfileDict["zgrid"]
magAxisR = gfileDict['rmaxis'] 
magAxisZ = gfileDict['zmaxis'] 
B_zGrid = gfileDict["bzrz"]
B_TGrid = gfileDict["btrz"]
B_rGrid = gfileDict["brrz"]

# get the total feild strength 
Bstrength = np.sqrt(np.square(B_zGrid) + np.square(B_TGrid) + np.square(B_rGrid))

# get the poloidal field strength 
Bpstrength = np.sqrt(np.square(B_zGrid) + np.square(B_rGrid))
# getBpoloidal = interp2d(rgrid, zgrid, Bpstrength, kind = 'cubic')

# create a function that can grab the B-feild magnitude at any r, z coordiante pair. 
getBStrength = interp2d(rgrid,zgrid,Bstrength, kind = 'cubic')

# def plot_poloidal_field(r_resolution, z_resolution, levels):
#     r_array = np.linspace(rgrid[0], rgrid[-1], r_resolution)
#     z_array = np.linspace(zgrid[0], zgrid[-1], z_resolution)
#     R, Z = np.meshgrid(r_array, z_array)
#     Bpoloidal = getBpoloidal(r_array, z_array)    

#     plt.contour(R, Z, Bpoloidal, levels=levels)
    
    


#returns the index of the array whose element is closest to value
def findNearestIndex(value, array):
    idx = (np.abs(array - value)).argmin()

    return idx

#adds the ray traces to ax
def plotRays(frequency, harmonics, species, r_resolution, z_resolution, levels):
    xlim = gfileDict["xlim"] #R points of the wall
    ylim = gfileDict["ylim"] #Z points of the wall
    rbbbs = gfileDict["rbbbs"] #R points of the LCFS
    zbbbs = gfileDict["zbbbs"] # Z points of the LCFS
    
    wr  = cqlrf_nc.variables["wr"][:] #major radius of the ray at each point along the trace
    wz  = cqlrf_nc.variables["wz"][:] #height of the ray at each point along the trace
    delpwr= cqlrf_nc.variables["delpwr"][:] #power in the ray at each point
    wr *= .01; wz*=.01 #convert to m from cm
    
    maxDelPwrPlot = 0.8#.8 #what portion of ray power must have been damped before we stop plotting that ray

    norm = plt.Normalize(0, 1)

    fig, ax = plt.subplots(figsize = (4.25,7.1))
    plt.subplots_adjust(left=0.22,bottom = .1)
    ax.set_ylabel("z (m)")
    ax.set_xlabel("R (m)")

    #plot the ray using a LineCollection which allows the colormap to be applied to each ray
    for ray in range(len(wr)):
        delpwr[ray,:] = delpwr[ray,:]/delpwr[ray,0] #normalize the ray power to that ray's starting power
        mostPowerDep = findNearestIndex(1 - maxDelPwrPlot, delpwr[ray]) #find the index of the last ray point we want to plot

        
        points = np.array([wr[ray][:mostPowerDep], wz[ray][:mostPowerDep]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)

        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, norm = norm,cmap=plt.cm.jet)
        # Set the values used for colormapping
        lc.set_array(delpwr[ray][:mostPowerDep])
        lc.set_linewidth(1)
        ax.add_collection(lc)

    ax.plot(xlim, ylim, 'r', lw = 2)#plot wall
    ax.plot(rbbbs, zbbbs, 'k', lw = 1.5)#plot LCFS

    ax.set_title(f"Plotting Rays until {(maxDelPwrPlot) * 100} %\n ray power deposition")
    ax.set_aspect('equal')
    
    drawFluxSurfaces(ax, levels)
    plotCyclotronHarmonics(ax, frequency, harmonics, species, r_resolution, z_resolution)
    ax.legend()


def plotCyclotronHarmonics(ax, frequency, harmonics, species, r_resolution, z_resolution):
    '''
    ax: plot to append to 
    frequency: launched wave fequency [Hz]
    harmonics: list of harmonics to plot (example: [1, 2, 3] will plot the 1st, second, and third cyclotron harmonics for species) 
    species: string with species of interest (example: 'D' for deutrium)
    r_resolution: number of radial points to search over per z coord to plot the harmonic
    z_resolution: number of z coords. 
    '''
    w_wave = frequency*2*np.pi
    r_points = np.linspace(rgrid[0], rgrid[-1], r_resolution)
    z_points = np.linspace(zgrid[0], zgrid[-1], z_resolution)
    Bfield = getBStrength(r_points, z_points).T
    q = species_charge[species]
    m = species_masses[species]
    omega_j = q*Bfield/m
    print(f'omega_j.shape: {omega_j.shape}')

    normalized_w_wave = w_wave / omega_j
    R, Z = np.meshgrid(r_points, z_points)

    CS = ax.contour(R, Z, normalized_w_wave.T, levels=harmonics, colors=('blue',), linestyles=('--',))
    ax.clabel(CS, fmt = '%2.1d', colors = 'blue', fontsize=12)


    # depricated 
    # harmonic_holder = np.zeros((z_resolution, len(harmonics)))

    # for ih in range(len(harmonics)):

    #     for iz in range(z_resolution):
    #         omega_zi_array = omega_j[:, iz] # get the cyclotron frequency along a radial line for this z 
    #         ir = np.abs((w_wave - harmonics[ih]*omega_zi_array)).argmin()
    #         harmonic_holder[iz, ih] = r_points[ir]

    #     ax.plot(harmonic_holder[:, ih], z_points, color='blue', linestyle='--', label=f'{harmonics[ih]}$\Omega$')
    
    
#draw poloidal flux surfaces
def drawFluxSurfaces(ax, levels):
    r = gfileDict["rgrid"]
    z = gfileDict["zgrid"]
    psirz = gfileDict["psirz"]
    
    psi_mag_axis = gfileDict["ssimag"]
    psi_boundary = gfileDict["ssibdry"]
    
    ## THIS NEEDS TO BE TOROIDAL RHO
    # normalize the psirz so that the norm is 1 on boundary and zero on axis 
    psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)  

    # create 2D interpolation to create a contour plot from 
    rInterp = np.linspace(np.min(r), np.max(r), 200)
    zInterp = np.linspace(np.min(z), np.max(z), 200)
    psirzNormInterp = interp2d(r,z, psirzNorm, kind = 'cubic')(rInterp, zInterp)
    
    rhosToPlot = np.arange(.1,1.1,.1)

    ax.contour(rInterp, zInterp, psirzNormInterp, np.square(rhosToPlot), colors= 'k', levels=levels)




#---------------------------
# Species creation area
#---------------------------

species_list = ['D']

amu_in_kg = 1.66054e-27 # [kg]
species_masses = {}
species_masses['D'] = 2.014 * amu_in_kg

species_charge = {}
species_charge['D'] = 1.6022e-19 # [C] 


# setup for the cyclotron resonance plotter
# 
# # read in frequncy from genray.in
#genray_input = netCDF4.Dataset(f'../shots/{shotNum}/genray_received.in','r') 
#print(genray_input.keys())
#print(cql_nc.variables.keys())
frequency = 96000000.0 # [Hz]
species = 'D'
harmonics = [4, 5, 6, 7, 8]
r_resolution = 100
z_resolution = 200

rhosToPlot = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])#np.arange(.1,1.1,.08)
plotRays(frequency, harmonics, species, r_resolution, z_resolution, levels=rhosToPlot)
plt.savefig(f'../shots/{shotNum}/{save_folder_and_name}')

fig, ax = plt.subplots(figsize = (4.25,7.1))
plt.subplots_adjust(left=0.22,bottom = .1)
ax.set_ylabel("z (m)")
ax.set_xlabel("R (m)")
drawFluxSurfaces(ax, levels=rhosToPlot)
plt.savefig('flux_surfaces.png')
plt.show()