"""
Plots the ray traces and the RF power deposition density
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.collections import LineCollection
from scipy.interpolate import interp2d

import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}')

import netCDF4
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d_krf001.nc','r')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

#returns the index of the array whose element is closest to value
def findNearestIndex(value, array):
    idx = (np.abs(array - value)).argmin()

    return idx

#adds the ray traces to ax
def plotRays():
    xlim = gfileDict["xlim"] #R points of the wall
    ylim = gfileDict["ylim"] #Z points of the wall
    rbbbs = gfileDict["rbbbs"] #R points of the LCFS
    zbbbs = gfileDict["zbbbs"] # Z points of the LCFS
    
    wr  = cqlrf_nc.variables["wr"][:] #major radius of the ray at each point along the trace
    wz  = cqlrf_nc.variables["wz"][:] #height of the ray at each point along the trace
    delpwr= cqlrf_nc.variables["delpwr"][:] #power in the ray at each point
    wr *= .01; wz*=.01 #convert to m from cm
    
    maxDelPwrPlot = .8 #what portion of ray power must have been damped before we stop plotting that ray

    norm = plt.Normalize(0, 1)

    fig,ax = plt.subplots(figsize = (4.25,7.1))
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
    
    drawFluxSurfaces(ax)

#draw poloidal flux surfaces
def drawFluxSurfaces(ax):
    r = gfileDict["rgrid"]
    z = gfileDict["zgrid"]
    psirz = gfileDict["psirz"]
    
    psi_mag_axis = gfileDict["ssimag"]
    psi_boundary = gfileDict["ssibdry"]
    
    ## THIS NEEDS TO BE TOROIDAL RHO
    psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)

    rInterp = np.linspace(np.min(r), np.max(r), 200)
    zInterp = np.linspace(np.min(z), np.max(z), 200)
    psirzNormInterp = interp2d(r,z, psirzNorm, kind = 'cubic')(rInterp, zInterp)
    
    rhosToPlot = np.arange(.1,1.1,.1)

    ax.contour(rInterp, zInterp, psirzNormInterp, np.square(rhosToPlot), colors= 'k')

#plot where power is absorbed in the plasma due to urf, collisional, and additional linear absorption
def plotRFAbsorption():  
    rya = np.copy(cql_nc.variables["rya"])
    powrft = cql_nc.variables["powrft"][-1]/1e6#convert to W/m^3
    
    fig,ax = plt.subplots()
    ax.plot(rya, powrft, label = 'RF power absorption density')

    ax.set_ylabel("W/m^3")
    ax.set_xlabel(r'$\rho_{pol}$', fontsize = 20)
    ax.legend(loc = 'best')
    fig.tight_layout()

def main():
    plotRays()
    plt.savefig('rays.png')
    plotRFAbsorption()
    plt.savefig('RFAbsorption')
    plt.show()

    # RF power to mode l
    rya = np.copy(cql_nc.variables["rya"])
    powrf = cql_nc.variables['powrf'][:]
    final_powrf = np.ma.getdata(powrf[-1, :, :])
    final_powrf = final_powrf * 1e6  # convert to MW 
    print(max(final_powrf[5,:]))

    num_harmonics_to_plot = 6

    for i in range(num_harmonics_to_plot):
        plt.plot(rya, final_powrf[i,:], label=f'harmonic: {i}')

    plt.legend()
    plt.xlabel('r/a')
    plt.ylabel(r'MW/$m^3$')
    plt.show()

main()
