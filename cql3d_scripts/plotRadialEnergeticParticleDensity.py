"""
Plots a heat map of energy and radial position of particles above a certain threshold energy
Integrated over pitch angle
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cbook as cbook
from matplotlib import cm
from matplotlib import ticker, cm 
 
import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import netCDF4
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]
cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d_krf001.nc','r')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

def plotEnergeticHeatMap():
    rya = cql_nc.variables["rya"][:]#radial points at which f was solved: "Normalized radial mesh at bin centers"

    #distribution function (lrz, jx, iy)
    #vnorm^3*s^3/cm^6
    f = cql_nc.variables["f"][:]

    #pitch angles mesh at which f is defined
    pitchAngleMesh = np.ma.getdata(cql_nc.variables["y"][:])   # y is pitch angle, x is velocity (momentum per mass)
    #normalized speed mesh of f
    normalizedVel = cql_nc.variables["x"][:]

    #energy  = restmkev*(gamma-1)
    #energies corresponding to velocities jx
    enerkev = cql_nc.variables["enerkev"][:] 
    #flux surface average energy per particle
    energy = cql_nc.variables["energy"][:] #(tdim, r0dim, species_dim)

    #minimum energy of particles to consider for plotting
    minEnergy = 2
    #index of that minimum energy in enerkev
    #this index is also the index for the corresponding velocity
    minEnergyIndex = np.where(enerkev < minEnergy)[0][-1]
    #distribution function for energetic particles
    energeticF = f[:,minEnergyIndex:,:]


    energeticF_integOverPitch = np.zeros((len(rya), len(enerkev[minEnergyIndex:])))
    for rhoIndex in range(len(rya)):
        #this is the angular part of the spherical jacobian
        integOverPitch = 2*np.pi*np.trapz(energeticF[rhoIndex,:]*np.sin(pitchAngleMesh[rhoIndex]), pitchAngleMesh[rhoIndex], axis = 1)
        energeticF_integOverPitch[rhoIndex,:] = integOverPitch
    
    relevantEnergies = enerkev[minEnergyIndex:]
    relevantVels = normalizedVel[minEnergyIndex:]
    energeticDensity = np.zeros((len(rya), len(relevantEnergies)-1))
    #the energies at which the values of energeticDensity are centered
    energyCenters = (relevantEnergies[1:] + relevantEnergies[:-1])/2
    

    #We calculate energy centers by integrating up to the velocity mesh point on either side of the corresponding energyCenter point
    #we then take the differences, which is the area in the bin between these two edges
    for velIndex in range(len(relevantVels)-1):
        #this is the radial part of the spherical jacobian
        densityLower = np.ma.getdata(np.trapz(energeticF_integOverPitch[:,:velIndex]*relevantVels[:velIndex]**2,
            relevantVels[:velIndex]))
        densityUpper = np.ma.getdata(np.trapz(energeticF_integOverPitch[:,:velIndex+1]*relevantVels[:velIndex+1]**2,
            relevantVels[:velIndex+1]))

        if velIndex == 0:
            energeticDensity[:,velIndex] = densityUpper
            continue

        diff = densityUpper - densityLower

        energeticDensity[:,velIndex] = diff

    energeticDensity = energeticDensity*1e6#convert to m^(-3)
    energeticDensity = energeticDensity.astype('float64')
    rya = rya.astype('float64')

    #"""
    fig, ax = plt.subplots()
    pcm = ax.pcolormesh(energyCenters, rya, energeticDensity, 
                    norm=colors.LogNorm(vmin=1, vmax=np.max(energeticDensity)),
                    shading = 'nearest')
    # print('---')
    # print(rya.shape)
    # pcm = ax.pcolormesh(rya, energyCenters, energeticDensity, 
    #                 norm=colors.LogNorm(vmin=1, vmax=np.max(energeticDensity)),
    #                 shading = 'nearest')

    fig.colorbar(pcm, ax=ax)#, extend='max')

    ax.set_xlim([0,500])
    plt.savefig('heatmap.png')
    plt.show()

    #this code here was used to test that this difference method of getting the particle spatial density at each radial and velocity point
    #do actually give you the correct profile when integrated over velocity
    """
    correct = np.trapz(energeticF_integOverPitch*normalizedVel[minEnergyIndex:]**2, 
                normalizedVel[minEnergyIndex:],axis = -1)

    #density = np.trapz(energeticF_integOverPitch[rhoIndex][lowerIndex:upperIndex]*normalizedVel[lowerIndex:upperIndex]**2, 
    #            normalizedVel[lowerIndex:upperIndex])

    trial1 = np.sum(energeticDensity,axis = -1)
    
    trial2 = np.trapz(energeticDensity, dx = 1, axis = -1)

    fig,ax = plt.subplots()
    ax.plot(rya, correct/np.max(correct) + .025,label = 'target')
    ax.plot(rya, trial1/np.max(trial1), label = "trial")
    ax.plot(rya, trial2/np.max(trial2), label = "trial2")
    ax.legend()
    plt.show()
    """

plotEnergeticHeatMap()
