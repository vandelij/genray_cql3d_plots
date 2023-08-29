"""
Plots n_para and n_perp evolution of the rays
Can readily be modified to instead plot the evolution of the wave numbers or wavelengths
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.collections import LineCollection
from plasmapy.formulary import (plasma_frequency, gyrofrequency)
from astropy import units as u
from matplotlib.lines import Line2D
import matplotlib 

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import netCDF4
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

fig,axes = plt.subplots(2,1, figsize = (7,6))

minPower = 0 #when the ratio of ray power to ray starting power is below this number, the trace ends

def investigatePropagation():
    fileSuffixes = ['', '_1.2Tescal']
    labels = ['174079', '174079, Tescal = 1.2']
    linewidths = [1,2]
    lines = []
    for j in range(len(fileSuffixes)):
        fileSuffix = fileSuffixes[j]
        cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/scanResults/cql3d_krf001{fileSuffix}.nc','r')

        nparas = cqlrf_nc.variables["wnpar"][:] #n_|| of the ray at each point along the ray trace
        nperps = cqlrf_nc.variables["wnper"][:] #n_perp of the ray at each point along the ray trace
        radialVariable = cqlrf_nc.variables["spsi"][:] #rho_pol of the ray at each point along the ray trace
        delpwr= cqlrf_nc.variables["delpwr"][:] #power in the ray at each point
        density = cqlrf_nc.variables["sene"][:]
        B = cqlrf_nc.variables["sbtot"][:]
        w = 2*np.pi*4.6e9*u.rad/u.s #angular frequency of the wave
        ws = cqlrf_nc.variables["ws"][:] #poloidal distance along ray
        
        distBins = np.linspace(0, np.max(ws),int(1e4))
        
        distanceFromN_acc = np.zeros(len(distBins))
        avgRhoOfRays = np.copy(distanceFromN_acc)
        dn_para_dl = np.copy(distanceFromN_acc)
        avgNPara = np.copy(distanceFromN_acc)
        powerInRay = np.copy(distanceFromN_acc)
            
        counter = np.copy(distanceFromN_acc)
        for i in range(len(nparas)):
            if np.max(nparas[i]) > 0:
                continue

            delpwrRatios = delpwr[i]/np.max(delpwr[i])
            lastIndex = np.argmin(np.abs(delpwrRatios - minPower))
            

            npara_alongRay = nparas[i][:lastIndex] 
            rho_alongRay = radialVariable[i][:lastIndex] 
            dist_alongRay =ws[i][:lastIndex] 
            density_alongRay = density[i][:lastIndex] 
            B_alongRay = B[i][:lastIndex] 

            distBinIndices = np.digitize(dist_alongRay, distBins, right = False) -1
            counter[distBinIndices] += 1
            avgNPara[distBinIndices] += npara_alongRay

            #if '3e19' in fileSuffix:
            #    axes[0].plot(dist_alongRay, npara_alongRay)


            w_pe = plasma_frequency(density_alongRay*(u.cm)**-3, 'e-')
            w_pi = plasma_frequency(density_alongRay*(u.cm)**-3, 'D')
            w_ce = gyrofrequency(B_alongRay*(u.G), 'e-', signed = False)
            w = 2*np.pi*4.6e9*u.rad/u.s #angular frequency of the wave
            n_acc = -1*((w_pe/w_ce)+np.sqrt(1+w_pe**2/w_ce**2 - w_pi**2/w**2)).value

            #gradient = np.gradient(npara_alongRay, ws[i])
            #gradient[gradient == np.inf] = 0
            #dn_para_dl += gradient

            distanceFromN_acc[distBinIndices] += n_acc - npara_alongRay
            avgRhoOfRays[distBinIndices] += rho_alongRay

            powerInRay[distBinIndices] += delpwrRatios[:lastIndex] 

        counter[counter < 1]= 1

        avgNPara /= counter
        powerInRay /= counter
        avgRhoOfRays /= counter
        distanceFromN_acc /= counter
        dn_para_dl /= counter

        splitSuffix = fileSuffix.split('_')

        norm = plt.Normalize(0, 1)

        points_diff = np.array([distBins, avgNPara]).T.reshape(-1, 1, 2)
        segments_diff = np.concatenate([points_diff[:-1], points_diff[1:]], axis=1)
        lc_diff = LineCollection(segments_diff, norm = norm,cmap=plt.cm.jet, linewidth = linewidths[j])
        lines.append(lc_diff)
        lc_diff.set_array(powerInRay)
        axes[0].add_collection(lc_diff)

        points_grad = np.array([distBins, distanceFromN_acc]).T.reshape(-1, 1, 2)
        segments_grad = np.concatenate([points_grad[:-1], points_grad[1:]], axis=1)
        lc_grad = LineCollection(segments_grad, norm = norm,cmap=plt.cm.jet, linewidth = linewidths[j])
        lc_grad.set_array(powerInRay)
        axes[1].add_collection(lc_grad)

        axes[0].autoscale()
        axes[1].autoscale()

        axes[0].set_xlim(-10, 300)
        axes[1].set_xlim(-10, 300)
        #axes[0].plot(distBins[avgNPara < 0], avgNPara[avgNPara < 0], label = labels[j])
        #axes[1].plot(distBins[avgNPara < 0], distanceFromN_acc[avgNPara < 0], )




    cmap = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(0,1),
         cmap = plt.get_cmap('jet'))
    cmap.set_array([])
    ticks = np.linspace(0,1,5)

    #formatter = tkr.ScalarFormatter(useMathText=True)
    #formatter.set_powerlimits((0,0))
       
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar = fig.colorbar(cmap, cax=cbar_ax, shrink = 1, ticks = ticks,)
    cbar.set_label(r"fractional power remaining in rays")


    axes[0].legend(lines, labels)


    axes[0].set_title(f"Discharge {shotNum}")
    #axes[0].legend(fontsize = 10)

    axes[0].set_ylabel(r"$<n_{||}>$")
    axes[1].set_ylabel(r"$<n_{acc} - n_{||}>$")

    axes[0].set_xlabel(r"pol distance along ray (cm)")
    axes[1].set_xlabel(r"pol distance along ray (cm)")

    #fig.tight_layout()
    plt.show()
    plt.savefig('investigatePropagation.png')

investigatePropagation()
