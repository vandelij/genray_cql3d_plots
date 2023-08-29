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
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}/')
import getInputFileDictionary
genInput = getInputFileDictionary.getInputFileDictionary('genray',pathprefix=f'{parentdir}/')
cqlInput = getInputFileDictionary.getInputFileDictionary('cql3d',pathprefix=f'{parentdir}/')

import netCDF4
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]
cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d_krf001.nc','r')
"""
T_e = inputFileDict['setup']['tein']*inputFileDict['setup']['tescal']
T_i = inputFileDict['setup']['tiin']#*inputFileDict['setup']['tiscal']#I don't have tiscal defined
n_e = inputFileDict['setup']['enein(1,1)']*1e6

rhos = inputFileDict['setup']['ryain']
"""
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 14)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

def plotNEvolution():
    #n_acc = 

    nparas = np.copy(cqlrf_nc.variables["wnpar"]) #n_|| of the ray at each point along the ray trace
    nperps = np.copy(cqlrf_nc.variables["wnper"]) #n_perp of the ray at each point along the ray trace
    radialVariable = (np.copy(cqlrf_nc.variables["spsi"])) #rho_pol of the ray at each point along the ray trace
    delpwr= np.copy(cqlrf_nc.variables["delpwr"]) #power in the ray at each point
    density = np.copy(cqlrf_nc.variables["sene"])
    B = np.copy(cqlrf_nc.variables["sbtot"])
    w = 2*np.pi*4.6e9*u.rad/u.s #angular frequency of the wave
    """
    

    #parallel and perpendicular wave numbers
    kparas = (w/2.998e8)*nparas
    kperps = (w/2.998e8)*nperps

    #parallel and perpendicular wavelengths
    lambda_paras = 2*np.pi/kparas
    lambda_perps = 2*np.pi/kperps
    """

    fig,axes = plt.subplots(2,1,figsize = (7,7))
    minRatioToPlot = .2 #when the ratio of ray power to ray starting power is below this number, the trace ends
    avgN_para = 0 #average n_para when the power ratio reaches minRatioToPlot
    counter = 0
    for i in range(len(nparas)):
        density_alongRay = density[i]
        B_alongRay = B[i]
        w_pe = plasma_frequency(density_alongRay*(u.cm)**-3, 'e-')
        w_pi = plasma_frequency(density_alongRay*(u.cm)**-3, 'D')
        w_ce = gyrofrequency(B_alongRay*(u.G), 'e-', signed = False)
        w = 2*np.pi*4.6e9*u.rad/u.s #angular frequency of the wave
        #print(f"{w}, {np.max(w_ce)}, {np.max(w_pe)}, {np.max(w_pe)}")
        n_acc = -1*((w_pe/w_ce)+np.sqrt(1+w_pe**2/w_ce**2 - w_pi**2/w**2))

        delpwrRatios = delpwr[i]/np.max(delpwr[i])
        
        n_end = nparas[i][delpwrRatios > minRatioToPlot][-1]
        #this if statement prevents rays that end without depositing enough power from being counted
        if n_end < 0:
            counter += 1
            avgN_para += n_end
    
        axes[0].plot(radialVariable[i][delpwrRatios > minRatioToPlot],nparas[i][delpwrRatios > minRatioToPlot])
        #axes[0].plot(radialVariable[i][delpwrRatios > minRatioToPlot],n_acc[delpwrRatios > minRatioToPlot], linestyle = 'dotted')
        axes[1].plot(radialVariable[i][delpwrRatios > minRatioToPlot],nperps[i][delpwrRatios > minRatioToPlot])
    
    print(f"avg stopping n:{avgN_para/counter}")

    axes[0].set_ylabel(r"$n_{\parallel}$"); axes[1].set_ylabel(r"$n_{\perp}$")
    axes[0].set_xlabel(r"$\rho_{pol}$"); axes[1].set_xlabel(r"$\rho_{pol}$")
    axes[0].set_title(f"Plotting Rays until {(1-minRatioToPlot) * 100} % power deposition")
    
    axes[0].set_ylim([-5,-1])
    #axes[0].set_xlim([.4,1])
    
    fig.tight_layout()
    plt.show()

plotNEvolution()
