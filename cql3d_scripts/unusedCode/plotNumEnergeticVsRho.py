import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import netCDF4
 
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
cql_nc = netCDF4.Dataset(f'{parentdir}/cql3d.nc','r')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

def plotNumEnergeticVsRho_multipleRuns():
    energyThreshold = 150 #keV

    fig,ax = plt.subplots()
    """
    labels = ['27','31','23']
    remoteDirectories = ['HXR_147634_27_50lrz_2000enorm_250iy_400jx_50lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_400jx_200lz',
        'HXR_147634_23_50lrz_2500enorm_250iy_250jx_200lz']
    """
    labels = ['1000 enorm, 450 jx', '2500 enorm 250 jx', '2500 enorm, 400 jx']
    remoteDirectories = ['HXR_147634_31_50lrz_1000enorm_250iy_450jx_200lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_250jx_200lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_400jx_200lz']

    for l in range(len(labels)):
        remoteDirectoryFile = open('../remoteDirectory.txt','w')
        newRemoteDirectory = '~/scratch/genray_batch/' + remoteDirectories[l]
        
        remoteDirectoryFile.truncate()
        remoteDirectoryFile.seek(0)
        remoteDirectoryFile.write(newRemoteDirectory)
        remoteDirectoryFile.close()
        try:
            os.system('python3 justRetrieveCQLnc.py')
        except:
            pass
        cql_nc = netCDF4.Dataset(f'{parentdir}/cql3d.nc','r')

        f = cql_nc.variables["f"][:]
        rya = cql_nc.variables["rya"][:]
        x = cql_nc.variables["x"][:]
        pitchAngleMesh = np.ma.getdata(cql_nc.variables["y"][:])
        energies = cql_nc.variables["enerkev"][:]

        threshIndex = np.argmin(np.abs(energies-energyThreshold))
        relevantX = x[threshIndex:]

        numEnergetic = np.zeros(len(rya))

        
        for rhoIndex in range(len(rya)):
            integOverPitch = 2*np.pi*np.trapz(f[rhoIndex,:]*np.sin(pitchAngleMesh[rhoIndex]), pitchAngleMesh[rhoIndex], axis = 1)
            energeticDensity = np.trapz(integOverPitch[threshIndex:]*relevantX**2, relevantX)
            total = np.trapz(integOverPitch*x**2, x)
    
            numEnergetic[rhoIndex] = (energeticDensity)*1e6#convert to m^(-3)
            
        ax.plot(rya, numEnergetic, lw = 2, label = labels[l])

    ax.set_ylabel(r"$n_i (E \geq $" + f"{energyThreshold} keV" + r"$)$  ($m^{-3}$)")
    ax.set_xlabel(r"$\rho_{tor}$")
    ax.set_ylim([1,1e19])
    ax.set_yscale('log')
    ax.legend(loc= 'best')
    fig.tight_layout()
    plt.show()
    
plotNumEnergeticVsRho_multipleRuns()
