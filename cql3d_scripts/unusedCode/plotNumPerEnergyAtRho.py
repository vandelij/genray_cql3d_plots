import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import netCDF4
 
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import getInputFileDictionary
inputFileDict = getInputFileDictionary.getInputFileDictionary('cql3d',pathprefix=f'{parentdir}/')


cql_nc = netCDF4.Dataset(f'{parentdir}/cql3d.nc','r')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

def plotNumPerEnergyAtRho():
    fig,ax = plt.subplots()

    rhoIndex = 27
    """
    labels = ['1000 enorm, 450 jx', '2500 enorm 250 jx', '2500 enorm, 400 jx']
    remoteDirectories = ['HXR_147634_31_50lrz_1000enorm_250iy_450jx_200lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_250jx_200lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_400jx_200lz']
    """
    labels = ['27','31','23']
    remoteDirectories = ['HXR_147634_27_50lrz_2000enorm_250iy_400jx_50lz',
        'HXR_147634_31_50lrz_2500enorm_250iy_400jx_200lz',
        'HXR_147634_23_50lrz_2500enorm_250iy_250jx_200lz']
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
        vnorm = cql_nc.variables["vnorm"][:]
        rya = cql_nc.variables["rya"][:]
        x = cql_nc.variables["x"][:]
        pitchAngleMesh = np.ma.getdata(cql_nc.variables["y"][:])
        energies = cql_nc.variables["enerkev"][:]

        integOverPitch = 1e6*2*np.pi*np.trapz(f[rhoIndex,:]*np.sin(pitchAngleMesh[rhoIndex]), pitchAngleMesh[rhoIndex], axis = 1)

        energyBinsStepSize = 5
        numSets = int(len(x)/energyBinsStepSize)
        densities = np.zeros(numSets)
        energyBinCenters = np.zeros(numSets)
        for i in range(numSets):
            densities[i] = np.trapz(integOverPitch[i*5:i*5+6]*x[i*5:i*5+6]**2, x[i*5:i*5+6])
            energyBinCenters[i] = energies[i*5+2]
     
        """
        ne = np.zeros(len(rya))
        fRelevant = f[:, :, :]

        for i in range(0, len(rya)):
            integFOverVel = np.ma.getdata(np.trapz(fRelevant[i,:,:]*x[:, None]**2, 
                x, axis = 0))
            #print(f"{type(integFOverVel), type(pitchAngleMesh)}")
            ne[i] = 2*np.pi*np.trapz(integFOverVel*np.sin(pitchAngleMesh[i]), pitchAngleMesh[i], axis = 0)
        print(ne[8])
        n_e = inputFileDict['setup']['enein(1,1)']
        rhos = inputFileDict['setup']['ryain']
        ax.plot(rya,ne)
        ax.plot(rhos,n_e)
        plt.show()
        """


    #"""
        ax.plot(energyBinCenters, densities, lw = 2,label = labels[l])
    ax.legend(loc = 'best')
    ax.set_ylabel(r"Ion Density ($m^{-3}$)")
    ax.set_xlabel("Energy (keV)")
    ax.set_ylim(bottom = 1)
    ax.set_title(r"Num of ions vs energy at $\rho = $"+f" {rya[rhoIndex]}")
    ax.set_xlim([0,500])#2.5e3])
    ax.set_yscale('log')
    fig.tight_layout()
    plt.show()
    #"""
plotNumPerEnergyAtRho()
