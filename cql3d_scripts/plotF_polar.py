import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import netCDF4
from matplotlib import ticker, cm 
 
import os,sys
# currentdir = os.path.dirname(os.path.realpath(__file__))
# parentdir = os.path.dirname(currentdir)
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
# cql_nc = netCDF4.Dataset(f'{parentdir}/cql3d.nc','r')
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]
cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)

def plot_f():
    f = cql_nc.variables["f"][:]
    rya = cql_nc.variables["rya"][:]
    x = cql_nc.variables["x"][:]
    pitchAngleMesh = np.ma.getdata(cql_nc.variables["y"][:])
    enorm = cql_nc.variables["enorm"][:]
    vnorm = cql_nc.variables["vnorm"][:]


    rhoIndex = 8
    pitchAngles = pitchAngleMesh[8,:]
    relevantF = f[8,:,:]

    V, Theta = np.meshgrid(x,pitchAngles)
    VPARA = V*np.cos(Theta); VPERP = V*np.sin(Theta)

    fig,ax = plt.subplots()
    ax.pcolormesh(VPARA, VPERP, np.log(relevantF.T+1))
    #ax.contourf(VPARA, VPERP, np.log(relevantF.T), cmap=cm.PuBu_r, levels = 100)
    ax.set_aspect('equal')
    ax.set_xlabel("$v_\parallel / v_{norm}$")
    ax.set_ylabel("$v_\perp / v_{norm}$")
    ax.set_ylim([0,1]); ax.set_xlim([-1,1])
    plt.show()
    
    """

    integOverPitch = np.zeros((len(x),len(rya)))

    for i in range(len(rya)):
        for j in range(len(x)):
            integOverPitch[j,i] = 2*np.pi*x[j]**2 * np.trapz(f[i,j,:]*np.sin(pitchAngleMesh[i,:]), pitchAngleMesh[i,:])

    avgEnergies = np.zeros(len(rya))
    for k in range(len(rya)):
        avgEnergies[k] = np.trapz(integOverPitch[:,k]*energies,x)/np.trapz(integOverPitch[:,k],x)

    fig,ax = plt.subplots()
    #ax.plot(rya, avgEnergies)
    #ax.plot(energies, integOverPitch[:,6])
    #ax.imshow(integOverPitch, origin = 'lower')#ax.pcolormesh(rya, x, integOverPitch.T, shading='nearest')
    integOverPitch[integOverPitch == 0] = 1
    ax.pcolormesh(rya, energies, (integOverPitch))
    ax.set_xlabel(r'$\rho_{tor}$')
    fig.tight_layout()
    plt.show()
    """


    """
    def getTargetNe(self):
        ne = np.zeros(len(rya))
        minCQL3DEnergyIndex = np.where(cql3dEnergies < self.E_pMin)[0][-1]
        fRelevant = f[:, minCQL3DEnergyIndex:, :]

        for i in range(0, len(rya)):
            integFOverVel = np.ma.getdata(np.trapz(fRelevant[i,:,:]*normalizedVel[minCQL3DEnergyIndex:, None]**2, 
                normalizedVel[minCQL3DEnergyIndex:, None], axis = 0))
            #print(f"{type(integFOverVel), type(pitchAngleMesh)}")
            ne[i] = 2*np.pi*np.trapz(integFOverVel*np.sin(pitchAngleMesh[i]), pitchAngleMesh[i], axis = 0)

        return ne
    """
plot_f()
