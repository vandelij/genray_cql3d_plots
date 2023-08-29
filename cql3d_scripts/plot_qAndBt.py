import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib
import netCDF4

remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
sys.path.append(parentdir)
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'..')

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 15)
plt.rc('figure', titlesize = 16)
rya = cql_nc.variables["rya"][:]

qProfileCQL3D = cql_nc.variables["qsafety"][:]
qMid = gfileDict["qmid"]

rgrid = gfileDict["rgrid"]
zgrid = gfileDict["rgrid"]
B_zGrid = gfileDict["bzrz"]
B_TGrid = gfileDict["btrz"]
B_rGrid = gfileDict["brrz"]

B_pol = np.sqrt(B_rGrid**2 + B_zGrid**2)

BStrength = np.sqrt(B_TGrid**2 + B_rGrid**2 + B_zGrid**2)

midplaneB = BStrength[:][int(len(BStrength[0])/2)]
midplaneB_pol = B_pol[:][int(len(B_pol[0])/2)]
midplaneB_T = B_TGrid[:][int(len(B_TGrid[0])/2)]

fig, ax = plt.subplots()
pcm = ax.pcolormesh(rgrid, zgrid, BStrength, 
                norm=colors.LogNorm(vmin=np.min(BStrength), vmax=np.max(BStrength)),
                shading = 'nearest')

fig.colorbar(pcm, ax=ax)#, extend='max')

plt.show()

"""
fig, axes = plt.subplots(2,1, figsize = (7,8))

axes[0].plot(rya, qProfileCQL3D)#(rgrid, qMid)#
axes[0].set_ylabel(f"Safety Factor")
axes[0].set_xlabel(r"$\rho$", fontsize = 20)

axes[1].plot(rgrid, midplaneB, label = "B")
#axes[1].plot(rgrid, midplaneB_pol/midplaneB, label = "B_pol/B")
#axes[1].plot(rgrid, midplaneB_T/midplaneB, label = "B_T/B")

axes[1].set_yscale('log')

axes[1].legend()
axes[1].set_xlabel('R (m)')
axes[1].set_ylabel(r"$B_{T, mid}$ (T)")
fig.tight_layout()
plt.show()
"""