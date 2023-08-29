"""
Plots the n_para at which significant electron landau damping occurs
This code can take up to ~30s to run. There's probably a better way to do this plotting than I came up with
This method does not do a great job of plotting near the xpoints since the flux surfaces are quite far apart there
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d, interp1d
import matplotlib.cm as cm
import matplotlib
import netCDF4

import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import getInputFileDictionary
cqlInputFileDict = getInputFileDictionary.getInputFileDictionary('cql3d',pathprefix=f'{parentdir}/')
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}/')

#below are the variables required to work with the magnetic field
rgrid = gfileDict["rgrid"]
zgrid = gfileDict["zgrid"]
magAxisR = gfileDict['rmaxis'] 
magAxisZ = gfileDict['zmaxis'] 
B_zGrid = gfileDict["bzrz"]
B_TGrid = gfileDict["btrz"]
B_rGrid = gfileDict["brrz"]

Bstrength = np.sqrt(np.square(B_zGrid) + np.square(B_TGrid) + np.square(B_rGrid))
getBStrength = interp2d(rgrid,zgrid,Bstrength, kind = 'cubic')


#relevant variables to find the normalized poloidal flux
psirz = gfileDict["psirz"]
psi_mag_axis = gfileDict["ssimag"]
psi_boundary = gfileDict["ssibdry"]
    
psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)
#interpolated function for poloidal flux
psirzNormFunc = interp2d(rgrid, zgrid[9:-10], psirzNorm[:, 9:-10].T)

rInterp = np.linspace(np.min(rgrid), np.max(rgrid), 200)
zInterp = np.linspace(-1.2, 1.2, 200)#we need to restrict the Z so that it plots flux surfaces inside the LCFS and not in the divertor
psirzNormInterp = interp2d(rgrid,zgrid, psirzNorm, kind = 'cubic')(rInterp, zInterp)

ryain = cqlInputFileDict["setup"]["ryain"]
Tes = cqlInputFileDict["setup"]["tein"]

TeInterpFunc = interp1d(ryain, Tes, kind = 'cubic')
ryainInterp = np.linspace(0,1, 200)
TeInterps = TeInterpFunc(ryainInterp)

#returns the rho at a given position
def getRhoFromRZ(r,z):
    return np.sqrt(psirzNormFunc(r,z))

#draws the flux surface pertaining to rho
def drawParticularFluxSurface(rho, ax):
    #slices on zgrid have to do with making sure the flux surface drawn is in the plasma
    return ax.contour(rInterp, zInterp, psirzNormInterp, [rho**2], colors= 'k')

#returns the path of the flux surface corresponding to rho
def getFluxSurfacePathFromRho(rho):

    dummyFig, dummyax = plt.subplots()
    path = drawParticularFluxSurface(rho, dummyax).collections[0].get_paths()[0]
    
    v = path.vertices
    x = np.array(v[:,0])
    y = np.array(v[:,1])
    plt.close(dummyFig)
    return [x,y]

dampingNs = 6.4/np.sqrt(TeInterps)#this corresponds to the quasi-linear approximation
dampingTriples = [None]*len(ryainInterp)


#Each rho position is going to get a triple corresponding to (x coordinates of the flux surface,
#                                                             y coordinates of the flux surface,
#                                                             and the n_para for ELD)
for i in range(len(ryainInterp)):
    rho = ryainInterp[i]
    x,y = getFluxSurfacePathFromRho(rho)

    dampingTriples[i] = [x,y,dampingNs[i]]

norm = matplotlib.colors.Normalize(vmin=min(dampingNs), vmax=max(dampingNs))
print(f"min damping: {min(dampingNs)}, max damping: {max(dampingNs)}")
fig,ax = plt.subplots(figsize = (5*.8,5))
for i in range(len(ryainInterp)):
    ax.scatter(dampingTriples[i][0], dampingTriples[i][1], color = cm.get_cmap('viridis')(norm(dampingTriples[i][2])), s = 5)

cmap = matplotlib.cm.ScalarMappable(norm = norm,cmap = plt.get_cmap('viridis'))
cmap.set_array([])
cbar = fig.colorbar(cmap, ax = ax, shrink = .75)
cbar.set_label(r"max n$_{||}$")
ax.set_aspect('equal')

fig.tight_layout()

plt.show()
plt.savefig('plotDampingN_2D.png')
