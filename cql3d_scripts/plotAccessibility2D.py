"""
Plots the minimum propagating n_para according to the accessibility condition inside the LCFS
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
zInterp = np.linspace(-1.2,1.2, 200)#we need to restrict the Z so that it plots flux surfaces inside the LCFS and not in the divertor
psirzNormInterp = interp2d(rgrid,zgrid, psirzNorm, kind = 'cubic')(rInterp, zInterp)

ryain = cqlInputFileDict["setup"]["ryain"]
ns = cqlInputFileDict["setup"]["enein(1,1)"]*1e6

nInterpFunc = interp1d(ryain, ns, kind = 'cubic')
ryainInterp = np.linspace(0,1, 200)
nInterps = nInterpFunc(ryainInterp)

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

def getFieldStrength(r,z):
    strength = getBStrength(r,z)

    if type(strength) == np.float64:
        strength = np.array([strength])

    return strength

def getW2_pe(n):#electron plasma frequency squared
    return n*q**2/(m_e*eps_0)
def getW2_pD(n):#deuteron plasma frequency squared
    return n*q**2/(m_D*eps_0)
def getW_ce(B):#electron cyclotron frequency
    return -q*B/m_e

m_e = 9.109e-31 #electron mass
m_D = 3.343e-27 #deuteron mass
q = 1.6e-19 #elementary charge
eps_0 = 8.85e-12 #permitivity of free space

#returns the accessibility criterion
def getAccess(n, B):
    w2_pD = getW2_pD(n); w2_pe = getW2_pe(n); w_ce = getW_ce(B)
    w = 4.6e9*2*np.pi #wave's angular frequency

    return np.sqrt(1 - w2_pD/w**2 + w2_pe/w_ce**2) + np.sqrt(w2_pe)/np.abs(w_ce)

#Each rho position is going to get a triple corresponding to (x coordinates of the flux surface,
#                                                             y coordinates of the flux surface,
#                                                             and min n_para at each point on the flux surface)
accessTriples = [None]*len(ryainInterp)
#min and max access are used for the colorbar
minAccess = np.inf
maxAccess = 0
for i in range(len(ryainInterp)):
    rho = ryainInterp[i]
    #get the flux surface coordinates, the density at the surface, and the field at all the flux surface's points
    x,y = getFluxSurfacePathFromRho(rho)
    n = nInterps[i]
    Bs = np.vectorize(getFieldStrength)(x,y)

    accessConds = getAccess(n, Bs)
    minHere = np.min(accessConds); maxHere = np.max(accessConds)
    if minHere < minAccess:
        minAccess = minHere
    if maxHere > maxAccess:
        maxAccess = maxHere
    accessTriples[i] = [x,y,accessConds]

norm = matplotlib.colors.Normalize(vmin=minAccess, vmax=maxAccess)

fig,ax = plt.subplots(figsize = (5*.8,5))
for i in range(len(ryainInterp)):
    colors = [cm.get_cmap('viridis')(norm(access)) for access in accessTriples[i][2]]
    ax.scatter(accessTriples[i][0], accessTriples[i][1], color = colors, s = 5)

cmap = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(minAccess, maxAccess),cmap = plt.get_cmap('viridis'))
cmap.set_array([])
cbar = fig.colorbar(cmap, ax = ax, shrink = .75)
cbar.set_label(r"access")
ax.set_aspect('equal')

fig.tight_layout()

plt.show()
