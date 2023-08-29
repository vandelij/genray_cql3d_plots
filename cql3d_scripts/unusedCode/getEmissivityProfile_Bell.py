"""
This file finds the emissivity profile of chosen chords using the count rates provided by CQL3D
Each count rate is associated with a rho according to where its relevant sightline becomes most tangent to the magnetic field
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d
from numpy.linalg import norm
from mpl_toolkits import mplot3d
from matplotlib.patches import FancyArrowPatch
import matplotlib.cm as cm
import matplotlib

cqlinput = model.cqlinput
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('figure', titlesize = 18)


#polar thetas as measured from the z axis
thet1s = np.array(cqlinput.get_contents('setup', 'thet1'))*np.pi/180.
#toroidal thetas as measured from the x axis
thet2s = np.array(cqlinput.get_contents('setup', 'thet2'))*np.pi/180.
#location of XR detector
x_sxr = cqlinput.get_contents('setup', 'x_sxr')[0]/100.  # [m]
y_sxr = 0#by convetion of CQL3D
z_sxr = cqlinput.get_contents('setup', 'z_sxr')[0]/100.  # [m]
#number of sight lines
nv = cqlinput.get_contents('setup', 'nv')[0]

#vars for magnetic field calculations
rgrid = proj.model1.efit_gfile1.get_contents("table", "rgrid")
zgrid = proj.model1.efit_gfile1.get_contents("table", "zgrid")
B_zGrid = proj.model1.efit_gfile1.get_contents("table","bzrz")
B_TGrid = proj.model1.efit_gfile1.get_contents("table","btrz")
B_rGrid = proj.model1.efit_gfile1.get_contents("table","brrz")

#returns interpolated versions of the magnetic fields
def getMagFieldFuncs():
    B_T = interp2d(rgrid,zgrid,B_TGrid)
    B_r = interp2d(rgrid,zgrid,B_rGrid)
    B_z = interp2d(rgrid,zgrid,B_zGrid)

    return [B_T, B_r, B_z]

B_T, B_r, B_z = getMagFieldFuncs()

#relevant variables to find the normalized poloidal flux
psirz = proj.model1.efit_gfile1.get_contents("table", "psirz")
psi_mag_axis = proj.model1.efit_gfile1.get_contents("table", "ssimag")
psi_boundary = proj.model1.efit_gfile1.get_contents("table", "ssibdry")
    
psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)
#interpolated function for poloidal flux
psirzNormFunc = interp2d(rgrid, zgrid[9:-10], psirzNorm[9:-10, :])

nc = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc
#energy bins used by CQL3D for the XR detector
en_ = nc.get_contents("variables","en_")
#count rates of the chords. eflux[0] is thermal bremsstrahlung and eflux[1] is the nonthermal 
eflux = nc.get_contents("variables","eflux")
#min photon energy we are looking for
emin = cqlinput.get_contents('setup', 'enmin')[0]
#max photon energy we are looking for
emax = cqlinput.get_contents('setup', 'enmax')[0]


"""
Choose which chords you want to use for the profile
Chord number goes from 1 to 123, so be careful about indexing arrays according to chord number
"""
#all chords that are ~tangent to the plasma
_tangentialChords = np.arange(1,67,1)
#the chords with the smallest angles of tangency relative to the magnetic field
_minAngleLine = np.array([1,2,12,13,23,24,33,34,44,54,63])
#a band of small tangent angle sightlines
_smallAngleBand = np.array([1,2,11,12,13,22,23,24,33,34,43,44,53,54,63,64])
#used for the fake, ideal sightlines
_manyFakeChords = np.arange(1,101,1)
#slice across the midplane
_midplaneSlice = np.array([4, 14,24, 34, 43, 53,62])

#set chords equal to the desired list of sightlines
chords = _midplaneSlice

np.set_printoptions(linewidth = 160)

def testBell():
    radii = getRadiiOfTangency(False, chords-1)
    rhos_r = np.array([getRhoFromRZ(radius,0)[0] for radius in radii])
    rhos_r.sort()
    
    S_is = getS_is(radii, 1, 1.67+.67)
    rhos_s = np.array([getRhoFromRZ(s,0)[0] for s in S_is])
    rhos_s.sort()

    #print(f"radii: {radii}" + f"\nS_is:{S_is}")

    LMatrix = getLMatrix(radii, S_is)
    
    MMatrix = getMMatrix(LMatrix, radii)
    
    MInverse = np.linalg.pinv(MMatrix)
    print(f"Bell m inverse matrix\n : {MInverse}")

    brightnesses = np.flip(np.take(getCountPerChord(), chords - 1))
  
    emissivities = np.matmul(MInverse, brightnesses)
    #print(f"brightnesses: {brightnesses}")
    #print(f"emissivities: {emissivities}")

    rhos = np.array([getRhoFromRZ(radius, 0)[0] for radius in radii])

    rhoSortedEmissivities = emissivities[rhos.argsort()]
    
    rhos.sort()

    
    fig, ax = plt.subplots()

    print(f"emis: {emissivities}")
    
    ax.plot(rhos, rhoSortedEmissivities)
    ax.set_xlim([0,1])
    #ax.set_ylim([0, max(emissivities)/.75])
    ax.set_title("Bell inversion process, midplane slice of existing sightlines", y = 1.01)
    ax.set_ylabel("emissivity (counts/s)")
    ax.set_xlabel(r"$\rho$", fontsize = 20)
    fig.show()
    


def getRadiiOfTangency(isOffMidplane, chords):
    if not isOffMidplane:
        thet2sOfChords = np.take(thet2s, chords)
        return np.flip(np.sin(thet2sOfChords)*x_sxr)
        
        
def getS_is(radiiOfTangency, innerWallR, outerWallR):
    S_is = np.zeros(len(radiiOfTangency) + 1)
    S_is[1:-1] = (radiiOfTangency[:-1] + radiiOfTangency[1:])/2
    S_is[0] = innerWallR
    S_is[-1] = outerWallR

    return S_is[1:]

def getLMatrix(radiiOfTangency, S_is):
    L = np.zeros((len(radiiOfTangency), len(S_is)))
    for i in range(0, len(radiiOfTangency)):
        for j in range(0 ,len(S_is)):
            s_j = S_is[j]
            r_i = radiiOfTangency[i]
            if s_j > r_i and i == j:
                L[i][j] = 2 * np.sqrt(s_j**2 - r_i**2)
            if s_j > r_i and i != j:
                val = 2 * np.sqrt(s_j**2 - r_i**2) - 2 * np.sqrt(S_is[j-1]**2 - r_i**2)
                if np.isnan(val):
                    print(f"second part: {np.sqrt(S_is[j-1]**2 - r_i**2)}")
                    print(f"S_is[j-1]: {S_is[j-1]}, r_i: {r_i}")
            
                L[i][j] = val
            if s_j < r_i:
                L[i][j] = 0

            

    return L

def getMMatrix(LMatrix, radiiOfTangency):
    n = 50
    LShape = LMatrix.shape

    MMatrix = LMatrix
    for i in range(0, LShape[0]):
        for j in range(0 , LShape[1]):
            if i != j:
                MMatrix[i,j] = LMatrix[i,j] * (radiiOfTangency[i]/radiiOfTangency[j])**n
            
    return MMatrix


#sets approriate bounds and draws flux surfaces
def setupPoloidalView(fig, ax):
    ax.set_xlabel("R (m)")
    ax.set_ylabel("z (m)")

    ax.set_ylim([-1.5, 1.5])
    ax.set_xlim([.5, 2.5])
    
    ax.set_aspect(1)
    
    drawLCFSOnPlot(ax)


def drawLCFSOnPlot(ax):
    ax.contour(rgrid, zgrid, psirzNorm, np.square(np.arange(0,1.1,.1)), colors= 'k')


#returns normalized magnetic field vector
#only valid for quadrants I and II due to arctan
def getMagFieldDir(x,y,z):
    R = np.sqrt(x**2 + y**2)
    magFieldDirRZPlane = np.array([B_r(R, z)[0], B_T(R,z)[0], B_z(R,z)[0]])
            
    toroidalAngle = np.arctan2(y,x)
    rotationMatrix = getRotationMatrix(toroidalAngle)
    magFieldDir = np.matmul(rotationMatrix, magFieldDirRZPlane)

    return magFieldDir/norm(magFieldDir)

#returns the rho at a given position
def getRhoFromRZ(r,z):
    return np.sqrt(psirzNormFunc(r,z))

#abuses matplotlib's contour function to get a collection of points making up the flux surface of a given rho
def getFluxSurfacePathFromRho(rho):

    dummyFig, dummyax = plt.subplots()
    path = drawParticularFluxSurface(rho, dummyax).collections[0].get_paths()[0]
    
    v = path.vertices
    x = v[:,0]
    y = v[:,1]
    return [x,y]


def drawParticularFluxSurface(rho, ax):
    return ax.contour(rgrid, zgrid[9:-10], psirzNorm[9:-10, :], [rho**2], colors= 'k')


#rotates psi in the RZ plane in order to find psi at any given point in R^3
def getPsi3D(x,y,z):
    toroidalAngle = np.arctan2(currentX, currentY)
    rotationMatrix = getRotationMatrix(-toroidalAngle)
    RZPlanePoint = np.matmult(rotationMatrix, np.array([x,y,z]))
    return psirzNormFunc(RZPlanePoint[0], RZPlanePoint[2])

#returns the right handed rotation matrix associated with the passed theta
def getRotationMatrix(theta):
    return np.array([(np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0 ,1)])

def getEtendues():
    ###min line etendues:###
    #return np.array([0.0273, 0.0278, 0.0286, 0.0289, 0.0296, 0.0296, 0.0301, 0.0301, 0.0303, 0.0304, 0.0306])
    
    ###66 tangential sightlines etendues:###
    #return np.array([0.0273, 0.0278, 0.0282, 0.0284, 0.0284, 0.0284, 0.0282, 0.0278, 0.0273, 0.0276, 0.0282, 0.0286, 0.0289, 0.0291, 0.0291, 0.0289, 0.0286, 0.0282, 0.0276, 0.0284, 0.0289, 0.0293, 0.0296, 0.0296, 0.0296, 0.0293, 0.0289, 0.0284, 0.0284, 0.0291, 0.0296, 0.0299, 0.0301, 0.0301, 0.0299, 0.0296, 0.0291, 0.0284, 0.0291, 0.0296, 0.0301, 0.0303, 0.0304, 0.0303, 0.0301, 0.0296, 0.0291, 0.0289, 0.0296, 0.0301, 0.0304, 0.0306, 0.0306, 0.0304, 0.0301, 0.0296, 0.0289, 0.0293, 0.0299, 0.0303, 0.0306, 0.0307, 0.0306, 0.0303, 0.0299, 0.0293])
    
    ###18 sightlines of the small angle band
    #return np.array([0.0273, 0.0278, 0.0282, 0.0282, 0.0286, 0.0289, 0.0291, 0.0293, 0.0296, 0.0296, 0.0301, 0.0301, 0.0304, 0.0303, 0.0306, 0.0304, 0.0306, 0.0303])

    return np.zeros(nv)+0.0273



#returns the counts/s of nonthermal bremsstrahlung emission each detector sees
def getCountPerChord():
    dE = en_[1]-en_[0]

    energyBinsMesh, void= np.meshgrid(en_, np.zeros(eflux.shape[1]))
    #I might be slightly approximated these etendues
    etendues = getEtendues()

    etenduesMesh = np.stack([etendues for i in range(len(en_))], axis = -1)

    efluxNormed = (eflux[1]* 624150974000 * etenduesMesh/energyBinsMesh)*dE

    efluxNormedPerChord = np.sum(efluxNormed, axis = 1)

    assert (efluxNormedPerChord.shape)[0] == nv
    assert len(efluxNormedPerChord.shape) == 1

    return efluxNormedPerChord


def plotMagField():
    fig, ax = plt.subplots()
    ax.quiver(rgrid, zgrid, B_rGrid, B_zGrid)
    fig.show()

#(str(np.round(np.linspace(133.5, 157, 100),3).tolist())[1:-1]).replace(",", "")
ans(testBell())