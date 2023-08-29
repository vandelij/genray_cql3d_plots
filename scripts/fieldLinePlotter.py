import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy.interpolate import interp2d
from numpy.linalg import norm

import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import getGfileDict
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}/')

rgrid = gfileDict["rgrid"]
zgrid = gfileDict["zgrid"]
B_zGrid = gfileDict["bzrz"]
B_TGrid = gfileDict["btrz"]
B_rGrid = gfileDict["brrz"]

#returns interpolated versions of the magnetic fields
def getMagFieldFuncs():
    B_T = interp2d(rgrid,zgrid,B_TGrid)
    B_r = interp2d(rgrid,zgrid,B_rGrid)
    B_z = interp2d(rgrid,zgrid,B_zGrid)

    return [B_T, B_r, B_z]

B_T, B_r, B_z = getMagFieldFuncs()

#relevant variables to find the normalized poloidal flux
psirz = gfileDict["psirz"]
psi_mag_axis = gfileDict["ssimag"]
psi_boundary = gfileDict["ssibdry"]
    
psirzNorm = (psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis)
#interpolated function for poloidal flux
psirzNormFunc = interp2d(rgrid, zgrid[9:-10], psirzNorm[9:-10, :])

def main():
    numSteps = int(1e5)
    a = .69
    R_0 = 1.69
    ### Change the multiplier on a to get different flux surfaces ###
    startingPoint = np.array([R_0+a*.7, 0,0])
    currentPoint = startingPoint

    xList= np.zeros(numSteps+1); yList= np.zeros(numSteps+1); zList= np.zeros(numSteps+1);
    xList[0] = currentPoint[0]; yList[0] = currentPoint[1]; zList[0] = currentPoint[2];

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-R_0 - a, R_0 + a); ax.set_ylim(-R_0 - a, R_0 + a); ax.set_zlim(-1.2, 1.2);

    for i in range(0,numSteps):
        scale=1e2
        BunitVector = getMagFieldDir(currentPoint)/scale
        newPoint = BunitVector + currentPoint
        xList[i+1] = newPoint[0]; yList[i+1] = newPoint[1]; zList[i+1] = newPoint[2];

        #ax.plot([currentPoint[0], newPoint[0]], [currentPoint[1], newPoint[1]], [currentPoint[2], newPoint[2]])
        
        currentPoint = newPoint

    ax.plot(xList, yList, zList)

    plt.show()

        


#returns the right handed rotation matrix associated with the passed theta
def getRotationMatrix(theta):
    return np.array([(np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0 ,1)])

#returns normalized magnetic field vector
#only valid for quadrants I and II due to arctan
def getMagFieldDir(r):
    x,y,z = r
    R = np.sqrt(x**2 + y**2)
    magFieldDirRZPlane = np.array([B_r(R, z)[0], B_T(R,z)[0], B_z(R,z)[0]])
            
    toroidalAngle = np.arctan2(y,x)
    rotationMatrix = getRotationMatrix(toroidalAngle)
    magFieldDir = np.matmul(rotationMatrix, magFieldDirRZPlane)

    return magFieldDir/norm(magFieldDir)

#returns the rho at a given position
def getRhoFromRZ(r,z):
    return np.sqrt(psirzNormFunc(r,z))

main()
