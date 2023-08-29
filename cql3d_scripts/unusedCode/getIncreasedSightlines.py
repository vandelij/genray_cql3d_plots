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
thet1s = np.array([105.73, 101.81, 97.78, 93.67, 89.52, 85.38, 81.28, 77.28, 73.39, 108.0, 104.08, 100.02, 95.86, 91.63, 87.39, 83.17, 79.03, 75.0, 71.12, 106.35, 102.29, 98.1, 93.82, 89.5, 85.19, 80.92, 76.76, 72.74, 108.58, 104.55, 100.36, 96.06, 91.69, 87.3, 82.94, 78.66, 74.51, 70.51, 106.76, 102.61, 98.32, 93.93, 89.49, 85.06, 80.69, 76.42, 72.3, 108.9, 104.8, 100.55, 96.17, 91.72, 87.25, 82.81, 78.46, 74.24, 70.19, 106.91, 102.72, 98.39, 93.96, 89.48, 85.01, 80.6, 76.3, 72.15])*np.pi/180.
#toroidal thetas as measured from the x axis
thet2s = np.array([133.8, 133.8, 133.8, 133.8, 133.8, 133.8, 133.8, 133.8, 133.8, 137.21, 137.21, 137.21, 137.21, 137.21, 137.21, 137.21, 137.21, 137.21, 137.21, 140.76, 140.76, 140.76, 140.76, 140.76, 140.76, 140.76, 140.76, 140.76, 144.43, 144.43, 144.43, 144.43, 144.43, 144.43, 144.43, 144.43, 144.43, 144.43, 148.2, 148.2, 148.2, 148.2, 148.2, 148.2, 148.2, 148.2, 148.2, 152.05, 152.05, 152.05, 152.05, 152.05, 152.05, 152.05, 152.05, 152.05, 152.05, 155.92, 155.92, 155.92, 155.92, 155.92, 155.92, 155.92, 155.92, 155.92])*np.pi/180.
#location of XR detector
x_sxr = cqlinput.get_contents('setup', 'x_sxr')[0]/100.  # [m]
y_sxr = 0#by convetion of CQL3D
z_sxr = cqlinput.get_contents('setup', 'z_sxr')[0]/100.  # [m]
#number of sight lines
nv = cqlinput.get_contents('setup', 'nv')[0]
pinholeDiameter = .5/100

cql3D_detector_xs = np.array([3.12301, 3.12301, 3.12301, 3.12301, 3.12301, 3.12301, 3.12301, 3.12301, 3.12301, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13008, 3.13714, 3.13714, 3.13714, 3.13714, 3.13714, 3.13714, 3.13714, 3.13714, 3.13714, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.14421, 3.15127, 3.15127, 3.15127, 3.15127, 3.15127, 3.15127, 3.15127, 3.15127, 3.15127, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.15834, 3.16540, 3.16540, 3.16540, 3.16540, 3.16540, 3.16540, 3.16540, 3.16540, 3.16540])
cql3D_detector_ys = np.array([-0.19913, -0.19913, -0.19913, -0.19913, -0.19913, -0.19913, -0.19913, -0.19913, -0.19913, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.18332, -0.16751, -0.16751, -0.16751, -0.16751, -0.16751, -0.16751, -0.16751, -0.16751, -0.16751, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.15170, -0.13589, -0.13589, -0.13589, -0.13589, -0.13589, -0.13589, -0.13589, -0.13589, -0.13589, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.12008, -0.10427, -0.10427, -0.10427, -0.10427, -0.10427, -0.10427, -0.10427, -0.10427, -0.10427])
cql3D_detector_zs = np.array([0.07770, 0.05770, 0.03770, 0.01770, -0.00230, -0.02230, -0.04230, -0.06230, -0.08230, 0.08770, 0.06770, 0.04770, 0.02770, 0.00770, -0.01230, -0.03230, -0.05230, -0.07230, -0.09230, 0.07770, 0.05770, 0.03770, 0.01770, -0.00230, -0.02230, -0.04230, -0.06230, -0.08230, 0.08770, 0.06770, 0.04770, 0.02770, 0.00770, -0.01230, -0.03230, -0.05230, -0.07230, -0.09230, 0.07770, 0.05770, 0.03770, 0.01770, -0.00230, -0.02230, -0.04230, -0.06230, -0.08230, 0.08770, 0.06770, 0.04770, 0.02770, 0.00770, -0.01230, -0.03230, -0.05230, -0.07230, -0.09230, 0.07770, 0.05770, 0.03770, 0.01770, -0.00230, -0.02230, -0.04230, -0.06230, -0.08230])

def getCQL3DDetectorDisplacements():
    return [cql3D_detector_xs - x_sxr, cql3D_detector_ys - y_sxr, cql3D_detector_zs - z_sxr]
    
cql3D_detector_disp_xs, cql3D_detector_disp_ys, cql3D_detector_disp_zs =  getCQL3DDetectorDisplacements()


nc = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc
#energy bins used by CQL3D for the XR detector
en_ = nc.get_contents("variables","en_")
#count rates of the chords. eflux[0] is thermal bremsstrahlung and eflux[1] is the nonthermal 
eflux = nc.get_contents("variables","eflux")
#min photon energy we are looking for
emin = cqlinput.get_contents('setup', 'enmin')[0]
#max photon energy we are looking for
emax = cqlinput.get_contents('setup', 'enmax')[0]

np.set_printoptions(linewidth = 1e10)

_midplaneBand = np.array([4,5,6, 14,15, 23,24,25, 33,34, 42,43,44, 52,53, 61,62,63])

_tangentBand = np.array([3,13,23,33,43,53,63,  4,14,24,34,44,54,64,])

chordsToIncrease = _tangentBand


#import re
#sentence = " ".join(re.split("\s+", sentence, flags=re.UNICODE))

#creates a plot of count rate vs rho
def multiplySightlines():
    thet1sIncreased = np.zeros(5*len(chordsToIncrease))
    thet2sIncreased = np.zeros(5*len(chordsToIncrease))

    for i in range(0, len(chordsToIncrease)):
        chordNum = chordsToIncrease[i]
        adjustments = np.array(getEtendueBasedAdjustments(chordNum))#getAdjustmentBounds(chordNum)
        thet1sIncreased[5*i] = thet1s[chordNum-1]
        thet2sIncreased[5*i] = thet2s[chordNum-1]

        thet1sIncreased[5*i+1] = thet1s[chordNum-1] + adjustments[0]
        thet2sIncreased[5*i+1] = thet2s[chordNum-1]

        thet1sIncreased[5*i+2] = thet1s[chordNum-1] + adjustments[1]
        thet2sIncreased[5*i+2] = thet2s[chordNum-1]

        thet1sIncreased[5*i+3] = thet1s[chordNum-1]
        thet2sIncreased[5*i+3] = thet2s[chordNum-1] + adjustments[2]

        thet1sIncreased[5*i+4] = thet1s[chordNum-1]
        thet2sIncreased[5*i+4] = thet2s[chordNum-1] + adjustments[3]
    
    import re
    thet1sIncreased = (thet1sIncreased*180/np.pi)
    thet2sIncreased = (thet2sIncreased*180/np.pi)

    thet1sIncreasedSTR = str(thet1sIncreased)[1:-2]
    thet2sIncreasedSTR = str(thet2sIncreased)[1:-2]

    thet1sIncreasedSTR = " ".join(re.split("\s+", thet1sIncreasedSTR, flags=re.UNICODE))
    thet2sIncreasedSTR = " ".join(re.split("\s+", thet2sIncreasedSTR, flags=re.UNICODE))

    print(f"increased thet1s: {thet1sIncreasedSTR}")
    print(f"increased thet2s: {thet2sIncreasedSTR}")


def getPolarTorAnglesFromVector(v):
    return [np.arctan2(np.sqrt(v[1]**2 + v[0]**2), v[2]), np.arctan2(v[1], v[0])]



def getAdjustmentBounds(chordNum):
    collimatorDiameter = 12.5/1000
    detectorDiameter = 5/1000
    collimatorBlockWidth = 7.6/100

    thet1 = thet1s[chordNum-1]
    thet2 = thet2s[chordNum-1]

    losDirRelToBlock = np.array([np.cos(thet2-thet2s[62-1])*np.sin(thet1),
                        np.sin(thet2-thet2s[62-1])*np.sin(thet1),
                        np.cos(thet1)])
    
    xStep = collimatorBlockWidth
    yStep = losDirRelToBlock[1]*(xStep/losDirRelToBlock[0])
    zStep = losDirRelToBlock[2]*(xStep/losDirRelToBlock[0])
    
    length = np.sqrt(xStep**2 +yStep**2 +zStep**2)
    deltaTheta = np.arctan((collimatorDiameter/2 - detectorDiameter/2)/length)

    thet1AdjToTopPinEdge, thet1AdjToBottomPinEdge, \
        thet2AdjToRightPinEdge, thet2AdjToLeftPinEdge = getAdjustmentsToPinhole(chordNum, thet1, thet2, losDirRelToBlock)

    return [max(-deltaTheta, thet1AdjToTopPinEdge),
            min(deltaTheta, thet1AdjToBottomPinEdge),
            max(-deltaTheta,thet2AdjToRightPinEdge),
            min(deltaTheta, thet2AdjToLeftPinEdge)]

def getAdjustmentsToPinhole(chordNum, thet1, thet2, losDirRelToBlock):
    unRotatedDetectorDisplacement = np.array([cql3D_detector_disp_xs[chordNum-1],
                                                cql3D_detector_disp_ys[chordNum-1],
                                                cql3D_detector_disp_zs[chordNum-1]])


    rotatedDetectorDisplacement = np.matmul(getRotationMatrix(-thet2s[62-1]), unRotatedDetectorDisplacement)
    losCenterAtPinholeOpening = np.array([0, 
                                        rotatedDetectorDisplacement[1] + losDirRelToBlock[1] * (-rotatedDetectorDisplacement[0]/losDirRelToBlock[0]),
                                        rotatedDetectorDisplacement[2] + losDirRelToBlock[2] * (-rotatedDetectorDisplacement[0]/losDirRelToBlock[0])])

    distToTopEdge = pinholeDiameter/2 - losCenterAtPinholeOpening[2]
    distToBottomEdge = losCenterAtPinholeOpening[2] +pinholeDiameter/2
    distToRightEdge = losCenterAtPinholeOpening[1] + pinholeDiameter/2
    distToLeftEdge = pinholeDiameter/2 - losCenterAtPinholeOpening[1]

    #print(f"t, b, r, l: {distToTopEdge, distToBottomEdge, distToRightEdge, distToLeftEdge}")

    #print(f"center point: {losCenterAtPinholeOpening}")

    #print(f"disp: {rotatedDetectorDisplacement}, los: {losDirRelToBlock}")

    assert distToRightEdge > 0 and distToLeftEdge > 0 and distToBottomEdge > 0 and distToTopEdge > 0

    distToPinholeOpening = norm(rotatedDetectorDisplacement)
    
    adjustmentAngles = np.array([-np.arctan(distToTopEdge/distToPinholeOpening),
                                np.arctan(distToBottomEdge/distToPinholeOpening),
                                -np.arctan(distToRightEdge/distToPinholeOpening),
                                np.arctan(distToLeftEdge/distToPinholeOpening)])
    #print(f"pinhole adjustmentAngles: {np.degrees(adjustmentAngles)}")

    return adjustmentAngles

def getAdjustmentBounds_pinholeVersion(chordNum):
    #pinholeLocation = np.array([x_sxr, y_sxr, z_sxr])
    #vector pointing from pinhole to detector when chord 62 (central chord) is pointing along the x axis
    cql3D_detector_disp_xs, cql3D_detector_disp_ys, cql3D_detector_disp_zs
    unRotatedDetectorDisplacement = np.array([cql3D_detector_disp_xs[chordNum-1],
                                                cql3D_detector_disp_ys[chordNum-1],
                                                cql3D_detector_disp_zs[chordNum-1]])

    rotatedDetectorDisplacement = np.matmul(getRotationMatrix(-thet2s[62-1]), unRotatedDetectorDisplacement)

    thet1 = thet1s[chordNum-1]
    thet2 = thet2s[chordNum-1]
    thet2Rotated = thet2 - thet2s[62-1]

    minThet1 = np.arctan(np.abs(rotatedDetectorDisplacement[0]/(pinholeDiameter/2 - rotatedDetectorDisplacement[2])))
    maxThet1 = np.pi - np.arctan(np.abs(rotatedDetectorDisplacement[0]/(-pinholeDiameter/2 - rotatedDetectorDisplacement[2])))
  

    angleToRightPinholeEdge = np.arctan(np.abs((pinholeDiameter/2 - rotatedDetectorDisplacement[1])/rotatedDetectorDisplacement[0]))
    angleToLeftPinholeEdge = np.arctan(np.abs((-pinholeDiameter/2 - rotatedDetectorDisplacement[1])/rotatedDetectorDisplacement[0]))

    adjustments = np.array([minThet1 - thet1, 
                            maxThet1-thet1, 
                            np.abs(thet2Rotated) - angleToLeftPinholeEdge,
                            np.abs(thet2Rotated) - angleToRightPinholeEdge])

    #assert adjustments[0] < 0 and adjustments[1] > 0 and adjustments[2] < 0 and adjustments[3] > 0

    return adjustments

def getEtendueBasedAdjustments(chordNum):
    etendues = getEtendues()
    etendue = etendues[chordNum]
    return [-etendue, etendue, -etendue, etendue]

def getEtendues():
    return np.array([0.0273, 0.0278, 0.0282, 0.0284, 0.0284, 0.0284, 0.0282, 0.0278, 0.0273, 0.0276, 0.0282, 0.0286, 0.0289, 0.0291, 0.0291, 0.0289, 0.0286, 0.0282, 0.0276, 0.0284, 0.0289, 0.0293, 0.0296, 0.0296, 0.0296, 0.0293, 0.0289, 0.0284, 0.0284, 0.0291, 0.0296, 0.0299, 0.0301, 0.0301, 0.0299, 0.0296, 0.0291, 0.0284, 0.0291, 0.0296, 0.0301, 0.0303, 0.0304, 0.0303, 0.0301, 0.0296, 0.0291, 0.0289, 0.0296, 0.0301, 0.0304, 0.0306, 0.0306, 0.0304, 0.0301, 0.0296, 0.0289, 0.0293, 0.0299, 0.0303, 0.0306, 0.0307, 0.0306, 0.0303, 0.0299, 0.0293])

def getRotationMatrix(theta):
    return np.array([(np.cos(theta), -np.sin(theta), 0), (np.sin(theta), np.cos(theta), 0), (0, 0 ,1)])

ans(multiplySightlines())



