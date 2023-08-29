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
nen = cqlinput.get_contents('setup', 'nen')[0]

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
_all = np.arange(1,nv+1,1)
#slice across the midplane
_midplaneSlice = np.array([4, 14,24, 34, 43, 53,62])

_imageBand = np.array([1,11,21,31,41,51,61, 2,12,22,32,42,52,62,  3,13,23,33,43,53,63,  4,14,24,34,44,54,64,  5,15,25,35,45,55,65])
_imageBand2 = np.array([1,11,21, 2,12,22,32,42,52,62,  3,13,23,33,43,53,63,  4,14,24,34,44,54,64,  25,35,45,55,65, 10,20])


_minAngleLineOptAngles = np.array([1,2,12,13,23,24,33,34,44,54,63])
_followsCurrentOptAngles = np.array([4,12,13,14,21,22,23,24,32,33,34,35,55,62,63])

_followsCurrentExisting= np.array([4,12,13,21,22,23,24,32,33,34,54,62,63])

#set chords equal to the desired list of sightlines
chords = _imageBand2
inversionParaPow = 5

#creates a plot of count rate vs rho
def getEmissivityProfile():
    """
    notice that the three arrays below have length of nv and not len(chords)
    """
    #rhos of tangency
    rhos = np.zeros(nv)
    #(x,y,z) locations of tangency. Note: this is in CQL3D coordinates, not typical D_IIID coordinates
    tangentLocs = [None]*nv
    #the largest result of applying the dot product to the magnetic field and the sightline direction
    maxParas = np.zeros(nv)+1
    lengthsInCurrentPeak = np.zeros(nv)
    #[0. , .2,.4, .5, .6, .66, .7, .733, .766, .8, .9]
    rhoBins = np.array([0, .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95])


    assert len(np.where(rhoBins > 1)[0]) == 0
    #0. , .2, .4, .5, .6, .64, .68, .72, .76, .8,.9,
    #np.array([0. , .2, .4, .5, .6, .7, .75,.8,.9,])
    #np.array([0. , .2, .4, .5, .6, .65, .7, .75,.8,.9, .95])
    rhoBinCenters = rhoBins + np.append((rhoBins[1:] - rhoBins[:-1])/2, [(1- rhoBins[-1])/2])
    rhoLengthMatrix = np.zeros((len(rhoBinCenters), nv))

    innerCurrentPeakRho = .625
    outerCurrentPeakRho = .725
    
    #following four vairables are to bound how far along a sightline to look for tangency
    minR = 1
    maxR = np.sqrt(x_sxr**2 + y_sxr**2)#R of the HXR diagnostic
    boundingZ = 1.12
    majorRad = 1.67+.67
    
    dx = (majorRad - minR)/100#see header comment of get_tangential_radius

    distFunc = getRelevantDistFunct()

    #the 3D plotting is meant to help diagnose if the location of tangency is really being found
    #it really only works for one sightline at a time
    fig3D = plt.figure()
    ax3D = plt.axes(projection = '3d')

    plot3D = False
    did3DPlot = False
    for chordNum in chords:
        if 54 <= chordNum <= 52:#set both these numbers to the desired sightline number to plot it in 3D
            plot3D = True
            did3DPlot = True

        rho, tangentLoc, maxPara, lengthInCurrentPeak = get_tangential_rho(chordNum, minR, maxR, boundingZ, majorRad, 
                                                            dx, innerCurrentPeakRho, outerCurrentPeakRho,rhoLengthMatrix,
                                                            rhoBins, distFunc, fig3D, ax3D, plot3D)
        rhos[chordNum-1] = rho
        tangentLocs[chordNum-1] = (tangentLoc)
        maxParas[chordNum-1] = maxPara
        lengthsInCurrentPeak[chordNum-1] = lengthInCurrentPeak

        if plot3D:
            #plotting the associated flux surface
            pathR, pathZ = getFluxSurfacePathFromRho(rhos[chordNum-1])
            lenZ = len(pathZ)
            revolveTheta = np.linspace(0, np.pi, lenZ).reshape(1,lenZ)
            R_column = pathR.reshape(lenZ,1)
            x = R_column.dot(np.cos(revolveTheta))
            y = R_column.dot(np.sin(revolveTheta))
            zs, rs = np.meshgrid(pathZ, pathR)
            ax3D.plot_wireframe(x, y, zs.T, rcount = 25, ccount = 25, color = 'black')
            
        plot3D = False


    if did3DPlot:
        ax3D.set_ylabel("y (m)")
        ax3D.set_xlabel("x (m)")
        ax3D.set_zlabel("z (m)")
        ax3D.set_xlim([-3, 3])
        ax3D.set_ylim([-3,3])
        ax3D.set_zlim([-boundingZ, boundingZ])
        fig3D.show()
    
    plotRhosOfChords = False
    if plotRhosOfChords:
        fig,ax = plt.subplots()
        rhos = np.zeros(nv)
        for chordNum in chords:
            tangentLoc = tangentLocs[chordNum-1]
            rho = getRhoFromRZ(np.sqrt((tangentLoc[0])**2 + (tangentLoc[1])**2), tangentLoc[2])[0]
            
            rhos[chordNum-1] = rho  
        rhos = rhos[chords-1]
        rhos.sort()
        ax.scatter(rhos, [1]*len(rhos))
        fig.show()
    
    plotTangentsAndData = True
    if plotTangentsAndData:
        fig, axes = plt.subplots(1,2)
        dischargeNumber = proj.model1.efit_gfile1.eval("gfile_path")[1:-6]
        power1 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(1)")[0]
        power2 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(2)")[0]
        scaledPower1 = power1 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(1)")[0]
        scaledPower2 = power2 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(2)")[0]

        absorbedPowerMW = (scaledPower1 + scaledPower2)/1e6
    
        assert len(dischargeNumber) == 6

        n_para = (proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmax(1)")[0] + proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmin(1)")[0])/2

        fig.suptitle(f"Discharge: {dischargeNumber},  Absorbed Power: {absorbedPowerMW : .2f} MW, n_para = {n_para}\n"+\
            f"Looking at photons with energies {emin} - {emax} keV")

        setupPoloidalView(fig, axes[0])#sets approriate bounds and draws flux surfaces
    
        etendues = getEtendues()
        maxDegrees = np.degrees(np.arccos(np.min(maxParas)))
               
        for chordNum in chords:
            tangentX = tangentLocs[chordNum-1][0]
            tangentY = tangentLocs[chordNum-1][1]
            tangentZ = tangentLocs[chordNum-1][2]

            tangentR = np.sqrt(tangentX**2 + tangentY**2)
            
            #plot area of measure assuming the sightline viewing volume is a cone
            distanceVectorToTangent = np.array([tangentX-x_sxr, tangentY-y_sxr, tangentZ-z_sxr])
            distance = norm(distanceVectorToTangent)
            etendue = etendues[chordNum-1]

            tangentDegrees = np.degrees(np.arccos(maxParas[chordNum-1]))
            #draw a circle according to the width of the area of measure with the color according to the degree of tangency
            circleColor = cm.viridis(tangentDegrees/maxDegrees)
            edgeColor = circleColor#'k'
            axes[0].add_patch(plt.Circle((tangentR, tangentZ), distance*np.tan(etendue),
                facecolor = circleColor, edgecolor = edgeColor))
            #for when the colormap to show degree of tangency makes it hard to read the sightline number
            textColor = 'w'
            if tangentDegrees > .85*maxDegrees:
                textColor = 'k'

            axes[0].text(tangentR, tangentZ, str(chordNum),
                color = textColor, horizontalalignment='center', verticalalignment='center')
    
        axes[0].set_title("Projection of areas of measure and tangency locations\n" + 
            r"Showing flux surfaces for $\rho = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]$", y=1.01)
        #add color bar
        cmap = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(0,maxDegrees), 
            cmap = plt.get_cmap('viridis'))
        cmap.set_array([])
        cbar = fig.colorbar(cmap, ax = axes[0])
        cbar.set_label(r"min $\theta_{\hat{k}\cdot\hat{B}}$ (degrees)")

        fig.set_size_inches(14, 8)
        
        plotInversion = True
        if plotInversion:
                    
            relevantRhoLengthMatrix = rhoLengthMatrix[:, chords-1]
            assert relevantRhoLengthMatrix.shape == (len(rhoBinCenters), len(chords))
            invertedLengthMatrix = np.linalg.pinv(relevantRhoLengthMatrix)
            countMatrix = getCountMatrix(chords)

            countsSumOverEnergy= np.sum(countMatrix, axis = -1)
    
            emissivities = np.matmul(invertedLengthMatrix.T, countsSumOverEnergy)
            axes[1].plot(rhoBinCenters, emissivities/max(emissivities))

            rya = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rya")
            curtor = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","curtor") #amps/cm**2
            toroidalCurrent = curtor[-1,:]/100
            currentAxis = axes[1].twinx()
            currentAxis.plot(rya, toroidalCurrent, label = "Current Profile", color = 'k', linestyle = '-.') 
            currentAxis.set_ylabel('MA/m$^2$')
            currentAxis.set_ylim([0,np.max(toroidalCurrent)/.8])



            axes[1].set_title(f"{len(rhoBinCenters)} rho bins, inversionParaPow = {inversionParaPow} \n bin centers: {np.round(rhoBinCenters.tolist(),3)}", fontsize = 12, y=1.01)
            axes[1].set_xlim([0,1])
            #axes[1].set_ylim([-.1, 1])
            axes[1].set_ylabel("emissivity (counts/s)")
            axes[1].set_xlabel(r"$\rho$", fontsize = 20)
            axes[1].set_xticks(np.linspace(0,1,11))

            fig.tight_layout(rect=[0, 0.0, 1, 0.92])
            fig.subplots_adjust(left = 0.05, bottom = .1, right = .95, top = .84)
            fig.show()

            print(f"all emissvities >= 0: {np.min(emissivities) >= 0}")
    
            
        plotBrightness = False
        if plotBrightness:
        
            for chordNum in chords:
                axes[1].text(rhos[chordNum-1], countPerChord[chordNum-1], str(chordNum), 
                    color = 'tab:blue', horizontalalignment='center', verticalalignment='center')

            #axes[1].set_ylim([0, countPerChord[49]/.75])
            #axes[1].set_ylim([0, 18e3])
            axes[1].set_ylim([0, np.max(countPerChord[(chords-1)])/.75])
            axes[1].set_xlim([0,1])
            axes[1].set_title("Line-integrated emissivity at tangency", y = 1.01)
            axes[1].set_ylabel("emissivity (counts/s)")
            axes[1].set_xlabel(r"$\rho$", fontsize = 20)
        
            fig.tight_layout(rect=[0, 0.0, 1, 0.92])
            fig.subplots_adjust(left = 0.05, bottom = .1, right = .95, top = .84)
            fig.show()

    compareDeinvertedInversion = True
    if compareDeinvertedInversion:
        fig, ax = plt.subplots()

        countMatrix = getCountMatrix(chords)
        countsSumOverEnergy = np.sum(countMatrix, axis = -1)

        relevantRhoLengthMatrix = rhoLengthMatrix[:, chords-1]
        assert relevantRhoLengthMatrix.shape == (len(rhoBinCenters), len(chords))
        invertedLengthMatrix = np.linalg.pinv(relevantRhoLengthMatrix)    
        emissivities = np.matmul(invertedLengthMatrix.T, countsSumOverEnergy)

        reIntegratedEmission = np.matmul(relevantRhoLengthMatrix.T, emissivities)
        print(reIntegratedEmission.shape)
        assert reIntegratedEmission.shape == (len(chords),)

        xRe = np.zeros(len(chords))
        yRe = np.zeros(len(chords))
    
        xBri = np.zeros(len(chords))
        yBri = np.zeros(len(chords))

        for i in range(len(chords)):
            chordNum = chords[i]
            tangentX = tangentLocs[chordNum-1][0]
            tangentY = tangentLocs[chordNum-1][1]
            
            tangentR = np.sqrt(tangentX**2 + tangentY**2)
            side = 1
            if tangentR < proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rmag")/100:
                side = -1
            xRe[i] = side*rhos[chordNum-1]
            yRe[i] = reIntegratedEmission[i]

            xBri[i] = side*rhos[chordNum-1]
            yBri[i] = countsSumOverEnergy[chords.tolist().index(chordNum)]


        yRe = yRe[xRe.argsort()]
        xRe.sort()
        ax.plot(xRe,yRe, label = "Reintegrated Emissivity") 

        yBri = yBri[xBri.argsort()]
        xBri.sort()
        ax.plot(xBri,yBri, label = "Brightness") 


        ax.set_ylim([0, np.max(reIntegratedEmission)/.75])
        ax.set_xlim([-1,1])
        
        ax.set_ylabel("emissivity (counts/s)")
        ax.set_xlabel(r"$\rho$ of tangency")

        ax.legend()

        fig.tight_layout(rect=[0, 0.0, 1, .92])
        ax.set_title(f"{len(rhoBinCenters)} rho bins, cosine power = {inversionParaPow} \n bin centers: {np.round(rhoBinCenters.tolist(),3)}", fontsize = 12, y=1.05)
        #fig.subplots_adjust(left = 0.05, bottom = .1, right = .95, top = .84)
        fig.show()


        


    plotLengthInPeak = False
    if plotLengthInPeak:
        fig, axes = plt.subplots(1,2)
        dischargeNumber = proj.model1.efit_gfile1.eval("gfile_path")[1:-6]
        power1 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(1)")[0]
        power2 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(2)")[0]
        scaledPower1 = power1 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(1)")[0]
        scaledPower2 = power2 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(2)")[0]

        absorbedPowerMW = (scaledPower1 + scaledPower2)/1e6
    
        assert len(dischargeNumber) == 6

        n_para = (proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmax(1)")[0] + proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmin(1)")[0])/2

        fig.suptitle(f"Discharge: {dischargeNumber},  Absorbed Power: {absorbedPowerMW : .2f} MW, n_para = {n_para}\n"+\
            f"Looking at photons with energies {emin} - {emax} keV")
        for chordNum in chords:
            tangentX = tangentLocs[chordNum-1][0]
            tangentY = tangentLocs[chordNum-1][1]
            tangentZ = tangentLocs[chordNum-1][2]

            tangentR = np.sqrt(tangentX**2 + tangentY**2)
            if tangentR > proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rmag")/100:
                axes[0].scatter(rhos[chordNum-1], lengthsInCurrentPeak[chordNum-1], color = 'tab:blue')
                axes[1].scatter(rhos[chordNum-1], countPerChord[chordNum-1], color = 'tab:blue')
            else:  
                axes[0].scatter(-rhos[chordNum-1], lengthsInCurrentPeak[chordNum-1], color = 'r')
                axes[1].scatter(-rhos[chordNum-1], countPerChord[chordNum-1], color = 'r')
           
        axes[1].scatter(0, -4000, color = 'tab:blue', label = "LFS")
        axes[1].scatter(0, -4000, color = 'tab:red', label = "HFS")
        axes[1].legend()
        
        axes[0].scatter(0, -4000, color = 'tab:blue', label = "LFS")
        axes[0].scatter(0, -4000, color = 'tab:red', label = "HFS")
        axes[0].legend()

        axes[0].set_ylim([0, np.max(lengthsInCurrentPeak[(chords-1)])/.75])
        axes[0].set_xlim([-1,1])
        axes[1].set_ylim([0, np.max(countPerChord[(chords-1)])/.75])
        axes[1].set_xlim([-1,1])

        axes[0].set_title(r"Length of chord between $|\rho| = $" + f"${innerCurrentPeakRho}$ and " +\
                                r"$|\rho| = $" + f"${outerCurrentPeakRho}$", y = 1.01)
        axes[0].set_ylabel("Length (m)")
        axes[0].set_xlabel(r"$\rho$ of tangency")
        axes[1].set_title("Line-integrated emissivity at tangency", y = 1.01)
        axes[1].set_ylabel("emissivity (counts/s)")
        axes[1].set_xlabel(r"$\rho$ of tangency")

        fig.set_size_inches(14, 8)
        fig.tight_layout(rect=[0, 0.0, 1, 0.92])
        fig.subplots_adjust(left = 0.05, bottom = .1, right = .95, top = .84)
        #axes[1].axes.set_aspect('auto')
        fig.show()



"""
returns the rho at which a line of sight is tangent to the magnetic field
this is done by starting at the detector and taking a step dx in the -x direction and corresponding steps in the y and z directions.
At each point, the magnitude of the magnetic field is normalized to 1 and its direction is 
compared to the direction of the line of sight via the dot product.
The result is then compared against maximum product produced at previous points.
The location of the highest degree of parallelity between the magnetic field and line of sight is the returned
Future work may be required to make sure there is a minimum amount of parallelity

"""
def get_tangential_rho(chordNum, minR, maxR, boundingZ, majorRad, dx, innerCurrentPeakRho,
            outerCurrentPeakRho, rhoLengthMatrix, rhoBins, distFunc, fig, ax, plot3D):
    torTheta = thet2s[chordNum-1]
    polarTheta = thet1s[chordNum-1]
    losDir = np.array([np.cos(torTheta)*np.sin(polarTheta), np.sin(torTheta)*np.sin(polarTheta), np.cos(polarTheta)])

    #current point being looked at
    currentX = x_sxr
    currentY = y_sxr
    currentZ = z_sxr

    #location of tangency
    tangent_loc = -1
    #max dot product result seen so far
    maxPara = -1

    dl = np.sqrt(dx**2 + (losDir[1]*(-dx/losDir[0]))**2 + (losDir[2]*(-dx/losDir[0]))**2)
    lengthInCurrentPeak = 0

    y = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","y")

    
    #while we are looking at a point inside a reasonable volume
    while(np.abs(currentZ) < boundingZ and minR < np.sqrt(currentX**2 + currentY**2) <= maxR):
        currentR = np.sqrt(currentX**2 + currentY**2)
        #we're starting at the detector, outside of the device
        #no point in looking for tangency outside of the outer wall
        if currentR < majorRad:
            magFieldUnitDir = getMagFieldDir(currentX, currentY, currentZ)

            #how parallel are the direction of the magnetic field and the line of sight
            parallelity = np.abs(np.dot(magFieldUnitDir, losDir))
           
            #if this is the most tangential point found so far, record it
            if parallelity > maxPara:
                tangent_loc = [currentX, currentY, currentZ]
                maxPara = parallelity

            if plot3D:#plot the direction of the magnetic field
                ax3D.quiver([currentX], [currentY], [currentZ], 
                        [magFieldUnitDir[0]], [magFieldUnitDir[1]], [magFieldUnitDir[2]],
                        color = 'b', length = .1)

        rho = getRhoFromRZ(currentR, currentZ)
        if rho <= 1:
            lengthBinNum = getBinNum(rho, rhoBins)
            tangentAngle = np.arccos(parallelity)
            
            rhoLengthMatrix[lengthBinNum][chordNum-1] += dl*(np.cos(tangentAngle)**inversionParaPow)
            
            if innerCurrentPeakRho < rho < outerCurrentPeakRho:
                lengthInCurrentPeak += dl#*(parallelity**pow)
        

        #get the next point to look at based on the direction of the line of sight
        newX = -dx+currentX
        newY = losDir[1]*(-dx/losDir[0]) + currentY
        newZ = losDir[2]*(-dx/losDir[0]) + currentZ
        
        currentX = newX
        currentY = newY
        currentZ = newZ
    
    if plot3D:#plot line of sight and location of tangency
        ax.plot([x_sxr, 5*losDir[0]+x_sxr], [y_sxr, 5*losDir[1]+y_sxr], [z_sxr, 5*losDir[2]+z_sxr], color = 'g', label = 'line of sight')
        ax.scatter3D([tangent_loc[0]],[tangent_loc[1]],[tangent_loc[2]], color = 'r')
    

    bestR = np.sqrt(tangent_loc[0]**2 + tangent_loc[1]**2)
    rho = getRhoFromRZ(bestR, tangent_loc[2])
    #print(f"num: {chordNum}, maxPara: {maxPara}, rho:{rho}")
    #print(f"acos(maxPara): {np.degrees(np.arccos(maxPara))} degrees\n")
    
    return [rho, tangent_loc, maxPara, lengthInCurrentPeak]


def getRelevantDistFunct():
    f = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","f")

    c = 299792458
    normalizedVel = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_rfnc.get_contents("variables","x")
    vnorm = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_rfnc.get_contents("variables","vnorm")
    energies = (6.242e15)*(-1 + np.sqrt(1 + np.square(normalizedVel*vnorm/100)/c**2))*(9.109e-31*c**2)

    minEnergy = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","enmin")
    maxEnergy = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","enmax")
    minEnergyIndex = np.where(energies < minEnergy)[0][-1]
    maxEnergyIndex = np.where(energies > maxEnergy)[0][0]

    print(f"min: {minEnergyIndex}, max: {maxEnergyIndex}")
    print(f"minE: {energies[minEnergyIndex]}, maxE: {energies[maxEnergyIndex]}")

    fRelevant = f[:, minEnergyIndex:maxEnergyIndex+1, :]
    return fRelevant


def getBinNum(val, bins):
    
    return np.where(bins < val)[0][-1]

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
    return np.array([0.0273, 0.0278, 0.0282, 0.0284, 0.0284, 0.0284, 0.0282, 0.0278, 0.0273, 0.0276, 0.0282, 0.0286, 0.0289, 0.0291, 0.0291, 0.0289, 0.0286, 0.0282, 0.0276, 0.0284, 0.0289, 0.0293, 0.0296, 0.0296, 0.0296, 0.0293, 0.0289, 0.0284, 0.0284, 0.0291, 0.0296, 0.0299, 0.0301, 0.0301, 0.0299, 0.0296, 0.0291, 0.0284, 0.0291, 0.0296, 0.0301, 0.0303, 0.0304, 0.0303, 0.0301, 0.0296, 0.0291, 0.0289, 0.0296, 0.0301, 0.0304, 0.0306, 0.0306, 0.0304, 0.0301, 0.0296, 0.0289, 0.0293, 0.0299, 0.0303, 0.0306, 0.0307, 0.0306, 0.0303, 0.0299, 0.0293])
    
    ###18 sightlines of the small angle band
    #return np.array([0.0273, 0.0278, 0.0282, 0.0282, 0.0286, 0.0289, 0.0291, 0.0293, 0.0296, 0.0296, 0.0301, 0.0301, 0.0304, 0.0303, 0.0306, 0.0304, 0.0306, 0.0303])

    #return np.zeros(nv)+0.0273



#returns the counts/s of nonthermal bremsstrahlung emission each detector sees
def getCountMatrix(chords):
    dE = en_[1]-en_[0]

    energyBinsMesh, void= np.meshgrid(en_, np.zeros(eflux.shape[1]))
    #I might be slightly approximated these etendues
    etendues = getEtendues()

    etenduesMesh = np.stack([etendues for i in range(len(en_))], axis = -1)

    efluxNormed = (eflux[1]* 624150974000 * etenduesMesh/energyBinsMesh)*dE


    return efluxNormed[chords-1, :]


def plotMagField():
    fig, ax = plt.subplots()
    ax.quiver(rgrid, zgrid, B_rGrid, B_zGrid)
    fig.show()

#(str(np.round(np.linspace(133.5, 157, 100),3).tolist())[1:-1]).replace(",", "")

ans(getEmissivityProfile())