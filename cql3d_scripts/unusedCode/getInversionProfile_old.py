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
import scipy

cqlinput = model.cqlinput
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 15)
plt.rc('figure', titlesize = 16)

np.set_printoptions(linewidth = 200)

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

f = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","f")
#flips the pitch angle axis so that the values correspond to flippedPAMesh
flippedF = np.flip(f, axis = 2)
symmetricFlippedF = np.append(f, flippedF, axis = 2)

c = 299792458
normalizedVel = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_rfnc.get_contents("variables","x")
vnorm = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_rfnc.get_contents("variables","vnorm")
cql3dEnergies = (6.242e15)*(-1 + np.sqrt(1 + np.square(normalizedVel*vnorm/100)/c**2))*(9.109e-31*c**2)

pitchAngleMesh = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","y")
#shifts the values of the pitch angles so 0 degrees is now looking into a sightline when entirely tangent to the magnetic field
flippedPAMesh = np.flip(np.pi - pitchAngleMesh, axis = 1)
symmetricFlippedPAMesh = np.append(pitchAngleMesh-np.pi, flippedPAMesh, axis = 1)

rya = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rya")
curtor = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","curtor") #amps/cm**2

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

_imageBand = np.array([2,12,22,32,42,52,62,  3,13,23,33,43,53,63,  4,14,24,34,44,54,64,  5,15,25,35,45,55,65])
_imageBand2 = np.array([1,11,21,31,41,51,61, 2,12,22,32,42,52,62,  3,13,23,33,43,53,63,  4,14,24,34,44,54,64,  5,15,25,35,45,55,65, ])


_minAngleLineOptAngles = np.array([1,2,12,13,23,24,33,34,44,54,63])
_followsCurrentOptAngles = np.array([4,12,13,14,21,22,23,24,32,33,34,35,55,62,63])

_followsCurrentExisting= np.array([4,12,13,21,22,23,24,32,33,34,54,62,63])

#set chords equal to the desired list of sightlines
chords = _imageBand2
cosinePowers = [1,2,3,3,3]

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
    
    rhoBinEdges = [np.array([rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .9, rya[-1]]),
                   np.array([rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95, rya[-1]]),
                   np.array([rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95, rya[-1]]),
                   np.array([rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95, rya[-1]]),
                   np.array([rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95, rya[-1]])]

    assert np.min(rhoBinEdges[0]) < 1 and np.min(rhoBinEdges[1]) < 1 and np.min(rhoBinEdges[2]) < 1 and np.min(rhoBinEdges[3]) < 1 and np.min(rhoBinEdges[4]) < 1
    #0, .2, .4, .5, .6, .66, .7, .733, .766, .8, .9
    #rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .9, rya[-1]
    #[rya[0], .1, .2, .3, .4, .5, .575, .625, .675, .725, .775, .825, .95, rya[-1]]
    #0, .1, .2, .3, .4, .5, .6, .65, .7, .75, .8, .9
    #0, .1, .2, .3, .4, .5, .6, .7, .8, .9
    
    rhoBinCenters = [None] * len(rhoBinEdges)
    for centerIndex in range(len(rhoBinCenters)):
        rhoBinCenters[centerIndex] = rhoBinEdges[centerIndex][:-1] + (rhoBinEdges[centerIndex][1:] - rhoBinEdges[centerIndex][:-1])/2

    
    energyBinEdges = np.array([en_[0], 75, 125, 175, 225, en_[-1]])
    numEnergyBins = len(energyBinEdges)-1



    #energyBinEdges[0] = en_[0]; energyBinEdges[-1] = en_[-1]
    #for i in range(1, len(energyBinEdges)-1):
    #    index = i*cql3dEnergyBinsPerCut
    #    energyBinEdges[i] = en_[index]

    rhoLengthMatrices = [None]*numEnergyBins
    for i in range(numEnergyBins):
        rhoLengthMatrices[i] = np.zeros((len(rhoBinCenters[i]), nv))

    innerCurrentPeakRho = .625
    outerCurrentPeakRho = .725
    
    #following four vairables are to bound how far along a sightline to look for tangency
    minR = 1
    maxR = np.sqrt(x_sxr**2 + y_sxr**2)#R of the HXR diagnostic
    boundingZ = 1.12
    majorRad = 1.67+.67
    
    dx = (majorRad - minR)/10#see header comment of get_tangential_radius

    for chordNum in chords:
        rho, tangentLoc, maxPara, lengthInCurrentPeak = get_tangential_rho(chordNum, minR, maxR, boundingZ, majorRad, 
                                                            dx, innerCurrentPeakRho, outerCurrentPeakRho,rhoLengthMatrices,
                                                            rhoBinEdges, energyBinEdges)
        rhos[chordNum-1] = rho
        tangentLocs[chordNum-1] = (tangentLoc)
        maxParas[chordNum-1] = maxPara
        lengthsInCurrentPeak[chordNum-1] = lengthInCurrentPeak

    
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
        fig, axes = plt.subplots(1,2, dpi = 100,gridspec_kw={'width_ratios': [1, 1.25]})
        dischargeNumber = proj.model1.efit_gfile1.eval("gfile_path")[1:-6]
        power1 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(1)")[0]
        power2 = proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","powers(2)")[0]
        scaledPower1 = power1 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(1)")[0]
        scaledPower2 = power2 * proj.model1.genray_cql3d_loki.cql3d_loki.cqlinput.get_contents("rfsetup","pwrscale(2)")[0]

        absorbedPowerMW = (scaledPower1 + scaledPower2)/1e6
    
        assert len(dischargeNumber) == 6

        n_para = (proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmax(1)")[0] + proj.model1.genray_cql3d_loki.genray_loki.genray_in.get_contents("grill","anmin(1)")[0])/2

        fig.suptitle(f"Discharge: {dischargeNumber},  Absorbed Power: {absorbedPowerMW : .2f} MW, n_para = {n_para: 0.2f}\n"+\
            f"Looking at photons with energies {emin} - {emax} keV\n"+
            f"rho bin edges: {np.round(rhoBinEdges[0],3).tolist()}")

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
            "Showing flux surfaces for\n"+ r"$\rho = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]$", y=1.01, fontsize = 13)
        #add color bar
        cmap = matplotlib.cm.ScalarMappable(norm = matplotlib.colors.Normalize(0,maxDegrees), 
            cmap = plt.get_cmap('viridis'))
        cmap.set_array([])
        cbar = fig.colorbar(cmap, ax = axes[0], shrink = .75)
        cbar.set_label(r"min $\theta_{\hat{k}\cdot\hat{B}}$ (degrees)")

        #fig.set_size_inches(14, 8)
        
        plotInversion = True
        if plotInversion:
                    
            relevantRhoLengthMatrices = [rhoLengthMatrix[:,chords-1] for rhoLengthMatrix in rhoLengthMatrices]

            #print(f"matrix: {relevantRhoLengthMatrices[0]}")
            print(f"sum over chords: {np.sum(relevantRhoLengthMatrices[1], axis = 1)}")
            print(f"sum over rhos: {np.sum(relevantRhoLengthMatrices[1], axis = 0)}")
            print(f"total sum: {np.sum(np.sum(relevantRhoLengthMatrices[1], axis = 0), axis = 0)}")

            assert relevantRhoLengthMatrices[0].shape == (len(rhoBinCenters[0]), len(chords))
            invertedLengthMatrices = [None]*numEnergyBins

            countMatrix = getCountMatrix(chords)
            emissivitiesList = [None]*numEnergyBins

            for energyBinIndex in range(0, numEnergyBins):
                invertedLengthMatrices[energyBinIndex] = np.linalg.pinv(relevantRhoLengthMatrices[energyBinIndex])

                minEnergy = energyBinEdges[energyBinIndex]
                maxEnergy = energyBinEdges[energyBinIndex+1]

                CQL3DMinEnergyIndex = findNearestIndex(minEnergy, en_)
                CQL3DMaxEnergyIndex = findNearestIndex(maxEnergy, en_)


                countsAtEnergies = np.sum(countMatrix[:, 
                                            CQL3DMinEnergyIndex:CQL3DMaxEnergyIndex], axis = -1)
    
                if energyBinIndex == 1:
                    print(f"\nbrightness for 1st bin: {countsAtEnergies}")
    
                emissivities = np.matmul(invertedLengthMatrices[energyBinIndex].T, countsAtEnergies)
                emissivitiesList[energyBinIndex] = emissivities

                minEnergy = energyBinEdges[energyBinIndex]
                maxEnergy = energyBinEdges[energyBinIndex+1]
                                
                ax = plt.subplot(numEnergyBins, 2, 2*(energyBinIndex+1))
                ax.plot(rhoBinCenters[energyBinIndex], emissivities/max(emissivities), label = "Normalized Inversion")

                extraStr = ""
                if energyBinIndex == 0:
                    extraStr = ""#.8->.838, .862 -> .925"
                ax.set_title(f"${minEnergy: .2f} < E_{{HXR}} <  {maxEnergy: .2f}$, n = {cosinePowers[energyBinIndex]},     " + extraStr, fontsize = 12)#, n = {getCosinePower((maxEnergy-minEnergy)/2 + minEnergy):.2f}")
                ax.set_xlim([0,1])
                ax.set_ylim([min(.8*emissivities/max(emissivities)), 1.05])
                #ax.set_ylabel("emissivity (counts/s)")
                ax.set_xlabel(r"$\rho$")
                ax.set_xticks(np.linspace(0,1,11))
                
                #print(f"all emissvities >= 0: {np.min(emissivities) >= 0}")

                targetProfile = getElectronDistBetweenEnergies(minEnergy, maxEnergy)
                ax.plot(rya, targetProfile/max(targetProfile), label = "Normalized electron density", color = 'k') 
                loc = 'upper left'
                if energyBinIndex in []:
                    loc = 'upper right'
                ax.legend(fontsize = 9, loc= loc)        

            fig.set_size_inches((13,10))
            fig.tight_layout(rect=[0, 0.0, 1, 0.92])
            
            fig.subplots_adjust(left = 0.075, bottom = .075, right = .97, top = .87, hspace = .75, wspace = .1)
            fig.show()
    
            plotReIntegratedEmission = False
            if plotReIntegratedEmission:
                figRe, axRe = plt.subplots()

                reIntegratedEmission = [None]*numEnergyBins
                for i in range(0, numEnergyBins):
                    reIntegratedEmission[i] = np.matmul(relevantRhoLengthMatrices[i].T, emissivitiesList[i])
                summedReIntegrated = np.zeros(len(reIntegratedEmission[0]))
                for energySlice in reIntegratedEmission:
                    summedReIntegrated  += energySlice

                countsPerChord = np.sum(countMatrix, axis = 1)
                assert len(countsPerChord) == len(chords)
                
                rhosOfChords = np.zeros(len(chords))
                for i in range(len(chords)):
                    chordNum = chords[i]
                    magAxisR = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rmag")/100
                    tangentR = np.sqrt(tangentLocs[chordNum-1][0]**2 + tangentLocs[chordNum-1][1]**2)
                    rhosOfChords[i] = rhos[chordNum-1]
                    if tangentR < magAxisR:
                        rhosOfChords[i] *= -1
                    
                summedReIntegratedSorted = summedReIntegrated[np.argsort(rhosOfChords)]
                countsPerChordSorted = countsPerChord[np.argsort(rhosOfChords)]

                rhosOfChords.sort()
                axRe.plot(rhosOfChords, summedReIntegratedSorted, label = "Reintegrated Emissivity")
                axRe.plot(rhosOfChords, countsPerChordSorted, label = "Brightness")
                axRe.legend()

                axRe.set_xlim([-1,1])
                axRe.set_ylabel("Line-integrated emissivity (counts/s)")
                axRe.set_xlabel(r"$\rho$", fontsize = 20)
            
                figRe.tight_layout(rect=[0, 0.0, 1, 0.92])
                #figRe.subplots_adjust(left = 0.1, bottom = .1, right = .95, top = .95)
                figRe.show()

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

        #fig.set_size_inches(14, 8)
        fig.tight_layout(rect=[0, 0.0, 1, 0.92])
        fig.subplots_adjust(left = 0.05, bottom = .1, right = .95, top = .88)
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
            outerCurrentPeakRho, rhoLengthMatrices, rhoBinEdges, energyBinEdges):
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

        rho = getRhoFromRZ(currentR, currentZ)[0]
        if rho <= rya[-1]:
            tangentAngle = np.arccos(parallelity)

            for energyBinIndex in range(0, len(energyBinEdges)-1):
                dE = energyBinEdges[energyBinIndex+1] - energyBinEdges[energyBinIndex]
                
                energyEdge = energyBinEdges[energyBinIndex]
                energy = energyBinEdges[energyBinIndex]+dE/2
               
                n = cosinePowers[energyBinIndex]

                lengthBinNum = getBinNum(rho, rhoBinEdges[energyBinIndex])
                rhoLengthMatrices[energyBinIndex][lengthBinNum,chordNum-1] += dl*np.cos(tangentAngle)**n#*PAfactor

        #get the next point to look at based on the direction of the line of sight
        newX = -dx+currentX
        newY = losDir[1]*(-dx/losDir[0]) + currentY
        newZ = losDir[2]*(-dx/losDir[0]) + currentZ
        
        currentX = newX
        currentY = newY
        currentZ = newZ
    
    bestR = np.sqrt(tangent_loc[0]**2 + tangent_loc[1]**2)
    rho = getRhoFromRZ(bestR, tangent_loc[2])
    
    return [rho, tangent_loc, maxPara, lengthInCurrentPeak]

def getBinNum(val, bins):
    return np.where(bins >= val)[0][0] - 1

def findNearestIndex(value, array):
    idx = (np.abs(array - value)).argmin()
    return idx

def getElectronDistBetweenEnergies(minEnergy, maxEnergy):
    y = np.zeros(len(rya))
    minCQL3DEnergyIndex = np.where(cql3dEnergies < minEnergy)[0][-1]
    maxCQL3DEnergyIndex = -2#np.where(cql3dEnergies > maxEnergy)[0][0]

    fRelevant = f[:, minCQL3DEnergyIndex:maxCQL3DEnergyIndex+1, :]
    for i in range(0, len(rya)):
        y[i] = np.sum(fRelevant[i,:,:])

    return y

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