"""
###
Deprecated
###
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.path as mplPath

np.set_printoptions(linewidth = 500)
def convertPoloidalRhoToToroidalRho(poloidalRhosToConvert):
    torFluxAsFuncOfPolRho = getToroidalFluxFunction()

    print(f"toroidal rhos: {torFluxAsFuncOfPolRho(poloidalRhosToConvert)}")

def getToroidalFluxFunction():
    qProfile = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","qsafety")
    poloidalRhos = proj.model1.genray_cql3d_loki.cql3d_loki.cql3d_nc.get_contents("variables","rya")

    interpedQ = interp1d(poloidalRhos, qProfile, fill_value = 'extrapolate', kind = 'cubic')

    polRhoInterp = np.linspace(0,1, 500)
    qInterp = interpedQ(polRhoInterp)

    toroidalFluxes = np.zeros(len(polRhoInterp))
    for i in range(1, len(qInterp)):
        toroidalFluxes[i] = np.trapz(qInterp[:i+1], polRhoInterp[:i+1])
    
    torFluxAsFuncOfPolRho = interp1d(polRhoInterp, toroidalFluxes/np.max(toroidalFluxes))
    
    return torFluxAsFuncOfPolRho

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

ans(convertPoloidalRhoToToroidalRho(np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.32, 0.34, 0.36, 0.38, 0.4, 0.42, 0.44, 0.46, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.8, 0.82, 0.84, 0.86, 0.88, 0.9, 0.92, 0.94, 0.96, 0.98, 1.0])))

