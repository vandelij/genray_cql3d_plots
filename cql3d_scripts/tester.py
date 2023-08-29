import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import *

#distribution function
f = proj.model1.genray_cql3d.cql3d.cql3d_nc.get_contents("variables","f")
#pitch angles mesh at which f is defined
pitchAngleMesh = proj.model1.genray_cql3d.cql3d.cql3d_nc.get_contents("variables","y")

c = 299792458
normalizedVel = proj.model1.genray_cql3d.cql3d.cql3d_rfnc.get_contents("variables","x")
vnorm = proj.model1.genray_cql3d.cql3d.cql3d_rfnc.get_contents("variables","vnorm")
#see cql3d manual for how these energies are obtained from the normalized velocity
cql3dEnergies = (6.242e15)*(-1 + np.sqrt(1 + np.square(normalizedVel*vnorm/100)/c**2))*(9.109e-31*c**2)

#cql3d rho bins
rya = proj.model1.genray_cql3d.cql3d.cql3d_nc.get_contents("variables","rya")

#returns the electron density profile of all electrons a certain energy
def integratef():
    ne = np.zeros(len(rya))
      
    for i in range(0, len(rya)):
        integFOverVel = np.trapz(f[i,:,:]*normalizedVel[:,None]**2, normalizedVel, axis = 0)
        ne[i] = 2*np.pi*np.trapz(integFOverVel*np.sin(pitchAngleMesh[i]), pitchAngleMesh[i], axis = 0)
    
    return ne

def compareNe():

    fig,ax = plt.subplots()

    cqlinput = proj.model1.genray_cql3d.cql3d.cqlinput
    ryain = cqlinput.get_contents("setup","ryain")
    ne = [x*1e6 for x in cqlinput.get_contents("setup","enein(1,1)")]
    ax.plot(ryain, ne)

    integratedf = integratef()
    ax.plot(rya, integratedf*1e6)
    fig.show()

def YvesG(k, E):
    k_c= 2*k**2/(2*k+511)
    dE = 10
    DE = dE/(2*np.sqrt(2*np.log(2)))
    f_k = .1

    a = (1-f_k)/(k_c + np.sqrt(np.pi/2)*DE)
    b = np.heaviside(k_c-E, .5) + np.heaviside(E-k_c,.5)*np.exp(-(E-k_c)**2/(2*DE**2))
    c = f_k/(np.sqrt(2*np.pi)*DE)*np.exp(-(E-k)**2/(2*DE**2))

    return a*b+c


def plotYvesG():
    k = 122
    Es = np.linspace(0,k*1.1,1000)
    Gs = YvesG(k,Es)

    fig,ax = plt.subplots()
    ax.plot(Es, Gs)
    fig.show()



def tryCrossSection():
    detectorLoc = np.array([10,10,10])
    
    fig, ax = plt.subplots()

    pinholeWidth = 1

    torAngle = np.pi -np.arctan(detectorLoc[1]/detectorLoc[0])
    polAngle = np.pi + np.arctan(detectorLoc[2]/detectorLoc[0])

    torAdjusts = np.linspace(-np.pi/4, np.pi/4, 500)
    polAdjusts = np.linspace(-np.pi/4, np.pi/4, 500)
    for torAdjust in torAdjusts:
        for polAdjust in polAdjusts:
            los = np.array([np.cos(torAngle + torAdjust)*np.sin(polAngle + polAdjust),
                np.sin(torAngle + torAdjust)*np.sin(polAngle + polAdjust),
                np.cos(polAngle + polAdjust)])
            #print(f"los: {los}")
            scaleToPinhole = np.abs(detectorLoc/los[0])
            pinholeScaled = los*scaleToPinhole + detectorLoc
            #print(f"pinholeScaled: {pinholeScaled}")
            if np.abs(pinholeScaled[1]) < pinholeWidth and np.abs(pinholeScaled[2]) < pinholeWidth:
                scaleToPlane = np.abs((.5 + detectorLoc)/los[0])
                planeScaled = los*scaleToPlane + detectorLoc
                ax.scatter(planeScaled[1], planeScaled[2])
                #print("here")
    fig.show()


compareNe()