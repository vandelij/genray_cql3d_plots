import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from matplotlib.collections import LineCollection

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 16)
plt.rc('axes', titlesize = 16)
plt.rc('figure', titlesize = 18)
plt.rc('legend', fontsize = 14)


from scipy.signal import find_peaks

import netCDF4
remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

cql_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d.nc','r')
cqlrf_nc = netCDF4.Dataset(f'../shots/{shotNum}/cql3d_krf001.nc','r')

def getSPA():
    radialVariable = (np.copy(cqlrf_nc.variables["spsi"]))
    delpwr= np.copy(cqlrf_nc.variables["delpwr"])
    
    averageSPA = 0
    bounceRho = .96
    
    fig,ax = plt.subplots()
    offset = 1500
    for i in range(len(delpwr)):
        peakIndices, _ = find_peaks(radialVariable[i], height = bounceRho)
        relevantIndices = peakIndices[peakIndices > offset]
        bounceIndex = relevantIndices[0]
        """
    
    
    
        radiiOfInterest = radialVariable[i][offset:int(len(radialVariable[i])/2)]
    
    
        bounceIndex = np.argmax(radiiOfInterest)#next(j for j,v in enumerate(radiiOfInterest) if v > bounceRho)
        """
        ax.scatter(bounceIndex, radialVariable[i][bounceIndex])
#        print(radialVariable[bounceIndex])
        SPA = 1-delpwr[i][bounceIndex]/delpwr[i][0]
        averageSPA += SPA
        
        drho = np.gradient(radialVariable[i])
        
#        ax.scatter(bounceIndex, drho[bounceIndex+offset])
        ax.plot(radialVariable[i])
        #ax2=ax.twinx();ax2.plot(radialVariable[i])
        
    plt.show()
    averageSPA /= len(delpwr)
    print(f"average SPA: {averageSPA}")
        
getSPA()
