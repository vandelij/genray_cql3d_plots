import aurora, numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp2d
from matplotlib.collections import LineCollection
from omfit_classes import omfit_eqdsk

#import matplotlib.cm as cm
sz=17
plt.rc('xtick', labelsize = sz-3)
plt.rc('ytick', labelsize = sz-3)
plt.rc('axes', labelsize = sz)
plt.rc('axes', titlesize = sz)
plt.rc('figure', titlesize = sz)
plt.rc('legend', fontsize = sz)

#sz=32
#tcks=sz-6
##plt.style.use('seaborn-darkgrid')
#plt.rc('axes',labelsize=sz, titlesize=sz, grid=False)
#plt.rc('xtick', labelsize=tcks)
#plt.rc('ytick', labelsize=tcks)
#plt.rc('figure',titlesize=sz)
#plt.rc('lines', linewidth=4)
#plt.rcParams['grid.linewidth'] = 2

import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
print("current dir: ", currentdir)
parentdir = os.path.dirname(currentdir)
print("parent dir: ",parentdir)
sys.path.append(parentdir)

# direct to patch containing eqdsk
#subdir='run1_BALOO_v1_nominal_new'
#subdir='run2_NTARC_zeta_nominal'
#subdir='run6_NTARC_zeta_n085_T23_nedge11'
subdir='run7_zeta_iter1b__small_pedestal_n_17'
eqdskDir=os.path.join(parentdir,subdir)

#eqdskDir='/nobackup1/jamalj/nt_arc/run1_BALOO_v1_nominal_new/'
#eqdskDir='/nobackup1/jamalj/nt_arc/run2_NTARC_zeta_nominal/'
#eqdskFile='gNTARC_k14d05H15-BALOO_v1' # old equlibrium for baloo_v1_nominal
#eqdskFile='gNARC_k14d05R455a120'
eqdskFile='gNTARC_iter1b_small_pedestal_n_17'

eqdskPath=os.path.join(eqdskDir,eqdskFile)

import getGfileDict
#gfileDict = getGfileDict.getGfileDict(pathprefix=f'{parentdir}/')
gfileDict = getGfileDict.getGfileDict(pathprefix=f'{eqdskDir}/')

import netCDF4
subOutPutDir='out/'
cqlrf_nc = netCDF4.Dataset(f'{eqdskDir}/'+subOutPutDir+'cql3d_krf001.nc','r')
cql_nc = netCDF4.Dataset(f'{eqdskDir}/'+subOutPutDir+'cql3d.nc','r')

rf_power = np.copy(cql_nc.variables['powers_int'][-1,0,4])

#print("\nkeys in cql3d_krf001.nc")
#cql3d_krf_keys_str=[]
#[cql3d_krf_keys_str.append(key) for key in cqlrf_nc.variables.keys()]
#cql3d_krf_keys_str=np.sort(cql3d_krf_keys_str)
#[print(key) for key in cql3d_krf_keys_str]
#
#print("\nkeys in cql3d.nc")
#cql3d_keys_str=[]
#[cql3d_keys_str.append(key) for key in cql_nc.variables.keys()]
#cql3d_keys_str=np.sort(cql3d_keys_str)
#[print(key) for key in cql3d_keys_str]

#returns the index of the array whose element is closest to value
def findNearestIndex(value, array):
    idx = (np.abs(array - value)).argmin()

    return idx
#gives a set of bin edges and a value, returns the bin that the val should be placed in
def getBinNum(val, bins):
    return np.where(bins >= val)[0][0] - 1

def plotRays(allrays = True, showpeak = True, shownegative = False,
         showpositive = False, raystop = None, color= 'power', 
         npower = True,  ridx = None):
   
    xlim = gfileDict['xlim']
    ylim = gfileDict['ylim']
    rbbbs = gfileDict['rbbbs']
    zbbbs = gfileDict['zbbbs']
    
    wr  = np.copy(cqlrf_nc.variables["wr"])
    wz  = np.copy(cqlrf_nc.variables["wz"])
    wnpar = np.copy(cqlrf_nc.variables["wnpar"])
    delpwr= np.copy(cqlrf_nc.variables["delpwr"]) # power in the ray channel ergs/sec
    delpwr0= cqlrf_nc.variables["delpwr"]
    nrayelt= cqlrf_nc.variables["nrayelt"] # Number of ray elements for each ray
    
    if allrays:
        ridx = range(len(wnpar[:,0]))
    num = 0
    for x in range(wnpar.shape[0]):
       idx = np.where(wnpar[x,:] == 0)
       if delpwr[x,0]!=0.0 and npower:
          delpwr[x,:] = delpwr[x,:]/delpwr[x,0]
       if len(idx[0]) != 0:
          wnpar[x,nrayelt[x]:] = wnpar[x, nrayelt[x]-1]
          wz[x,nrayelt[x]:] = wz[x, nrayelt[x]-1]
          wr[x,nrayelt[x]:] = wr[x, nrayelt[x]-1]
        
    fig,ax = plt.subplots(figsize = (5.25,7.1), dpi=300) #<------------------------------------
    #fig,ax = plt.subplots(figsize = (6.38, 8), dpi=300)
    plt.subplots_adjust(left=0.22,bottom = .1)
    ax.set_ylabel("z (m)")
    ax.set_xlabel("R (m)")
       
    norm = plt.Normalize(delpwr.min(), delpwr.max())

    wr *= .01; wz*=.01
    count = 0
    drawFluxSurfaces(ax)
    
    norm = plt.Normalize(0, 1)
    maxDelPwrPlot = .9
    for ray in ridx:
        mostPowerDep = findNearestIndex(1 - maxDelPwrPlot, delpwr[ray])

        points = np.array([wr[ray][:mostPowerDep], wz[ray][:mostPowerDep]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)


        # Create a continuous norm to map from data points to colors
        lc = LineCollection(segments, norm = norm, cmap = plt.cm.get_cmap('Spectral_r'))
        # Set the values used for colormapping
        lc.set_array(delpwr[ray][:mostPowerDep])
        lc.set_linewidth(1)
        ax.add_collection(lc)

    plotHarmonicContours(ax)
    ax.set_title('{:.0f}'.format(maxDelPwrPlot * 100) + "% ray power deposition\nfor " + '{:.2f}'.format(rf_power*1e-6)+ " MW coupled")
    #ax.plot(xlim, ylim, 'r') # walls defined in CHEASE
    ax.plot(rbbbs, zbbbs, 'black') # countour of LCFS?
    cbar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=plt.cm.get_cmap('Spectral_r')), ax=ax, shrink=0.95, fraction=0.15)
    #ax.set_aspect('equal')
    
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(label="Normalized Ray Power", size=sz)
    plt.tight_layout()
    plt.savefig("raytrace_2D_power_deposition.png", dpi=300)
    #fig.tight_layout(rect=[0, 0, .9, 1.0])


def drawFluxSurfaces(ax):
    r = gfileDict["rgrid"]
    z = gfileDict["zgrid"]
    psirz = gfileDict["psirz"]
    psi_mag_axis = gfileDict["ssimag"]
    psi_boundary = gfileDict["ssibdry"]

    ## sprt(psi_n); 2D numpy array (129, 129)
    psirzNorm = np.sqrt((psirz - psi_mag_axis)/(psi_boundary-psi_mag_axis))
    
    geqdsk = omfit_eqdsk.OMFITgeqdsk(eqdskPath)
    
    # convert 2D coords to srqt(toroidal_flux_normed)
    new_coordinates=np.empty((len(psirzNorm),len(psirzNorm)))
    for row_idx in range(len(psirzNorm)):
        for col_idx in range(len(psirzNorm)):
                new_coordinates[row_idx, col_idx]=aurora.coords.rad_coord_transform(x=psirzNorm[row_idx, col_idx], name_in='rhop', name_out='rhon', geqdsk=geqdsk)
    
    rInterp = np.linspace(np.min(r), np.max(r), 200)
    zInterp = np.linspace(np.min(z), np.max(z), 200)

    #psirzNormInterp = interp2d(r,z, psirzNorm, kind = 'cubic')(rInterp, zInterp)
    #ax.contour(rInterp, zInterp, (psirzNormInterp), np.square(np.arange(0,1.1,.1)), colors= 'k')

    rho_tor_NormInterp = interp2d(r,z, new_coordinates, kind = 'cubic')(rInterp, zInterp)

    #levels=np.arange(0,1.0,.1)
    levels=np.asarray([0.2,0.4,0.6,0.8])
    CS = ax.contour(rInterp, zInterp, rho_tor_NormInterp, levels, colors= 'black', alpha=0.65)
    ax.clabel(CS, inline=True, fmt='{:.1f}'.format, fontsize=12, inline_spacing=20.0, colors='black')

def plotHarmonicContours(ax):
    rgrid = gfileDict["rgrid"]
    zgrid = gfileDict["zgrid"]
    B_zGrid = gfileDict["bzrz"]
    B_TGrid = gfileDict["btrz"]
    B_rGrid = gfileDict["brrz"]
    B_magGrid = np.sqrt(B_zGrid**2 + B_rGrid**2 + B_TGrid**2)
    freq = cqlrf_nc.variables["freqcy"][:]
    ns = (2*freq/(1.52e7*B_magGrid))
    
    CS = ax.contour(rgrid, zgrid, ns, levels = [2,3,4,5,6,7,8,9,10], linewidths = 1.5, colors = 'k', linestyles = 'dashed')
    ax.clabel(CS, rightside_up = False, fmt = '%1.1f',inline=1, fontsize=12, manual = False)
    
    ns = (2*freq/(1.52e7*B_magGrid))

# ID FROM GRANT WHAT SHOULD BE DELETED, VS WHAT'S WORTH FIXING
def plotAbsorption():  
    urfpwrl = cqlrf_nc.variables["urfpwrl"][:]
    sdpwr = cqlrf_nc.variables["sdpwr"][:]
    spsi = cqlrf_nc.variables["spsi"][:]
    delpwr = cqlrf_nc.variables["delpwr"][:]*1e-7*1e-6#convert to MW
   
    dvol = cql_nc.variables["dvol"]
    rya = cql_nc.variables["rya"]
    powrft = cql_nc.variables["powrft"][-1]
    ionDep = powrft*dvol/1e6

    electronPower = urfpwrl * delpwr
    rhos = rya
    rhoBinEdges = (rya[1:] + rya[:-1])/2
    frontEdge = rya[0] - (rya[1] - rya[0])/2
    backEdge = rya[-1] + (rya[-1] - rya[-2])/2

    rhoBinEdges = np.concatenate(([frontEdge],rhoBinEdges,[backEdge]))

    linDep = np.zeros(len(rhos))

    print(f"len rho bin edges: {len(rhoBinEdges)}, len rya: {len(rya)}")

    ePowerTotal = 0
    for i in range(len(delpwr)):
        ePowerInLCFS = electronPower[i][spsi[i] <= rhoBinEdges[-1]]
        spsiInLCFS = spsi[i][spsi[i] <= rhoBinEdges[-1]]
        ePowerTotal += np.sum(ePowerInLCFS)
        indices = np.digitize(spsiInLCFS, rhoBinEdges, right = False)
        np.add.at(linDep, indices-1, ePowerInLCFS)

    fig,ax = plt.subplots()
    ax.plot(rya, linDep, label  = "ion power dep", lw=1.5)
    ax.plot(rya, ionDep, label = "electron power dep", lw=1.5)

    print(f"ion dep: {np.sum(ionDep)}, electronDep: {np.sum(linDep)}")

    #ax.plot(rya, ionDep + linDep, label = "total power dep", lw=1.5)

    ax.set_ylabel("MW")
    ax.set_xlabel(r'$\rho_{pol}$', fontsize = 20)
    ax.legend(loc = 'best')
    fig.tight_layout()

def main():
    plotRays()
    #plotAbsorption()
    plt.show()

main()
