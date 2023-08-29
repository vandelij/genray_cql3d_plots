import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__);dname = os.path.dirname(abspath);os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__));parentdir = os.path.dirname(currentdir);sys.path.append(parentdir)

remoteDirectory = open(f"../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = remoteDirectory.split('/')[-1].split('_')[1]

import getInputFileDictionary
inputFileDict = getInputFileDictionary.getInputFileDictionary('cql3d',pathprefix=f'{parentdir}/')

import numpy as np
import matplotlib.pyplot as plt

tiscal = 1#inputFileDict['setup']['tiscal']
tescal = 1#inputFileDict['setup']['tescal']
enescal = 1#inputFileDict['setup']['enescal']

try:
    tiscal = inputFileDict['setup']['tiscal']
except:
    pass
try:
    tescal = inputFileDict['setup']['tescal']
    print(f"tiscal: {tescal}")
except:
    pass
try:
    enescal = inputFileDict['setup']['enescal']
except:
    pass

T_e = inputFileDict['setup']['tein']*tescal
T_i = inputFileDict['setup']['tiin']*tiscal
n_e = inputFileDict['setup']['enein(1,1)']*1e6*enescal

rhos = inputFileDict['setup']['ryain']

dischargeNumber = os.getcwd().split('/')[-1]
if len(dischargeNumber) != 6:
    dischargeNumber = os.getcwd().split('/')[-2]


#####Setup and do plotting#####
plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 17)
plt.rc('figure', titlesize = 16)
plt.rc('legend',fontsize=20)

fig, ax = plt.subplots(figsize = (8*.8,6*.8))

ln1 = ax.plot(rhos, T_e, label = r'$T_e$', linewidth = 3,color = 'g')
ln2 = ax.plot(rhos, T_i, label = r'$T_i$', linewidth = 3,color = 'b')
ax1 = ax.twinx()
ln3 = ax1.plot(rhos, np.array(n_e), label = r'$n_e$', linestyle = 'dashed', color = 'k', linewidth = 3)
lns = ln1+ln2+ln3
ax.legend(lns, [l.get_label() for l in lns], loc = 'upper right')

ax.set_ylabel('temperature (keV)')
ax.set_xlabel(r'$\rho_{pol}$', fontsize = 20)
ax.set_xlim([0,1])
ax.set_ylim([0,7.5])
ax1.set_ylim([0,5e19])
ax1.set_ylabel(r'density (1/m$^{-3}$)')
#ax1.set_yticks(np.array([.5,1,1.5,2,2.5,3,3.5])*1e19)

fig.suptitle(f"CQL3D {shotNum} Profiles")
fig.tight_layout(rect=[0, 0.0, 1, .98])
plt.savefig('ne_Te_prof.png')
print('figure saved')
plt.show()
################################
