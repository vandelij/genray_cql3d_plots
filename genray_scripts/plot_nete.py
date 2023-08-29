import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import getInputFileDictionary
inputFileDict = getInputFileDictionary.getInputFileDictionary('genray',pathprefix=f'{parentdir}/')

import numpy as np
import matplotlib.pyplot as plt

plt.rc('xtick', labelsize = 14)
plt.rc('ytick', labelsize = 14)
plt.rc('axes', labelsize = 17)
plt.rc('figure', titlesize = 16)


temtab = inputFileDict['temtab']['prof']
dentab = inputFileDict['dentab']['prof']

T_e = np.zeros(int(len(temtab)/3))
T_i = np.copy(T_e)
n_e = np.copy(T_e)

for i in range(0,len(T_e)):
    T_e[i] = temtab[3*i]
    T_i[i] = temtab[3*i+1]
    n_e[i] = dentab[3*i]

T_e = T_e*inputFileDict['plasma']['temp_scale(1)']
T_i = T_i*inputFileDict['plasma']['temp_scale(2)']

rhosHelper = np.arange(1,len(T_e)+1,1)
rhos = (rhosHelper-1)/(len(T_e)-1)

fig, ax = plt.subplots(figsize = (8*.8,6*.8))

ln1 = ax.plot(rhos, T_e, label = r'$T_e$', linewidth = 3, color = 'g')
ln2 = ax.plot(rhos, T_i, label = r'$T_i$', linewidth = 3, color = 'b')

ax.set_ylabel('temperature (keV)')
ax.set_xlabel(r'$\rho_{pol}$', fontsize = 20)
ax.set_xlim([0,1])


ax1 = ax.twinx()
ln3 = ax1.plot(rhos, np.array(n_e), label = r'$n_e$', linestyle = 'dashed', color = 'k', linewidth = 3)
ax1.set_ylabel(r'density (1/m$^{-3}$)')
ax1.set_yticks(np.array([0,1,2,3,4,5,6])*1e19)

lns = ln1+ln2+ln3
ax.legend(lns, [l.get_label() for l in lns], loc = 'best', fontsize = 20)

dischargeNumber = parentdir.split('/')[-1]
fig.suptitle(f"Discharge {dischargeNumber} Genray Profiles", fontsize = 20)
fig.tight_layout(rect=[0, 0.0, 1, .98])

plt.show()

