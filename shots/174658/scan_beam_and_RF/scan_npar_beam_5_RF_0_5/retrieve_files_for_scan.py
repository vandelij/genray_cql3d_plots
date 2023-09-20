import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

nparv = [4, 5, 6, 7, 8 ,9, 10]

# Detailed scan 
for n in nparv:
    print(f'npar = {n}...')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/scan_npar_beam_5_RF_0_5/beam_5_RF_0_5_npar_{n}/cql3d.nc cql3d_npar_{n}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/scan_npar_beam_5_RF_0_5/beam_5_RF_0_5_npar_{n}/cql3d_krf001.nc cql3d_krf_npar_{n}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/scan_npar_beam_5_RF_0_5/beam_5_RF_0_5_npar_{n}/cql3d.ps cql3d_npar_{n}.ps')
