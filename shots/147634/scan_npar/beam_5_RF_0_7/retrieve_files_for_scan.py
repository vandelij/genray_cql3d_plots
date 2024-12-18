import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

print(f'Looking in shotNum {shotNum}')

# Detailed scan 

npars = [10]
# for npar in npars:
#     print(f'npar = {npar}..')
#     os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}_yuri_edits/cql3d.nc cql3d_npar_{npar}_yuri_edits.nc')
#     os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}_yuri_edits/cql3d_krf001.nc cql3d_krf_npar_{npar}_yuri_edits.nc')
#     os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}_yuri_edits/cql3d.ps cql3d_rfpwr_npar_{npar}_yuri_edits.ps')

# for npar in npars:
#     os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}/cqlinput cqlinput_npar_{npar}')
#     os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}/genray.in genray_npar_{npar}.in')
for npar in npars:
    print(f'npar = {npar}..')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}/cql3d.nc cql3d_npar_{npar}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}/cql3d_krf001.nc cql3d_krf_npar_{npar}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix_npar_beam_5_rf_0_7/npar_{npar}/cql3d.ps cql3d_rfpwr_npar_{npar}.ps')
