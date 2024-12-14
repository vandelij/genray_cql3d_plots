import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

print(f'Looking in shotNum {shotNum}')

# Detailed scan 

bmpwrs = ['0', '7_5', '10'] # [1,2,3,4,5,6,7,8,9,10,11]

for bmpwr in bmpwrs:
    print(f'bmpwr = {bmpwr}..')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_{bmpwr}_RF_0_7/cql3d.nc cql3d_beampwr_{bmpwr}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_{bmpwr}_RF_0_7/cql3d_krf001.nc cql3d_krf_beampwr_{bmpwr}.nc')
    os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_{bmpwr}_RF_0_7/cql3d.ps cql3d_rfpwr_beampwr_{bmpwr}.ps')
