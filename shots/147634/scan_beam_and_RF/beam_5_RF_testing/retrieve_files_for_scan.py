import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

# Detailed scan 


# 0.0 MW RF power 
print('0.0 MW...')
print('getting cql3d.nc ...')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_test/cql3d.nc cql3d_rfpwr_0.nc')
print('getting cqlkrf.nc ...')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_test/cql3d_krf001.nc cql3d_krf_rfpwr_0.nc')
print('getting post script file...')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_test/cql3d.ps cql3d_rfpwr_0.ps')
# # 0.7 MW RF power 
# print('0.70 MW...')
# print('getting cql3d.nc ...')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7_test/cql3d.nc cql3d_rfpwr_0_7.nc')
# print('getting cqlkrf.nc ...')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7_test/cql3d_krf001.nc cql3d_krf_rfpwr_0_7.nc')
# print('getting post script file...')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7_test/cql3d.ps cql3d_rfpwr_0_7.ps')

