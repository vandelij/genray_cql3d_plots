import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

# # 0.1 MW RF power 
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_1/cql3d.nc cql3d_rfpwr_0_1.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_1/cql3d_krf001.nc cql3d_krf_rfpwr_0_1.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_1/cql3d.ps cql3d_rfpwr_0_1.ps')

# # 0.5 MW RF power 
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_5/cql3d.nc cql3d_rfpwr_0_5.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_5/cql3d_krf001.nc cql3d_krf_rfpwr_0_5.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_5/cql3d.ps cql3d_rfpwr_0_5.ps')

# # 1 MW RF power 
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1/cql3d.nc cql3d_rfpwr_1.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1/cql3d_krf001.nc cql3d_krf_rfpwr_1.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1/cql3d.ps cql3d_rfpwr_1.ps')

# # 1.5 MW RF power 
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1_5/cql3d.nc cql3d_rfpwr_1_5.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1_5/cql3d_krf001.nc cql3d_krf_rfpwr_1_5.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_1_5/cql3d.ps cql3d_rfpwr_1_5.ps')

# # 2 MW RF power 
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_2/cql3d.nc cql3d_rfpwr_2.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_2/cql3d_krf001.nc cql3d_krf_rfpwr_2.nc')
# os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_2/cql3d.ps cql3d_rfpwr_2.ps')

# Detailed scan 

# 0.55 MW RF power 
print('0.55 MW..')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_55/cql3d.nc cql3d_rfpwr_0_55.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_55/cql3d_krf001.nc cql3d_krf_rfpwr_0_55.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_55/cql3d.ps cql3d_rfpwr_0_55.ps')

# 0.6 MW RF power 
print('0.6 MW..')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_6/cql3d.nc cql3d_rfpwr_0_6.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_6/cql3d_krf001.nc cql3d_krf_rfpwr_0_6.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_6/cql3d.ps cql3d_rfpwr_0_6.ps')

# 0.65 MW RF power 
print('0.65 MW..')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_65/cql3d.nc cql3d_rfpwr_0_65.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_65/cql3d_krf001.nc cql3d_krf_rfpwr_0_65.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_65/cql3d.ps cql3d_rfpwr_0_65.ps')

# 0.7 MW RF power 
print('0.7 MW..')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7/cql3d.nc cql3d_rfpwr_0_7.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7/cql3d_krf001.nc cql3d_krf_rfpwr_0_7.nc')
os.system(f'scp {username}@{host}:{rwdir}/scan_matrix/beam_5_RF_0_7/cql3d.ps cql3d_rfpwr_0_7.ps')