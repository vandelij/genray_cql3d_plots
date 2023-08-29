import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../../../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

# 0 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_0_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_0.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_0_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_0.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_0_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_0.ps')

# 2.5 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_2_5_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_2_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_2_5_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_2_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_2_5_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_2_5.ps')

# 5 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_5_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_5_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_5_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_5.ps')

# 7_5 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_7_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_7_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_7_5.ps')

# 7_5 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_7_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_7_5.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_7_5_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_7_5.ps')

# 10 MW beam power 
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_max_electron_general_ion_colmodl0/cql3d.nc cql3d_bmpwr_10.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_max_electron_general_ion_colmodl0/cql3d_krf001.nc cql3d_krf_bmpwr_10.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_max_electron_general_ion_colmodl0/cql3d.ps cql3d_bmpwr_10.ps')