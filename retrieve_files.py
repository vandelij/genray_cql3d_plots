import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

# os.system(f'scp {username}@{host}:{rwdir}/cql3d.nc shots/{shotNum}/cql3d.nc')
# os.system(f'scp {username}@{host}:{rwdir}/cql3d_krf001.nc shots/{shotNum}/cql3d_krf001.nc')
os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/eqdsk shots/{shotNum}/eqdsk')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/cql3d.nc shots/{shotNum}/cql3d.nc')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/cql3d_krf001.nc shots/{shotNum}/cql3d_krf001.nc')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/cql3d.ps shots/{shotNum}/cql3d.ps')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/cqlinput shots/{shotNum}/cqlinput_received')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/genray.nc shots/{shotNum}/')
# os.system(f'scp {username}@{host}:{rwdir}/case_beampwr_10_general_electron_general_ion_colmodl0_both_rf_copy_to_fix_power_issue/genray.in shots/{shotNum}/genray_received.in')
npara = 2.97
den = 3e19
#os.system(f'scp {username}@{host}:{rwdir}/cql3d.nc shots/{shotNum}/scanResults/cql3d_1.2Tescal.nc')
#os.system(f'scp {username}@{host}:{rwdir}/cql3d_krf001.nc shots/{shotNum}/scanResults/cql3d_krf001__1.2Tescal.nc')

# os.system(f'scp {username}@{host}:{rwdir}/cql3d.ps shots/{shotNum}/')
# os.system(f'scp {username}@{host}:{rwdir}/cqlinput shots/{shotNum}/cqlinput_received')

# os.system(f'scp {username}@{host}:{rwdir}/genray.nc shots/{shotNum}/')
# os.system(f'scp {username}@{host}:{rwdir}/genray.in shots/{shotNum}/genray_received.in')

