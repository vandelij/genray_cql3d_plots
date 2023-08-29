import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]

os.system(f'scp {username}@{host}:{rwdir}/cql3d.nc ../shots/{shotNum}/')
os.system(f'scp {username}@{host}:{rwdir}/cql3d_krf001.nc ../shots/{shotNum}/')
os.system(f'scp {username}@{host}:{rwdir}/cql3d.ps ../shots/{shotNum}/')
os.system(f'scp {username}@{host}:{rwdir}/cqlinput ../shots/{shotNum}/cqlinput_received')


