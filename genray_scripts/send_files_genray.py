import numpy as np
import subprocess, shlex, os

import os, sys
#these shenanigans relate to vscode not having the working directory as the directory of the file it runs
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import shotToEqdsk

# This script send input files to server  
#    set the name of eqdsk file in genray.in 
#    update genray.in file
#    send genray.in and gfile to host
host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../remoteDirectory.txt", "r").readlines()[0].strip()
shotNum = rwdir.split('/')[-1].split('_')[1]  # whatever is in the remoteDirectory file, extract first the end of a long directory string, then keep whatever comes after
# the first _ 
print('NEw STUFF')

os.system(f'scp ../shots/{shotNum}/genray.in {username}@{host}:{rwdir}')

os.system(f'scp ../shots/{shotNum}/genray.dat {username}@{host}:{rwdir}')

eqdskName = ''
eqdskName = shotToEqdsk.getEqdskName(shotNum)

os.system(f'cp ../shots/{shotNum}/{eqdskName} eqdsk')
os.system(f'scp eqdsk {username}@{host}:{rwdir}')
os.system('rm eqdsk')

os.system(f'ssh {username}@{host} cp ~/codes/genr.pbs {rwdir}')