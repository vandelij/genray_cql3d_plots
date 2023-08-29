import os

host = 'eofe7.mit.edu'
username = 'vandelij'
rwdir = open("../remoteDirectory.txt", "r").readlines()[0].strip()

os.system(f'ssh {username}@{host} "cd {rwdir}; sbatch genr.pbs"')


