!/bin/sh
#SBATCH -N 1
#SBATCH -n 32
#SBATCH -t 00:30:00
#SBATCH -p sched_mit_psfc
#SBATCH -J GENRAY

mpirun /home/frank/CODES/genray_custom/xgenray_mpi_intel.engaging