#!/bin/bash

#SBATCH --job-name=inner-prod
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --mem=2GB 
#SBATCH --time=00:10:00 
#SBATCH --verbose

module purge 
module load  amber/openmpi/intel/20.06 

mpiexec ./a.out
