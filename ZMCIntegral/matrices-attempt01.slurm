#!/bin/bash

#SBATCH -J zmc-ds-func-01 # name of the job

### set output files
#SBATCH -o zmc-matrices.%A.out #output file 
#SBATCH -e zmc-matrices.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./matrices-attempt-01.py
