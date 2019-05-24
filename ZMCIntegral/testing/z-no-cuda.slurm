#!/bin/bash

#SBATCH -J zmc-complex-test2 # name of the job

### set output files
#SBATCH -o complex.%A.out #output file 
#SBATCH -e complex.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./complex-no-cuda.py
