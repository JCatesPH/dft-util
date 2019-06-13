#!/bin/bash


#SBATCH -J test  # name of the job

### set output files
#SBATCH -o ./test.%A.out #output file 
#SBATCH -e ./test.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./cupy-test.py
