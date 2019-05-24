#!/bin/bash

#SBATCH -J zmc-complex # name of the job

### set output files
#SBATCH -o zmccomplex.%A.out #output file 
#SBATCH -e zmccomplex.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./complex-test.py
