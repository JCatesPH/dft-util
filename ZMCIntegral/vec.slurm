#!/bin/bash


#SBATCH -J zmc-vec # name of the job

### set output files
#SBATCH -o ./vectest.%A.out #output file 
#SBATCH -e ./vectest.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./vec-test.py
