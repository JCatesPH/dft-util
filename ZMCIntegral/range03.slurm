#!/bin/bash


#SBATCH -J zmc-range-03 # name of the job

### set output files
#SBATCH -o ./range3.%A.out #output file 
#SBATCH -e ./range3.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./range-03.py
