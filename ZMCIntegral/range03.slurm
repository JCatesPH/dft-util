#!/bin/bash


#SBATCH -J zmc-range-04 # name of the job

### set output files
#SBATCH -o ./range4.%A.out #output file 
#SBATCH -e ./range4.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./range-04.py
