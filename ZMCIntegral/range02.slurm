#!/bin/bash


#SBATCH -J zmc-range-02 # name of the job

### set output files
#SBATCH -o ./range2.%A.out #output file 
#SBATCH -e ./range2.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./range-02.py
