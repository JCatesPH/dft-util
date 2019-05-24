#!/bin/bash


#SBATCH -J zmc-const-test # name of the job

### set output files
#SBATCH -o %A.out #output file 
#SBATCH -e %A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./const-param-test.py
