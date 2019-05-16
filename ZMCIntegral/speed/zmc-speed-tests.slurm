#!/bin/bash

#SBATCH -J zmc-speed-sin # name of the job

### set output files
#SBATCH -o zmcspeed.%A.out #output file 
#SBATCH -e zmcspeed.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./ZMCSinIntegralSpeed.py
