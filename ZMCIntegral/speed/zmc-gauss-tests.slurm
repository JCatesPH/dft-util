#!/bin/bash

#SBATCH -J zmc-speed-gauss # name of the job

### set output files
#SBATCH -o zmcgauss.%A.out #output file 
#SBATCH -e zmcgauss.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./ZMCGaussianSpeed.py
