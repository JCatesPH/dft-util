#!/bin/bash


#SBATCH -J zmc-tf # name of the job

### set output files
#SBATCH -o ./tf1.%A.out #output file 
#SBATCH -e ./tf1.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./zmc-tf-test1.py
