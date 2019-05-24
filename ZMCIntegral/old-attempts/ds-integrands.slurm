#!/bin/bash


#SBATCH -J zmc-ds-int # name of the job

### set output files
#SBATCH -o ./ds-output/ds-constqx.%A.out #output file 
#SBATCH -e ./ds-output/ds-constqx.%A.err #output error file

### Parameters necessary for GPU computation
#SBATCH --qos gpu
#SBATCH -p gpu
#SBATCH --gres gpu:v100:1

# run application
srun python ./ds-integrands-03.py
