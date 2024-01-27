#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=CVDL_JML_%j_%N_%Y-%m-%d_%H-%M.out
#SBATCH --error=CVDL_JML_%j_%N_%Y-%m-%d_%H-%M.err
#SBATCH --partition=Nvidia2060

echo "Hello World!"
sleep 30