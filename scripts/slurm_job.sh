#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --partition=Nvidia2060
#TODO: check how to get files created by script

echo "Starting executio of CVDL_JML"
echo "$(python3 --version)"
echo "$(conda --version)"
# clone repository

# create python environment

# run jupyternotebook headless
# jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False validation_GCAM.ipynb
sleep 30