#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --partition=Nvidia2060

# run jupyternotebook headless
# jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False validation_GCAM.ipynb
echo "Hello World!"
echo "$(python3 --version)"
sleep 30