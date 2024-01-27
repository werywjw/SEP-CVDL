#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --partition=Nvidia2060
#TODO: check how to get files created by script

echo "Starting executio of CVDL_JML"
echo $SHELL
# clone repository

# create python environment
# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh

echo "$(python3 --version)"
echo "$(conda --version)"

# run jupyternotebook headless
# jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False validation_GCAM.ipynb
sleep 30