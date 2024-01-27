#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --partition=Nvidia2060
#TODO: check how to get files created by script

echo "Starting executio of CVDL_JML"
echo "Shell: $SHELL"
echo "Python version: $(python3 --version)"

# create python environment
# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh
source ~/miniconda3/bin/activate
echo "$(conda --version)"

# create conda environment
conda create --name myenv
conda create --name new_env --file installed_packages.txt
conda info --env | grep "active environment"

# run jupyternotebook headless
# jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False validation_GCAM.ipynb
sleep 30