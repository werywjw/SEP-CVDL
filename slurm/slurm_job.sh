#!/bin/bash
#SBATCH --job-name=CVDL_JML
#SBATCH --output=%x_%j_%N.out
#SBATCH --error=%x_%j_%N.err
#SBATCH --partition=Nvidia2060
#TODO: check how to get files created by script

echo "Starting execution of CVDL_JML"
echo "Shell: $SHELL"
echo "Python version: $(python3 --version)"

branch_name="<branch_name>" # TODO: change to your branch name
echo "Branch name: $branch_name"

jupyternotebook_to_execute="validation_GCAM.ipynb"
echo "Jupyter notebook to execute: $jupyternotebook_to_execute"

# create python environment
# install miniconda
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
~/miniconda3/bin/conda init zsh
source ~/miniconda3/bin/activate
echo "Conda version: $(conda --version)"
conda config --append channels conda-forge

# get repo for <branch_name> for everyone to use, TODO: change to your branch_name
rm -rf SEP-CVDL
git clone https://<github-token>/werywjw/SEP-CVDL.git # TODO: change to your github-token
cd SEP-CVDL/
git checkout $<branch_name>
GIT_CHECKOUT_PID=$!

# step 2: Wait for the Git checkout to finish
wait $GIT_CHECKOUT_PID

ls
# create conda environment
# delete existing
conda env remove --name myenv
# create a new environment from the list of installed packages
conda create --name myenv --file slurm/installed_packages.txt --yes
# activate the environment
conda activate myenv
# check the environment
conda info --env | grep "active environment"
# list the packages in the environment
conda list

# run jupyternotebook headless
jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False $jupyternotebook_to_execute

echo "Ending execution of CVDL_JML"
