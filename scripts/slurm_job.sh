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

# get repo
git clone https://github.com/werywjw/SEP-CVDL.git

# create conda environment
# delete existing
conda env remove --name myenv
# create a new environment from the list of installed packages
cd SEP-CVDL/
conda create --name myenv --file installed_packages.txt --yes
# activate the environment
conda activate myenv
# install packages
# conda install pip --yes
# install packages from requirements.txt
# pip install -r installed_packages.txt --yes
# check the environment
# conda info --env | grep "active environment"
# list the packages in the environment
conda list

# run jupyternotebook headless
jupyter nbconvert --to html --execute --ExecutePreprocessor.enabled=False validation_GCAM.ipynb
sleep 30