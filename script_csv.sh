#!/bin/bash

# Please change this to your own path
IMAGE_PATH="dataset/validation_set" 
# IMAGE_PATH='archive/RAF-DB/test'

# Activate your Python environment if needed # /path/to/your/virtualenv/bin/activate
# chmod +x script_csv.sh  
# source /Users/wery/venv/bin/activate
# ./script_csv.sh  

# Please change this to your own path
/opt/homebrew/bin/python3 /Users/wery/Desktop/SEP-CVDL/script_csv.py $IMAGE_PATH
