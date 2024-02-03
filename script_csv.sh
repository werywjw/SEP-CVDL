#!/bin/bash

# Please change this to your own path
IMAGE_PATH="dataset/vali"
# IMAGE_PATH='archive/RAF-DB/test'

# Activate your Python environment if needed # /path/to/your/virtualenv/bin/activate
# chmod +x script_csv.sh  
# source /Users/wery/venv/bin/activate
# ./script_csv.sh  

# Run the Python script with the image path
/opt/homebrew/bin/python3 /Users/wery/Desktop/SEP-CVDL/script_csv.py $IMAGE_PATH
