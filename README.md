# SEP-CVDL 
## Facial Emotion Recognition & Interpretation

😄 happiness

😲 surprise

😭 sadness

😡 anger

🤢 disgust

😨 fear

## Where are our slides 🎥?

On [goolge docs](https://docs.google.com/presentation/d/14AazB5FY5jLyB5-9R6Ix7LoMX8yNR_illDAaYDCm8_s/edit#slide=id.g2b4d85efaed_0_31). 
The shared link is available [here](https://docs.google.com/presentation/d/14AazB5FY5jLyB5-9R6Ix7LoMX8yNR_illDAaYDCm8_s/edit?usp=sharing).

## How to generate a requirements.txt?
```
pipreqs /Users/wery/Desktop/SEP-CVDL
pipreqs /Path/to/the/repository/SEP-CVDL
```

## How to run the script to get the CSV file of classification scores?
Note that please change the **filepath** to the image folder that you would like to try. 😄

- Option 1️⃣: Run the following command:
```
python3 script_csv.py
```

- Option 2️⃣: Use shell:
```
./script_csv.sh
```

## How to run the script to the video?

## Notes on Slurm

1. get your CIP account information 
2. activate romote access
3. git bash (ssh included) on windows / go to Terminal on MacBook
4. Login to remote host: (e.g., name lachen) command:
```
ssh <cip-kennung>@remote.cip.ifi.lmu.de
```
 + password
ok everthing

5. command:
```
sinfo | grep Nvidia2060
```

6. go to Nvidia2060 host, get there by picking one of the names listed after idle 
command: 
```
ssh <cip-kennung>@<server_name>
```
(e.g., ssh <cip-kennung>@chondrit, ssh <cip-kennung>@idle) asked to type yes + enter password

7. to get information of the server architechtur, 
Linux terminal command: lsb_release -a
get info of graphiccard command: nvidia-smi
list of python versions: 
kawka@chondrit:~ (0) [15:43:07] % python --version  
Python 2.7.18  
kawka@chondrit:~ (0) [15:45:06] % python3 --version  
Python 3.8.10  
information systemworkload command: htop 
exit: q

8. exit the server by command:
```
exit
```

9. exit the remote host by command:
```
exit
```

## Approach with self defined hook functions: 
https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569

## approach with pytorch Grad-CAM: 
https://github.com/jacobgil/pytorch-grad-cam?tab=readme-ov-file

## Grad-CAM with Torchcam: 
https://github.com/frgfm/torch-cam/blob/main/README.md

```
colab-convert livecam.ipynb livecam.py -nc -rm -o
```