# SEP-CVDL

## evaluation plan TODO
- [X] train on the RAF-DB training set firstly

- [X] see CSV results on RAF-DB test & validation given 

- [X] rerun models after CSV classification score is fine

- [X] rerun aggregated training data (FER&RAF-DB)

- [ ] see if given image with label in names.jpeg as validation can directly read


## requirements.txt
```
pipreqs /Users/wery/Desktop/SEP-CVDL
```

## How to get the classification scores in a CSV file?

Run the following command:
```
python3 script_csv.py
```
Note that please change the filepath to the image folder that you would like to try 😄

## notes on Slurm

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
