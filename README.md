# SEP-CVDL

## requirements.txt
```
pipreqs /Users/wery/Desktop/SEP-CVDL
```
# SEP-CVDL
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

## important git commands

```
git status

git add -A --dry-run
git add -A

git commit -m  'comment'
git push

git stash list
git stash clear
```
### Dataset

Original dataset web page:
http://www.whdeng.cn/raf/model1.html

Where I downloaded:
https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/code

0. happiness
1. surprise
2. sadness
3. anger
4. disgust
5. fear
6. Neutral (delete)

- archive/DATASET/test is from the RAF-DB dataset test folder
- archive/DATASET/train contains from: 
1. RAF-DB 
2. FER+ 
3. TFEID 
4. CK+

### Notes

The configuration parameters of the DLP-CNN and the hyper-parameters of the trianing process is caffe-expression: 
https://github.com/cmdrootaccess/caffe
