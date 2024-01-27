# SEP-CVDL

## requirements.txt
```bash
pipreqs /Users/wery/Desktop/SEP-CVDL
```
# SEP-CVDL
## notes on Slurm

1. get your CIP account information 
2. activate romote access
3. git bash (ssh included) on windows / go to Terminal on MacBook
4. Login to remote host: (e.g., name lachen) command:
```bash
ssh <cip-kennung>@remote.cip.ifi.lmu.de
```
 + password
ok everthing

5. command:
```bash
sinfo | grep Nvidia2060
```

6. go to Nvidia2060 host, get there by picking one of the names listed after idle 
command: 
```bash
ssh <cip-kennung>@<server_name>
```
(e.g., ssh <cip-kennung>@chondrit, ssh <cip-kennung>@idle) asked to type yes + enter password

7. to get information of the server architechtur

Linux terminal command: lsb_release -a
get info of graphiccard command: nvidia-smi
list of python versions: 
kawka@chondrit:~ (0) [15:43:07] % python --version  
Python 2.7.18  
kawka@chondrit:~ (0) [15:45:06] % python3 --version  
Python 3.8.10  
information systemworkload command: htop 

exit: q

8. clone project to server

when you cloning this repository onto the server it will be saved under your <lrz_user_name>
```bash
git clone https://github.com/werywjw/SEP-CVDL.git
```

login with unsername of github and github access token:
To generate a personal access token on GitHub:

    Navigate to Settings: Log in to your GitHub account and go to "Settings" by clicking on your profile picture in the top right corner and selecting "Settings" from the dropdown menu.

    Access Developer Settings: In the left sidebar, click on "Developer settings".

    Generate New Token: In the developer settings menu, click on "Personal access tokens", then click on the "Generate new token" button.

    Configure Token: You'll be prompted to enter your password and then to configure the token. You can give your token a descriptive name and select the scopes (permissions) that it should have. Make sure to only grant the permissions necessary for your intended use.

    Generate Token: After configuring the token, click the "Generate token" button.

    Copy Token: Once the token is generated, make sure to copy it to a secure location. GitHub will not display the token again.

    Use Token: You can now use this personal access token as a replacement for your password when accessing GitHub programmatically or via the API. Treat it with the same level of security as you would your password.

check branch:
```bash
git status
```

swith to your branch, to the correct directory: 
```bash
git checkout <branch_name>
ls
cd <path to directory>
```

9. run code

```bash
sbatch <slurm_script>
```

get joblist
```bash
squeue | grep <lrz_user_name>
```

check file results on server
```bash
ls
```

view content of files
```bash
cat <file_name>
```

10. push and pull commands for interactions github and server
```bash
git add -A
git commit -m '<message>'
git push
```
to get new changes from github to server
```bash
git pull
```
11. store your username and password from github on personal lrz-server

```bash
git config credential.helper store
```

12. exit lrz-server to jumphost
```bash
exit
```

11. exit jumphost
```bash
exit
```

## important git commands

```bash
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
