# SEP-CVDL

## Where is the overview of the pipeline (drawio)?

https://app.diagrams.net/#G15989rnA3gZtCfwwUN6p3uUDmJ0T69YZN


### Archieve folder contains 2 separate datasets (Training & Testing) contains all datasets together


### Dataset with clear 6 emotion-folders
- training_sets:
1. RAF-DB
2. FER
3. AffectNet
4. CK+
5. TFEID
- testing_sets:
1. RAF-DB
2. FER
- validation_set

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
5. AffectNet 
(https://www.kaggle.com/datasets/ibtsam/affectnet-yolo?resource=download)

### Notes

https://github.com/Tandon-A/emotic

The configuration parameters of the DLP-CNN and the hyper-parameters of the trianing process is caffe-expression: 
https://github.com/cmdrootaccess/caffe

### How to run the script to get the CSV file of classification scores?

```
./script_csv.sh
```