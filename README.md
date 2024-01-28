# SEP-CVDL

### How to run the script to get the CSV file of classification scores?

```
./script_csv.sh
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