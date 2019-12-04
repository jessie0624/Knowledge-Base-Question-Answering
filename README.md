# KB_QA (Knowledge Base Question Answering)

## Introduction
This is knowledge base QA task with the data from http://tcci.ccf.org.cn/conference/2017/taskdata.php (task 5: Open Domain Question Answering).
Chinese version introduction: https://blog.csdn.net/m0_37531129/article/details/103321814


## Repeat Task

### 1. prepare the environment 
1. My development environment is ubuntu 14.04 with pytorch 1.2.0.
2. Please make sure 'mysql' was installed in your desktop/PC. There are many guidances for mysql installing in Ubuntu.

### 2. repeat the model
You can clone or download this 'KB_QA' repository.

#### Step-1: prepare data. (no parameter, just run by script in termial by cmd like 'python xxx.py')
- run splitTest.py

    There are train and test dataset in NLPCC2017 task5. We can spilt test data by 1:1 to get test and dev data.
    
- run preCleanData.py 

    There are three functions in this script: getNERData, getDBData and getSimilarityData.
    You will get three folders named NERData, DBData, SIMData.
    
- run uploadDB.py

    (pls create a KB_QA database  in mysql).
    Running the script, it will upload data in DBData folder to KB_QA database.   

#### Step-2: train NER model.

#### Step-3: train classification model.





