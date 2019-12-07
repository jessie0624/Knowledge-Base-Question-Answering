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
please 'cd preProcessData'
- 1.  run splitTest.py

    There are train and test dataset in NLPCC2017 task5. We can spilt test data by 1:1 to get test and dev data.
    
- 2.  run preCleanData.py 

    There are three functions in this script: getNERData, getDBData and getSimilarityData.
    You will get three folders named NERData, DBData, SIMData.
    
- 3.  run uploadDB.py

    (pls create a KB_QA database  in mysql).
    Running the script, it will upload data in DBData folder to KB_QA database.   

#### Step-2: train NER model.

- 1. Pls download Bert Pretraining model(pytorch version) adding them to subModel/BertPreTrainedModel.
     and pls make sure NERData folder existed under preProcessData/NERData, there should be three text files(train.txt, dev.txt, test.txt)
    
- 2. training NER model
      
      run NERMain.py --data_dir preProcessData/NERData --vocab_file BertPreTrainedModel/vocab.txt --model_config BertPreTrainedModel/conig.json --output_dir output_model --pre_train_model BertPreTrainedModel/pytorch_model.bin --max_seq_length 64 --do_train --train_batch_size  8 --eval_batch_szie 8 --gradient_accumulation_steps 16 --num_train_epochs 8


#### Step-3: train classification model.

- 1. Pls make sure  Bert Pretraining model(pytorch version) under subModel/BertPreTrainedModel existed.
- 2. training classification model

    run SIMMain.py --data_dir preProcessData/SIMData --vocab_file BertPreTrainedModel/vocab.txt --model_config BertPreTrainedModel/config.json --output_dir output_model --pre_train_model BertPreTrainedModel/pytorch_model.bin --max_seq_length 64 --do_train --train_epoch_size 8 --eval_batch_size 8 --gradient_accumulation_steps 16 -num_train_epochs 8
    
#### Step-4: Run model test.
    python RunTask.py




