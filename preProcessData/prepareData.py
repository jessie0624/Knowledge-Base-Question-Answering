import sys
import os
import random
import pandas as pd 
import re

## 生成命名实体识别数据
## 生成知识库资料,用于存入database.
## 生成正负训练集,用于相似度训练.

"""
根据原始问题生成NER数据, 如下: 
你 0
知 0
道 0
计 B-LOC
算 I-LOC
机 I-LOC
应 I-LOC
用 I-LOC
基 I-LOC
础 I-LOC
这 0
本 0
书 0
的 0
作 0
者 0
是 0
谁 0
吗 0
？ 0
"""
dataDir = 'NLPCC2016KBQA'
NERDir = 'NERData'
SMIDir ='SMIData'
DBDir = 'DBData'

questionStr = "<question"
tripleStr = "<triple"
answerStr = "<answer"
startStr = "============="

txtFileList = ['train.txt','dev.txt','test.txt']
csvFileList = ['train.csv', 'test.csv', 'dev.csv']

def createNERData():
    '''
    基于原始数据,生成NRE数据,结果存放在NERData文件夹下
    '''
    for filename in txtFileList:
        quesTripAnsList = []
        queSeqList = []
        tagSeqList = []

        fileRoot = os.path.join(dataDir, filename)
        assert os.path.exists(fileRoot)
        with open(fileRoot, 'r', encoding='utf-8') as rf:
            queStr = ''
            triStr = ''
            ansStr = ''

            for line in rf:
                if line.startswith(questionStr):
                    queStr = line.strip()
                if line.startswith(tripleStr):
                    triStr = line.strip()
                if line.startswith(answerStr):
                    ansStr = line.strip()

                if line.startswith(startStr):
                    entities =  triStr.split('|||')[0].split('>')[1].strip()
                    queStr = queStr.split('>')[1].replace(" ", "").strip()
                    if entities in queStr:
                        # queList = list(queStr)
                        queSeqList.extend(list(queStr) + [" "]) 
                        tagList = ["0" for i in range(len(list(queStr)))]
                        tagStartIdx = queStr.find(entities)
                        # tagList[tagStartIdx] = 'B-LOC'
                        # tagList[tagStartIdx + 1: tagStartIdx + len(entities)] = ['I-LOC' for i in range(len())
                        for i in range(tagStartIdx, tagStartIdx + len(entities)):
                            if tagStartIdx == i:
                                tagList[i] = 'B-LOC'
                            else:
                                tagList[i] = 'I-LOC'
                        tagSeqList.extend(tagList + [' '])
                    
                    quesTripAnsList.append([queStr, triStr, ansStr])
        print(filename)
        print('\t'.join(tagSeqList[0:50]))
        print('\t'.join(queSeqList[0:50]))
        seqRet = [str(que) + ' ' + tag for que, tag in zip(queSeqList, tagSeqList)]
        if not os.path.exists(NERDir):
            os.mkdir(NERDir)
        with open(os.path.join(NERDir, filename), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(seqRet))
        
        df = pd.DataFrame(quesTripAnsList, columns= ['queStr', 'triStr', 'ansStr'])
        
        csvName = filename.split('.')[0] + '.csv'
        df.to_csv(os.path.join(NERDir, csvName), encoding='utf-8', index=False)

def createDBData():
    '''
    基于原始数据,生成知识库,用于存入database,结果存放在DBData文件夹下
    '''
    tripList = []
    for data_type in ["training", "testing"]:
        filename = os.path.join(dataDir, "NLPCC2016KBQA/nlpcc-iccpol-2016.kbqa." + data_type + "-data")
        with open(filename, 'r',encoding='utf-8') as f:
            queStr = ''
            triStr = ''
            for line in f:
                if questionStr in line:
                    queStr = line.strip()
                if tripleStr in line:
                    triStr = line.strip()
                if startStr in line:  #new question answer triple
                    entities = triStr.split("|||")[0].split(">")[1].strip()
                    queStr = queStr.split(">")[1].replace(" ","").strip()
                    if ''.join(entities.split(' ')) in queStr:
                        clean_triple = triStr.split(">")[1].replace('\t','').replace(" ","").strip().split("|||")
                        tripList.append(clean_triple)
                    else:
                        print(entities)
                        print(queStr)
                        print('------------------------')

    df = pd.DataFrame(tripList, columns=["entity", "attribute", "answer"])
    print(df)
    print(df.info())
    if not os.path.exists(DBDir):
        os.makedirs(DBDir)
    df.to_csv("DB_Data/clean_triple.csv", encoding='utf-8', index=False)

def createSimilarityData():
    """
    通过NREData 数据,构建匹配句子相似度的样本集,生成结果存放在SMIData 文件夹下
    """
    pattern = re.compile('^-+') #以-开头

    for filename in csvFileList:
        fileRoot = os.path.join(NERDir, filename)
        assert os.path.exists(fileRoot)
        
        attrClassifySample = []
        df = pd.read_csv(fileRoot, encoding='utf-8')
        df['attribute'] = df['triStr'].apply(lambda x: x.split("|||")[1].strip())
        attrList = list(set(df['attribute'].tolist())) ## 转成list  去重
        attrList = [attr.strip().replace(' ', '') for attr in attrList] ##去空格
        attrList = [re.sub(pattern, "", attr) for attr in attrList]#去除以-开头的行
        attrList = list(set(df['attribute'].tolist())) ## 转成list  去重

        for row in df.index:
            question, pos_attr = df.loc[row][['queStr','attribute']]
            question = question.strip().replace(' ', '') #去除尾部空格
            question = re.sub(pattern, '', question)# 去除以--开头的
            pos_attr = pos_attr.strip().replace(' ', '')
            pos_attr = re.sub(pattern, '', pos_attr)

            negAttrList = []
            while True:
                negAttrList = random.sample(attrList, 5)
                if pos_attr not in negAttrList:
                    break
            attrClassifySample.append([question, pos_attr, '1'])
            negAttrSample = [[question, neg_attr, '0'] for neg_attr in negAttrList]
            attrClassifySample.extend(negAttrSample)
        seqRet = [str(lineNum) + '\t' + '\t'.join(line) for (lineNum, line) in enumerate(attrClassifySample)]

        if not os.path.exists(SMIDir):
            os.makedirs(SMIDir)
        
        smiFileName = filename.split('.')[0] + '.txt'
        with open(os.path.join(SMIDir, smiFileName), 'w', encoding='utf-8') as f:
            f.write('\n'.join(seqRet))
    
    ## check seq len
    for filename in txtFileList:
        fileRoot = os.path.join(SMIDir,filename)

        max_len = 0
        print("****** {} *******".format(filename))
        with open(fileRoot,'r',encoding='utf-8') as f:
            for line in f:
                lines = line.split('\t')
                question = list(lines[1])
                attribute = list(lines[2])
                totalLen = len(question) + len(attribute)
                if totalLen > max_len:
                    max_len = totalLen
        print("max_len",max_len)


if __name__ == "__main__":
    createNERData()
    createSimilarityData()
    createDBData()