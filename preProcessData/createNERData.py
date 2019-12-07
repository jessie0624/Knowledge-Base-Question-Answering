import sys
import os
import pandas as pd  

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
fileNameList = ['train.txt','dev.txt','test.txt']

NERDir = 'NERData'

questionStr = "<question"
tripleStr = "<triple"
answerStr = "<answer"
startStr = "============="


for filename in fileNameList:
    
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
                    

                            



