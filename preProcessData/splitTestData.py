import pandas as pd 
import os

dataDir = 'NLPCC2016KBQA'
fileNameList =  ['nlpcc-iccpol-2016.kbqa.testing-data','nlpcc-iccpol-2016.kbqa.training-data']

for filename in fileNameList:
    fileRoot = os.path.join(dataDir, filename)
    file = []
    with open(fileRoot, 'r', encoding='utf-8') as rf:
        for line in rf:
            line = line.strip()
            if line.strip():
                file.append(line)
    
    if 'training' in filename:
        print('training data :', len(file))
        with open(os.path.join(dataDir, 'train.txt'), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(file))
    elif 'testing' in filename:
        assert len(file) % 4 == 0 ## 一个样本由4行构成
        testItemCount = len(file)/4 
        testCount = int(testItemCount/2)
        testLines = int(testCount * 4)
        print('original test data: ', len(file))
        print('after split, test data:', testCount,', dev data:', testCount)
        with open(os.path.join(dataDir, 'test.txt'), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(file[:testLines]))
        with open(os.path.join(dataDir, 'dev.txt'), 'w', encoding='utf-8') as wf:
            wf.write('\n'.join(file[testLines:]))

print('Data split done!') 
  

