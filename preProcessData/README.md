## preProcessData 
该目录主要完成数据切分,数据预处理得到NER数据,Smiliarity数据,和Database知识库.
- splitTestData : 数据切分, 将原始数据测试集切分成1:1的测试集和验证集
- prepareData: 
  - 1. 根据切分后的数据集,获取NER数据.
  - 2. 根据原始数据集,获取database 知识库.
  - 3. 根据NER数据集,获取smiliarity数据,用于分类预测训练.