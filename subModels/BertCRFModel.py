from typing import List, Optional
from transformers import BertForTokenClassification, BertTokenizer, BertConfig
from CRFModel import CRF
import torch
import torch.nn as nn
import os


MODEL_NAME = "pytorch_model.bin"
CONFIG_NAME = "bert-base-chinese-config.json"
VOB_NAME = "bert-base-chinese-vocab.txt"


class BertCRF(nn.Module):
    def __init__(self, config_name:str,
                model_name:str=None,
                num_tags: int=2,
                batch_first:bool=True) -> None:
        self.batch_first = batch_first
        if not os.path.exists(config_name):
            raise ValueError('{} config file not found'.format(config_name))
        else:
            self.config_name = config_name
        
        if model_name is not None:
            if not os.path.exists(model_name):
                raise ValueError(' {} model file not found'.format(model_name))
            else:
                self.model_name = model_name
        else:
            self.model_name = None
        
        if num_tags <= 0:
            raise ValueError(f'invalid number of tags:{num_tags}')

        super().__init__()
        #bert config文件
        self.bert_config = BertConfig.from_pretrained(self.config_name)
        self.bert_config.num_tags = num_tags
        self.model_kwargs = {'config': self.bert_config}

        if self.model_name is not None:
            self.bertModel = BertForTokenClassification.from_pretrained(self.model_name, **self.model_kwargs)
        else:
            self.bertModel = BertForTokenClassification(self.bert_config)
        
        self.crfModel = CRF(num_tags=num_tags, batch_first=batch_first)

    def forward(self, input_ids:torch.Tensor,
                tags:torch.Tensor=None,
                attention_mask:Optional[torch.ByteTensor] = None,
                token_type_ids=torch.Tensor,
                decode:bool =True, ## 是否预测编码
                reduction: str='mean') -> List:
        emissions = self.bertModel(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids)[0]
        ## bert 的输出是由: last_hidden (batch_size, seq_len, hidden_size) --> linear --> (batch_size, seq_len, config.num_labels)
        ## 所以emissions: 是 batch_size, seq_len, num_labels.

        # 这里在seq_len的维度上 去头[CLS], 去尾 有2种情况 <pad>, <SEP>
        new_emissions = emissions[:, 1:-1]
        new_mask = attention_mask[:, 2:].bool

        #如果tags 为None, 表示一个预测的过程，不能求得 loss, loss为Ｎｏｎｅ
        if tags is None:
            loss = None
            pass
        else:
            new_tags = tags[:, 1:-1]
            loss = self.crfModel(emissions=new_emissions, tags=new_tags,
                                mask=new_mask, reduction=reduction)
        if decode:
            tag_list = self.crfModel.decode(emissions=new_emissions, mask=new_mask)
            return [loss, tag_list]
        return [loss]

