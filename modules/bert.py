import torch.nn as nn
import torch
from pytorch_transformers import BertModel

class BertClassifier(nn.Module):

    def __init__(self,config) -> None:
        super(BertClassifier,self).__init__()
        # num_labels在这里等于2 ，本项目处理2元分类问题，若是
        self.num_labels = config.num_labels
        # pre-trained BERT model 预训练模型
        self.bert = BertModel(config)
        # Dropout 正则化 ，避免溢出
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 微调的时候加一个分类器
        self.classifier = nn.Linear(config.hidden_size,config.num_labels)
        # 权重初始化
        torch.nn.init.xavier_normal_(self.classifier.weight)

    def forward(self,input_ids,token_type_ids=None,attention_mask=None,position_ids=None,head_mask=None):

        # 预训练bert
        outputs = self.bert(input_ids,position_ids,token_type_ids,attention_mask,head_mask)
        # 输出最后一层
        pooled_output = outputs[-1]
        # 进行一次dropout正则优化
        pooled_output = self.dropout(pooled_output)

        return self.classifier(pooled_output)