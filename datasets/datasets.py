from lib2to3.pgen2 import token
import re
import torch
from xml.etree.ElementTree import tostring
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, SubsetRandomSampler

SEP_TOKEN = '[SEP]'
CLS_TOKEN = '[CLS]'
MAX_SEQ_LENGTH = 512
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SequenceDataset(Dataset):
    def __init__(self,dataset_file_path,tokenizer,regex_transformations={}):
        # 读取json文件并分配给标题变量
        df = pd.read_json(dataset_file_path,lines=True)
        df = df.drop(['article_link'],axis=1) # 这里是删掉某一行，大概是不需要标签?
        self.headlines = df.values


        # 正则表达式转换用于数据清理 比如转换'\n' -> ' ','wasn't -> was not
        self.regex_transformations = regex_transformations
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.headlines)

    def __getitem__(self, index):
        is_sarcastic,headline = self.headlines[index]
        for regex,value_to_replace_with in self.regex_transformations.items():
            headline = re.sub(regex,value_to_replace_with,headline)

        # 这里使用特殊的bert Tokenizer 来将字符串转换为标记，他能使子图处理词汇表外的单词
        # 比如 headline = Here is the sentence I want embeddings for.
        #     tokens = [here, is, the, sentence, i, want, em, ##bed, ##ding, ##s, for, .]

        tokens = self.tokenizer.tokenize(headline)

        # 在每个句子前面添加[CLS]，在末尾添加[SEP],用于分类
        tokens = [CLS_TOKEN]+tokens+[SEP_TOKEN]

        # 从词汇表中将标记转化为相应的ids
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        # 分类时将单个序列端的ID 设为0
        segment_ids = [0] * len(input_ids)

        # 输入mask ，其中有效标记的mask = 1，没有标记的mask = 0
        input_mask = [1] * len(input_ids)

        #padding_length 是用来让input_ids、input_mask、segment_ids达到 max_seq_length
        padding_length = MAX_SEQ_LENGTH - len(input_ids)

        input_ids = input_ids + [0] * padding_length

        input_mask = input_mask + [0] * padding_length      # 这个是词向量mask
        segment_ids = segment_ids + [0] * padding_length    # 这个是分段判断

        assert len(input_ids) == MAX_SEQ_LENGTH
        assert len(input_mask) == MAX_SEQ_LENGTH
        assert len(segment_ids) == MAX_SEQ_LENGTH

        return torch.tensor(input_ids,dtype=torch.long,device=DEVICE), \
               torch.tensor(segment_ids,dtype=torch.long,device=DEVICE), \
               torch.tensor(input_mask,device=DEVICE), \
               torch.tensor(is_sarcastic,dtype=torch.long,device=DEVICE)
               
