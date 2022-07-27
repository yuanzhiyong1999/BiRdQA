"""
@FileName ：unifiedqa-t5-large.py
@Author ：Zhiyong Yuan
@Date ：2022/4/22 14:17 
@Tools ：PyCharm
@Description：
"""
# coding: UTF-8
import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_type = 't5'  # 不同的类型 不同的数据处理方法
        self.model_name = 'unifiedqa-t5-large'
        self.dataset = dataset
        if dataset == 'BiRdQA':
            self.train_path = 'dataset/' + dataset + '/BiRdQA_en_train.csv'  # 训练集
            self.dev_path = 'dataset/' + dataset + '/BiRdQA_en_dev.csv'  # 验证集
            self.test_path = 'dataset/' + dataset + '/BiRdQA_en_test.csv'  # 测试集
        else:
            self.train_path = 'dataset/' + dataset + '/rs_train.jsonl'  # 训练集
            self.dev_path = 'dataset/' + dataset + '/rs_dev.jsonl'  # 验证集
            self.test_path = 'dataset/' + dataset + '/rs_test_hidden.jsonl'  # 测试集
        self.save_path = 'saved_dict/' + self.model_name + '.pt'  # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备

        self.num_choice = 5  # 选项数
        self.num_epochs = 5  # epoch数
        self.batch_size = 16  # mini-batch大小
        self.pad_size = 150  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-4  # 学习率
        self.pretrained_path = 'allenai/unifiedqa-t5-large'
        self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_path)
        self.weight_decay = 0
        self.seed = 42


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(config.pretrained_path)
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x, y):
        x = torch.permute(x, (1, 0, 2))
        y = torch.permute(y, (1, 0, 2))

        # 2*batch_size*seq_len
        input_ids = x[0]
        attention_mask = x[1]
        decoder_attention_mask = y[0]
        label_ids = y[1]
        
        input_ids = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        decoder_attention_mask = decoder_attention_mask.contiguous().view(-1, decoder_attention_mask.size(-1))
        label_ids = label_ids.contiguous().view(-1, label_ids.size(-1))
        
        loss = self.model(input_ids=input_ids, attention_mask=attention_mask,
                          decoder_attention_mask=decoder_attention_mask, labels=label_ids).loss
        predict = self.model.generate(input_ids)
        
        return loss, predict, label_ids
