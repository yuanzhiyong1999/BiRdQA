"""
@FileName ：roberta-large.py
@Author ：Zhiyong Yuan
@Date ：2022/4/23 20:26 
@Tools ：PyCharm
@Description：
"""
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaForMultipleChoice


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_type = 'en'  # 不同的类型 不同的数据处理方法
        self.model_name = 'roberta-large'
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
        self.num_epochs = 3  # epoch数
        self.batch_size = 2  # mini-batch大小
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.pretrained_path = 'pretrained_models/roberta-large'
        self.tokenizer = RobertaTokenizer.from_pretrained(self.pretrained_path)
        self.weight_decay = 1e-4
        self.seed = 42


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.roberta = RobertaForMultipleChoice.from_pretrained(config.pretrained_path)
        for param in self.roberta.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = torch.permute(x, (2, 0, 1, 3))
        # 2*batch_size*n_choice*seq_len
        input_ids = x[0]
        attention_mask = x[1]

        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits