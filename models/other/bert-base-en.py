# coding: UTF-8
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_type = 'en'  # 不同的类型 不同的数据处理方法
        self.model_name = 'bert-base-en'
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
        self.batch_size = 8  # mini-batch大小
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.pretrained_path = 'pretrained_models/bert-base-uncased'
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrained_path)
        self.weight_decay = 1e-4
        self.seed = 42


def init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrained_path)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 1)

        init_weights(self.classifier)

    def forward(self, x):
        x = torch.permute(x, (2, 0, 1, 3))
        # 3*batch_size*n_choice*seq_len
        input_ids = x[0]
        token_type_ids = x[1]
        attention_mask = x[2]

        # batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = out[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)
        return reshaped_logits
