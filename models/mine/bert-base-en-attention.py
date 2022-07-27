# coding: UTF-8
# 将bert输出的张量进行attention
# 效果一般 35左右
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from utils import attention


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
        self.batch_size = 2  # mini-batch大小
        self.pad_size = 256  # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-5  # 学习率
        self.pretrained_path = 'bert-base-uncased'
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
        # 3*batch_size*n_choice*seq_len
        x = torch.permute(x, (2, 0, 1, 3))

        input_ids = x[0]
        token_type_ids = x[1]
        attention_mask = x[2]

        batch_size = input_ids.shape[0]
        num_choices = input_ids.shape[1]
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # [batch_size*num_choices,seq_len,hidden_size]
        output_all_encoded_layers = out[0]
        # print(output_all_encoded_layers.shape)
        reshaped_out = output_all_encoded_layers.view(batch_size, num_choices, input_ids.size(-1), -1)
        # print(reshaped_out[0][:][:10])

        # print(torch.cat(torch.chunk(reshaped_out, 6, 1)[1:], 1).shape)
        # chunk用来等分 将reshaped_out在第一维上等分为6份
        # q = torch.chunk(reshaped_out, 6, 1)[0]
        # a = torch.cat(torch.chunk(reshaped_out, 6, 1)[1:], 1)
        # choice_out = torch.tensor([]).to('cuda')
        # for i in range(batch_size):
        #     q = reshaped_out[1][0]
        #     temp = torch.tensor([]).to('cuda')
        #     for j in range(1, num_choices):
        #         a = reshaped_out[i][j]
        #         v, _ = attention(a, q, q)
        #         temp = torch.cat((temp, v.unsqueeze(0)))
        #     choice_out = torch.cat((choice_out, temp.unsqueeze(0)))

        # mask = torch.zeros(2, 5, 256, 256)

        q = torch.index_select(reshaped_out, dim=1, index=torch.tensor([0]).to('cuda'))
        a = torch.index_select(reshaped_out, dim=1, index=torch.tensor([1, 2, 3, 4, 5]).to('cuda'))

        attention_mask = attention_mask.view(batch_size, num_choices, attention_mask.size(-1))
        mask = torch.index_select(attention_mask, dim=1, index=torch.tensor([0]).to('cuda'))
        mask = torch.cat((mask, torch.zeros(batch_size, 255, 256).to('cuda')), 1)
        mask = torch.unsqueeze(mask, 1)

        v, _ = attention(a, q, q, mask)
        v = self.dropout(v)
        ans = torch.index_select(v, dim=2, index=torch.tensor([0]).to('cuda'))

        logits = self.classifier(ans)
        reshaped_logits = logits.view(-1, num_choices - 1)
        return reshaped_logits
