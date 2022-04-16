# coding: UTF-8
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import time
from datetime import timedelta

def build_dataset(config):

    def load_dataset(path, pad_size=256,):
        text, target, corr = [], [], []
        data = pd.read_csv(path)

        for index, line in tqdm(data.iterrows()):
            riddle, choice0, choice1, choice2, choice3, choice4, label = line['riddle'], line['choice0'], \
                                                                         line['choice1'], line['choice2'],\
                                                                         line['choice3'], line['choice4'], \
                                                                         line['label']
            choice = [choice0, choice1, choice2, choice3, choice4]

            # 去掉（打一物）这样的提示
            riddle = riddle.split(' ')[0]
            group = []
            for i in choice:

                token = config.tokenizer.encode_plus(text=riddle, text_pair=i, add_special_tokens=True,
                                                   max_length=pad_size, padding='max_length', truncation=True,
                                                 return_attention_mask=True, return_tensors='pt')
                # print(token.input_ids.size())

                group.append([token.input_ids.tolist(), token.token_type_ids.tolist(), token.attention_mask.tolist()])

            group = torch.tensor(group)
            text.append(group)
            target.append(label)
            corr = torch.tensor(target)

        text = [i.tolist() for i in text]
        text = torch.tensor(text)
        question = torch.squeeze(text)

        return question, corr
    train_text, train_label = load_dataset(config.train_path, config.pad_size)
    # print(train_text.shape)
    # print(train_label.shape)
    # exit()
    dev_text, dev_label = load_dataset(config.dev_path, config.pad_size)
    test_text, test_label = load_dataset(config.test_path, config.pad_size)
    return train_text, train_label, dev_text, dev_label, test_text, test_label



def build_iterator(data_arrays, batch_size, is_train=False):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, num_workers=4, shuffle=is_train)

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
