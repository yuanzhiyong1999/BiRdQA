# coding: UTF-8
import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
import time
from datetime import timedelta


def fixed_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def build_zh_dataset(config):
    def load_dataset(path, pad_size=256, ):
        text, target, corr = [], [], []
        data = pd.read_csv(path)

        for index, line in tqdm(data.iterrows()):
            riddle, choice0, choice1, choice2, choice3, choice4, label = line['riddle'], line['choice0'], \
                                                                         line['choice1'], line['choice2'], \
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
    dev_text, dev_label = load_dataset(config.dev_path, config.pad_size)
    test_text, test_label = load_dataset(config.test_path, config.pad_size)
    return train_text, train_label, dev_text, dev_label, test_text, test_label


def build_en_dataset(config):
    def load_BiRdQA_dataset(path, pad_size=256, ):
        text, target, corr = [], [], []
        data = pd.read_csv(path)
        for index, line in tqdm(data.iterrows()):
            riddle, choice0, choice1, choice2, choice3, choice4, label = line['riddle'], line['choice0'], \
                                                                         line['choice1'], line['choice2'], \
                                                                         line['choice3'], line['choice4'], \
                                                                         line['label']
            choice = [choice0, choice1, choice2, choice3, choice4]
            group = []
            for i in choice:
                token = config.tokenizer.encode_plus(text=riddle, text_pair=i, add_special_tokens=True,
                                                     max_length=pad_size, padding='max_length', truncation=True,
                                                     return_attention_mask=True, return_tensors='pt')
                group.append([token.input_ids.tolist(), token.token_type_ids.tolist(), token.attention_mask.tolist()])
            group = torch.tensor(group)
            text.append(group)
            target.append(label)
            corr = torch.tensor(target)

        text = [i.tolist() for i in text]
        text = torch.tensor(text)
        question = torch.squeeze(text)
        return question, corr

    def load_riddlesense_dataset(path, pad_size=256, ):
        text, target, corr = [], [], []
        dict = {"A": "0", "B": "1", "C": "2", "D": "3", "E": "4", "hidden": "5"}
        data = pd.read_json(path, lines=True)
        for index, line in tqdm(data.iterrows()):
            riddle, choice0, choice1, choice2, choice3, choice4, label = line.question['stem'], \
                                                                         line.question['choices'][0]['text'], \
                                                                         line.question['choices'][1]['text'], \
                                                                         line.question['choices'][2]['text'], \
                                                                         line.question['choices'][3]['text'], \
                                                                         line.question['choices'][4]['text'], \
                                                                         line.answerKey

            for key, value in dict.items():
                label = label.replace(key, value)
            label = int(label)
            choice = [choice0, choice1, choice2, choice3, choice4]
            group = []
            for i in choice:
                token = config.tokenizer.encode_plus(text=riddle, text_pair=i, add_special_tokens=True,
                                                     max_length=pad_size, padding='max_length', truncation=True,
                                                     return_attention_mask=True, return_tensors='pt')
                group.append([token.input_ids.tolist(), token.token_type_ids.tolist(), token.attention_mask.tolist()])
            group = torch.tensor(group)
            text.append(group)
            target.append(label)
            corr = torch.tensor(target)

        text = [i.tolist() for i in text]
        text = torch.tensor(text)
        question = torch.squeeze(text)
        return question, corr

    if config.dataset == 'BiRdQA':
        train_text, train_label = load_BiRdQA_dataset(config.train_path, config.pad_size)
        dev_text, dev_label = load_BiRdQA_dataset(config.dev_path, config.pad_size)
        test_text, test_label = load_BiRdQA_dataset(config.test_path, config.pad_size)
    else:
        train_text, train_label = load_riddlesense_dataset(config.train_path, config.pad_size)
        dev_text, dev_label = load_riddlesense_dataset(config.dev_path, config.pad_size)
        test_text, test_label = load_riddlesense_dataset(config.test_path, config.pad_size)
    return train_text, train_label, dev_text, dev_label, test_text, test_label


def build_t5_dataset(config):
    def load_dataset(path, pad_size=150, ):
        ques, ans = [], []
        data = pd.read_csv(path)
        for index, line in tqdm(data.iterrows()):
            riddle, choice0, choice1, choice2, choice3, choice4, label = line['riddle'], line['choice0'], \
                                                                         line['choice1'], line['choice2'], \
                                                                         line['choice3'], line['choice4'], \
                                                                         line['label']
            choice = [choice0, choice1, choice2, choice3, choice4]
            text = riddle + ' \\n' + ' (A) ' + choice0 + ' (B) ' + choice1 + ' (C) ' + \
                   choice2 + ' (D) ' + choice3 + ' (E) ' + choice4

            encoding = config.tokenizer(text, max_length=pad_size, padding="max_length",
                                        truncation=True, return_tensors="pt")
            input_ids, attention_mask = encoding.input_ids, encoding.attention_mask
            target_encoding = config.tokenizer(choice[label], max_length=9, padding="max_length", return_tensors="pt")
            decoder_attention_mask, label_ids = target_encoding.attention_mask, target_encoding.input_ids

            # print(choice[label])
            # print(label_ids[0][0])
            # x = config.model.generate(input_ids)
            # print(x[0][1])
            # x = config.tokenizer.decode(x[0], skip_special_tokens=True)
            # print(x)
            # exit()

            label_ids[label_ids == config.tokenizer.pad_token_id] = -100
            ques.append([input_ids.tolist(), attention_mask.tolist()])
            ans.append([decoder_attention_mask.tolist(), label_ids.tolist()])

        ques = torch.tensor(ques)
        ans = torch.tensor(ans)

        ques = torch.squeeze(ques)
        ans = torch.squeeze(ans)

        return ques, ans

    train_text, train_label = load_dataset(config.train_path, config.pad_size)
    dev_text, dev_label = load_dataset(config.dev_path, config.pad_size)
    test_text, test_label = load_dataset(config.test_path, config.pad_size)
    return train_text, train_label, dev_text, dev_label, test_text, test_label


def build_rd_dataset(config):
    data = pd.read_json(config, lines=True)
    for index, line in tqdm(data.iterrows()):
        print(line.id)
        print(line.question['stem'])
        print(line.question['choices'][0]['label'], line.question['choices'][0]['text'])
        print(line.question['choices'][1]['label'], line.question['choices'][1]['text'])
        print(line.question['choices'][2]['label'], line.question['choices'][2]['text'])
        print(line.question['choices'][3]['label'], line.question['choices'][3]['text'])
        print(line.question['choices'][4]['label'], line.question['choices'][4]['text'])
        print(line.answerKey)
        exit()


def build_iterator(data_arrays, batch_size, is_train=False):
    """Construct a PyTorch data iterator."""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, num_workers=4, shuffle=is_train)


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
