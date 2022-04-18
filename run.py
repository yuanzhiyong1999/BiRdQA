# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network, test
from importlib import import_module
import argparse
from utils import build_dataset, build_iterator, get_time_dif

parser = argparse.ArgumentParser(description='Riddle dataset')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert-base-zh, bert-base-en, bert-wwm-ext,'
                                                             'roberta-wwm-ext-large, ernie' )
parser.add_argument('--dataset', type=str, required=True, help='choose a dataset: BiRdQA, riddlesense')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = args.dataset  # 数据集

    model_name = args.model  # 模型
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    start_time = time.time()
    print("Loading data...")
    train_text, train_label, dev_text, dev_label,  test_text, test_label = build_dataset(config)
    train_iter = build_iterator((train_text, train_label), config.batch_size, True)
    dev_iter = build_iterator((dev_text, dev_label), config.batch_size)
    test_iter = build_iterator((test_text, test_label), config.batch_size)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    model = x.Model(config).to(config.device)
    train(config, model, train_iter, dev_iter)

#    test
    test_model = x.Model(config).to(config.device)
    test(config, test_model, test_iter)

