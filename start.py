# coding: UTF-8
from train_eval import *
from importlib import import_module
import argparse
from utils import *

parser = argparse.ArgumentParser(description='Riddle dataset')
parser.add_argument('--model', type=str, required=True, help='choose a model: bert-base-zh, bert-base-en, bert-wwm-ext,'
                                                             'roberta-wwm-ext-large, ernie, unifiedqa-t5-large, '
                                                             'bert-large-en, roberta-large, albert-xxl')
parser.add_argument('--dataset', type=str, required=True, help='choose a dataset: BiRdQA, riddlesense')
args = parser.parse_args()

if __name__ == '__main__':
    dataset = args.dataset
    model_name = args.model
    x = import_module('models.' + model_name)
    config = x.Config(dataset)

    print('=='*10, 'config', '=='*10)
    print('model = ', config.model_name)
    print('dataset = ', dataset)
    print('batch_size = ', config.batch_size)
    print('lr = ', config.learning_rate)
    print('wd = ', config.weight_decay)
    print('=='*25)

    fixed_seed(config.seed)

    start_time = time.time()
    print("Loading data...")



    if config.model_type == 't5':
        train_text, train_label, dev_text, dev_label,  test_text, test_label = build_t5_dataset(config)
        train_iter = build_iterator((train_text, train_label), config.batch_size, True)
        dev_iter = build_iterator((dev_text, dev_label), config.batch_size)
        test_iter = build_iterator((test_text, test_label), config.batch_size)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        # train
        model = x.Model(config).to(config.device)
        t5_train(config, model, train_iter, dev_iter)

        #    test
        test_model = x.Model(config).to(config.device)
        t5_test(config, test_model, test_iter)
    else:
        if config.model_type == 'zh':
            # train_text, train_label, dev_text, dev_label,  test_text, test_label = build_zh_dataset(config)
            train_text, train_label, dev_text, dev_label, test_text, test_label = build_my_zh_dataset(config)
        else:
            # train_text, train_label, dev_text, dev_label, test_text, test_label = build_en_dataset(config)
            # train_text, train_label, dev_text, dev_label, test_text, test_label = build_my_en_dataset(config)
            train_text, train_label, dev_text, dev_label = build_CSQA_dataset(config)
        train_iter = build_iterator((train_text, train_label), config.batch_size, True)
        dev_iter = build_iterator((dev_text, dev_label), config.batch_size)
        # test_iter = build_iterator((test_text, test_label), config.batch_size)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)

        # train
        model = x.Model(config).to(config.device)
        train(config, model, train_iter, dev_iter)

    #    test
        test_model = x.Model(config).to(config.device)
        # test(config, test_model, test_iter)