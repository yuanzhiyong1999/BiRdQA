# coding: UTF-8
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup
from torch.nn.functional import softmax


# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=42):
    for name, w in model.named_parameters():
        if exclude not in name:
            if len(w.size()) < 2:
                continue
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


class Accumulator:
    """在n个变量上累加  累加器"""

    def __init__(self, n):
        # 若n=2 则self.data = [0.0,0.0]
        self.data = [0.0] * n

    def add(self, *args):
        # 若传来的*args为（4，5） 则结果为[4.0,5.0]
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    y_hat = y_hat.argmax(dim=1)
    num_correct = torch.eq(y_hat, y).sum().float().item()
    return num_correct


def train(config, model, train_iter, dev_iter):
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_iter), config.num_epochs * len(train_iter))
    criterion = nn.CrossEntropyLoss()
    bast_acc = 0

    model.train()
    for epoch in range(config.num_epochs):
        metric = Accumulator(3)
        loop = tqdm(enumerate(train_iter), total=len(train_iter))
        for i, (X, y) in loop:
            X = X.to(config.device)
            y = y.to(config.device)

            with autocast():
                outputs = model(X)
                loss = criterion(outputs, y)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss = loss.item()
            train_acc = accuracy(outputs, y)

            metric.add(train_loss, train_acc, y.shape[0])

            loop.set_description(f'TrainEpoch: [{epoch + 1}/{config.num_epochs}]')
            loop.set_postfix(acc=metric[1] / metric[2], loss=metric[0] / metric[2])

        eva_acc = evaluate(model, dev_iter, config.device)
        print(f"TrainEpoch:[{epoch + 1}/{config.num_epochs}], evaluate-acc = {eva_acc}")

        if bast_acc < eva_acc:
            bast_acc = eva_acc
            torch.save(model.state_dict(), config.save_path)
            print("Model Saved!")


def evaluate(model, data_iter, device):
    model.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        loop = tqdm(enumerate(data_iter), total=len(data_iter))
        for i, (X, y) in loop:
            X = X.to(device)
            y = y.to(device)

            with autocast():
                outputs = model(X)

            acc = accuracy(outputs, y)
            metric.add(acc, y.shape[0])

            loop.set_description(f'evaluate: ')
            loop.set_postfix(acc=metric[0] / metric[1])
    return metric[0] / metric[1]


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc = evaluate(model, test_iter, config.device)
    print("test-acc:", test_acc)


def t5_accuracy(y_hat, y):
    num_correct = 0
    for i in range(0, y_hat.shape[0]):
        index = True  # 标记是否成功
        for j in range(0, y_hat.shape[1]):
            if y_hat[i][j + 1] == 1:
                break
            if y_hat[i][j + 1] != y[i][j]:
                index = False
                break
        if index:
            num_correct = num_correct + 1
    return num_correct


def t5_train(config, model, train_iter, dev_iter):
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, len(train_iter), config.num_epochs * len(train_iter))
    bast_acc = 0

    model.train()
    for epoch in range(config.num_epochs):
        metric = Accumulator(3)
        loop = tqdm(enumerate(train_iter), total=len(train_iter))
        for i, (X, y) in loop:
            X = X.to(config.device)
            y = y.to(config.device)
            loss, predict, label_ids = model(X, y)
            # predict = config.tokenizer.decode(predict, skip_special_tokens=True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss = loss.item()
            train_acc = t5_accuracy(predict, label_ids)
            metric.add(train_loss, train_acc, y.shape[0])

            loop.set_description(f'TrainEpoch: [{epoch + 1}/{config.num_epochs}]')
            loop.set_postfix(acc=metric[1] / metric[2], loss=metric[0] / metric[2])

        eva_acc = t5_evaluate(model, dev_iter, config.device)
        print(f"TrainEpoch:[{epoch + 1}/{config.num_epochs}], evaluate-acc = {eva_acc}")

        if bast_acc < eva_acc:
            bast_acc = eva_acc
            torch.save(model.state_dict(), config.save_path)
            print("Model Saved!")


def t5_evaluate(model, data_iter, device):
    model.eval()
    metric = Accumulator(2)

    with torch.no_grad():
        loop = tqdm(enumerate(data_iter), total=len(data_iter))
        for i, (X, y) in loop:
            X = X.to(device)
            y = y.to(device)

            _, predict, label_ids = model(X, y)

            acc = t5_accuracy(predict, label_ids)
            metric.add(acc, y.shape[0])

            loop.set_description(f'evaluate: ')
            loop.set_postfix(acc=metric[0] / metric[1])
    return metric[0] / metric[1]


def t5_test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    test_acc = t5_evaluate(model, test_iter, config.device)
    print("test-acc:", test_acc)
