import os
import random
import time
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from GCN import SGCN
from torch import optim, nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


def seed_torch(seed=999):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    # 定义参数
    BATCH_SIZE = 64
    EPOCH = 4
    learning_rate = 0.0001

    # 读入数据
    sentence1 = torch.load('sentences1_new.pt')
    sentence2 = torch.load('sentences2_new.pt')
    label = torch.load('labels_new.pt')
    label = torch.Tensor(label).long()

    # 实例化模型
    sgcn = SGCN(device).to(device)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # 就这一行
        sgcn = nn.DataParallel(sgcn)
    # 优化器
    optimizer = optim.Adam(params=sgcn.parameters(), lr=learning_rate)
    # loss
    loss_func = CrossEntropyLoss().to(device)
    # loss_func = loss_func.to(device)
    # 记录训练次数
    total_train_step = 0
    # 记录测试次数
    total_test_step = 0

    # 划分数据
    train_inputs1, test_inputs1, train_inputs2, test_inputs2, train_labels, test_labels = train_test_split(
        sentence1, sentence2, label, random_state=666, test_size=0.2)

    train_data = TensorDataset(train_inputs1, train_inputs2, train_labels)
    test_data = TensorDataset(test_inputs1, test_inputs2, test_labels)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for epoch in range(EPOCH):
        start_time = time.time()
        total_test_loss = 0
        total_test_accuracy = 0
        total_train_loss = 0
        total_train_accuracy = 0
        # 训练
        sgcn.train()
        for index, (batch_input1, batch_input2, batch_label) in enumerate(tqdm(train_dataloader)):
            train_right = 0
            batch_input1 = batch_input1.to(device)
            batch_input2 = batch_input2.to(device)
            batch_label = batch_label.to(device)

            # 数据传入网络
            output = sgcn(batch_input1, batch_input2)
            # print("train_output", output)
            # 计算loss
            loss = loss_func(output, batch_label)
            # 优化器模型 先将梯度置零
            total_train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            # print("predict", len(predict))
            for i in range(0, batch_label.size(0)):
                if predict[i] == label[i]:
                    train_right += 1
            total_train_accuracy += (train_right / train_labels.shape[0])
            total_train_step += 1
            # if total_train_step % 18 == 0:
            #     print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
        # 测试
        sgcn.eval()
        # sgcn = sgcn.cuda()
        with torch.no_grad():
            for index, (batch_input1, batch_input2, batch_label) in enumerate(tqdm(test_dataloader)):
                test_right = 0
                batch_input1 = batch_input1.to(device)
                batch_input2 = batch_input2.to(device)
                batch_label = batch_label.to(device)
                # 数据传入网络
                output = sgcn(batch_input1, batch_input2)
                # print(output)
                # 计算loss
                loss = loss_func(output, batch_label)
                total_test_loss += loss.item()
                predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()
                # print("predict", len(predict))
                # print("predict", predict)
                # print("label", label)
                for i in range(0, batch_label.size(0)):
                    if predict[i] == label[i]:
                        test_right += 1
                # print("test_right", test_right)
                total_test_accuracy += (test_right / test_labels.shape[0])
        end_time = time.time()
        print("第 {}/{} 个epoch完成，用时 {} S".format(epoch + 1, EPOCH, end_time - start_time))
        print("训练集上Loss为: {}".format(total_train_loss))
        print("训练集上的正确率: {}".format(total_train_accuracy))
        print("测试集上Loss为: {}".format(total_test_loss))
        print("测试集上的正确率: {}".format(total_test_accuracy))
        total_test_step = total_test_step + 1


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("单GPU总用时 {} S".format(end_time - start_time))