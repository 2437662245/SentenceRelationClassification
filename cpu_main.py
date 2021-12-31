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
from single_GCN import SGCN
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():

    # seed_torch(999)
    # 定义参数
    BATCH_SIZE = 64
    EPOCH = 4
    learning_rate = 0.0002

    # 读入数据
    sentence1 = torch.load('E:\PythonProjects\sgcn\sentences1_new.pt')
    # print(sentence1.shape)  # torch.Size([1420, 44, 300])
    sentence2 = torch.load('E:\PythonProjects\sgcn\sentences2_new.pt')
    # print(sentence2.shape)  # torch.Size([1420, 36, 300])
    label = torch.load('E:\PythonProjects\sgcn\labels_new.pt')
    # label = label.long()
    # label = torch.Tensor(label).float()
    label = torch.Tensor(label).long()

    # 实例化模型
    sgcn = SGCN()
    # 优化器
    optimizer = optim.Adam(params=sgcn.parameters(), lr=learning_rate)
    # loss
    loss_func = CrossEntropyLoss()

    # 记录训练次数
    total_train_step = 0
    # 记录测试次数
    total_test_step = 0

    # 划分数据  1136/284
    train_inputs1, test_inputs1, train_inputs2, test_inputs2, train_labels, test_labels = train_test_split(
        sentence1, sentence2, label, random_state=666, test_size=0.2)

    train_data = TensorDataset(train_inputs1, train_inputs2, train_labels)
    test_data = TensorDataset(test_inputs1, test_inputs2, test_labels)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for epoch in range(EPOCH):
        start_time = time.time()
        # 训练
        sgcn.train()
        train_total_accuracy = 0
        train_total_loss = 0
        test_total_loss = 0
        test_total_accuracy = 0
        for index, (batch_input1, batch_input2, batch_label) in enumerate(tqdm(train_dataloader)):
            train_right = 0
            # 数据传入网络
            output = sgcn(batch_input1, batch_input2)
            # print("train_output:", output)
            # 计算loss
            loss = loss_func(output, batch_label)
            train_total_loss += loss.item()
            # 优化器模型 先将梯度置零
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
            label = batch_label.cpu().numpy().tolist()
            # print("predict", len(predict))
            # print("predict:", predict)
            # print("batch_label", batch_label)
            for i in range(0, batch_label.size(0)):
                if predict[i] == label[i]:
                    train_right += 1
            train_total_accuracy += (train_right / train_labels.shape[0])

            total_train_step += 1
            # if total_train_step % 18 == 0:
            #     print("训练次数: {}, Loss: {}".format(total_train_step, loss.item()))
        # 测试
        sgcn.eval()

        with torch.no_grad():
            for index, (batch_input1, batch_input2, batch_label) in enumerate(tqdm(test_dataloader)):
                test_right = 0
                # 数据传入网络
                output = sgcn(batch_input1, batch_input2)
                #print(output)
                # 计算lossx
                loss = loss_func(output, batch_label)
                test_total_loss += loss.item()
                predict = torch.argmax(output, dim=1).cpu().numpy().tolist()
                label = batch_label.cpu().numpy().tolist()
                # print("predict", predict)
                # print("label", label)
                # print("predict", len(predict))
                for i in range(0, batch_label.size(0)):
                    if predict[i] == label[i]:
                        test_right += 1
                # print("test_right", test_right)
                test_total_accuracy += (test_right / test_labels.shape[0])
        end_time = time.time()
        print("\n")
        print("第 {}/{} 个epoch完成，用时 {} S".format(epoch + 1, EPOCH, end_time - start_time))
        print("训练集上Loss为: {}".format(train_total_loss))
        print("训练集上的正确率: {}".format(train_total_accuracy))
        print("测试集上Loss为: {}".format(test_total_loss))
        print("测试集上的正确率: {}".format(test_total_accuracy))
        total_test_step = total_test_step + 1

if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print("CPU运行总用时{}".format(end_time-start_time))
