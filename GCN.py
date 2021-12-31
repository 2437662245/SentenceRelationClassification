import math
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from torch.nn import Parameter, init, ReLU


class GCN(nn.Module):
    def __init__(self, in_features, out_features, device, bias=True):
        super(GCN, self).__init__()
        self.device = device
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))    # 256 128
        self.relu = ReLU()
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        init.xavier_uniform(self.weight.data, gain=1)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, matrix_x, adj_matrix_processed):
        """
        64*80*256 64*80*80
        :param matrix_x: LSTM的输出
        :param adj_matrix_processed: 邻接矩阵
        :return:
        """
        weight_matrix = self.weight.repeat(matrix_x.shape[0], 1, 1)    # 为什么要repeat？ 调整维度，保证他俩个能够相乘？ 64 256 128
        matrix_last2 = torch.bmm(matrix_x, weight_matrix)   # 64*80*256  64*256*128         # 64 80 128
        output = torch.bmm(adj_matrix_processed.to(self.device), matrix_last2.to(self.device))      # 64 80 128
        output = self.relu(output)
        # if self.bias is not None:
        #     return output + self.bias.repeat(output.size(0))
        # else:
        return output

    def __repr__(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


def creat_dad_matrix(lstm_out1, lstm_out2):
    matrix_dad = []
    lstm_out1 = lstm_out1.cpu().detach().numpy()    # 64, 44 ,256
    lstm_out2 = lstm_out2.cpu().detach().numpy()  # 64, 36 ,256
    # print(lstm_out1.shape[0])   # 64
    # print(lstm_out1[0].shape)   # (44, 256)
    # print(lstm_out2[0].shape)   # (36, 256)
    for i in range(lstm_out1.shape[0]):  # 遍历所有的batch_size
        matrix_m = cosine_similarity(lstm_out1[i], lstm_out2[i])    # 计算相似矩阵M
        # print(matrix_m.shape)   # torch.Size([44, 36])
        matrix_m = torch.from_numpy(matrix_m)
        m_transpose = torch.transpose(matrix_m, 0, 1)   # M的转置矩阵
        # print(m_transpose.shape)    # torch.Size([36, 44])
        identity_matrix1 = torch.eye(matrix_m.shape[0])  # 44*44的单位矩阵
        identity_cat_m = torch.cat((identity_matrix1, matrix_m), 1)  # 按行拼接单位矩阵和M 44*80
        identity_matrix1 = torch.eye(m_transpose.shape[0])  # 36*36的单位矩阵
        m_transpose_cat_identity1 = torch.cat((m_transpose, identity_matrix1), 1)   # 按行拼接M转和单位矩阵
        matrix_a = torch.cat((identity_cat_m, m_transpose_cat_identity1), 0)    # 拼接出A~矩阵 80*80
        a_dim = matrix_a.shape[0]   # 矩阵A的维度 80
        arr_d = np.zeros((a_dim, a_dim), dtype=float)   # 80*80的0矩阵

        for k in range(a_dim):
            sum_row = 0
            for j in range(a_dim):
                sum_row += matrix_a[k][j].item()
            if sum_row > 0:
                arr_d[k][k] += 1 / math.sqrt(sum_row)
        matrix_d = torch.from_numpy(arr_d)  #
        matrix_a = matrix_a.to(torch.float32)
        matrix_d = matrix_d.to(torch.float32)
        matrix_dad_i = torch.matmul(torch.matmul(matrix_d, matrix_a), matrix_d)
        arr_dad_i = matrix_dad_i.numpy()
        matrix_dad.append(arr_dad_i)
        # print("matrix_dad[{}]:{}".format(i, matrix_dad_i))

    matrix_dad = np.array(matrix_dad, dtype=np.float32)
    # print(matrix_dad.shape)
    matrix_dad = torch.from_numpy(matrix_dad)
    # print(matrix_dad.shape)  # 64*80*80
    return matrix_dad


class SGCN(nn.Module):
    def __init__(self, device):
        super(SGCN, self).__init__()
        self.device = device
        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=128,
            num_layers=3,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        self.lstm2 = nn.LSTM(
            input_size=256,
            hidden_size=1280,
            num_layers=2,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )
        self.gcn = GCN(128 * 2, 128, self.device)
        # self.pooling = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.max_pooling = nn.MaxPool2d(kernel_size=3)
        self.avg_pooling = nn.AvgPool2d(kernel_size=3)
        self.fc = nn.Linear(42, 3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input_sentences1, input_sentences2):
        lstm_out1, _ = self.lstm(input_sentences1)  # 64*44*256
        lstm_out2, _ = self.lstm(input_sentences2)  # 64*36*256
        # for i in range(20):
        #     lstm_out1, _ = self.lstm2(lstm_out1)  # 64*44*256
        #     lstm_out2, _ = self.lstm2(lstm_out2)  # 64*36*256
        lstm_out = torch.cat((lstm_out1, lstm_out2), 1)  # X矩阵64*80*256
        # matrix_x = torch.transpose(lstm_out, 1, 2)  # X矩阵64*256*80
        dad_matrix = creat_dad_matrix(lstm_out1, lstm_out2)   # 64*80*80
        output_gcn = self.gcn(lstm_out.to(self.device), dad_matrix.to(self.device))   # 64* 80 *128
        output_max_pooling = self.max_pooling(output_gcn)
        output_avg_pooling = self.avg_pooling(output_gcn)
        # print("output_max_pooling.shape", output_max_pooling.shape)  # torch.Size([64, 26, 42])
        # print("output_avg_pooling.shape", output_avg_pooling.shape)  # torch.Size([64, 26, 42])
        output_pooling = torch.cat((output_max_pooling, output_avg_pooling), 1)
        # print(output_pooling.shape)  # torch.Size([64, 52, 42])

        output_fc = self.fc(output_pooling)
        # print("output_fc", output_fc)
        output = torch.max(output_fc, 1)[0]  # 输出每行最大值
        # print("output", output)
        output = self.softmax(output)
        return output
