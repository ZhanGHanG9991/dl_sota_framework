import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    """
    An RNN model followed by a Linear layer to map from seq_len to pred_len.
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.in_dim = configs.enc_in  # 通道数，输入和输出的channel维度

        # 定义RNN层，这里使用最简单的RNN，hidden_size与in_dim一致
        self.rnn = nn.RNN(input_size=self.in_dim, hidden_size=self.in_dim, batch_first=True)
        
        # 定义一个线性层，用于将时间维度从 seq_len 映射到 pred_len
        # 注意这里与原始模型类似，我们对张量进行permute后在线性层上操作
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        
        # 使用RNN对输入序列进行编码
        # RNN输入和输出: [B, L, C] -> [B, L, C]
        x, _ = self.rnn(x)  # x依旧是[B, seq_len, in_dim]

        # 将x的时间维度与通道维度交换，目的是对时间维度进行线性映射
        # x: [B, seq_len, in_dim] -> [B, in_dim, seq_len]
        x = x.permute(0, 2, 1)

        # 使用Linear将seq_len映射到pred_len
        # 线性层后x: [B, in_dim, pred_len]
        x = self.Linear(x)

        # 再次permute回去 [B, pred_len, in_dim]
        x = x.permute(0, 2, 1)

        # 输出维度: [Batch, Output length, Channel]
        return x
