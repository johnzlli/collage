import torch
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter

class MATMUL(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.parameter.Parameter(data = torch.randn(1024, 1024) * 0.01)
        self.q = nn.parameter.Parameter(data = torch.randn(1024, 1024) * 0.01)
        self.k = nn.parameter.Parameter(data = torch.randn(1024, 1024) * 0.01)
        self.v = nn.parameter.Parameter(data = torch.randn(1024, 1024) * 0.01)

        self.relu = nn.ReLU()

    def forward(self, x):
        q = torch.matmul(x, self.q)
        #k = torch.matmul(x, self.k)
        #v = torch.matmul(x, self.v)

        q = torch.reshape(q, (64,16,64)).permute(1,0,2)
        #k = torch.reshape(k, (64,16,64)).permute(1,0,2)
        #v = torch.reshape(v, (64,16,64)).permute(1,0,2)

        #logits = torch.matmul(q, k)
        #output = torch.matmul(logits, v)

        #output = torch.reshape(output, (64, 1024))
        output = torch.reshape(q, (64, 1024))
        output = torch.matmul(output+1, self.w)
        k = torch.matmul(output, self.k)
        v = torch.matmul(output, self.v)
        k = torch.reshape(k, (64,16,64)).permute(1,0,2)
        v = torch.reshape(v, (64,16,64)).permute(1,0,2)
        y = torch.matmul(v, k)

        return y

data_raw=np.ones((64, 1024), np.float32)
data_x = torch.from_numpy(data_raw)
net = MATMUL()
torch.onnx.export(net, data_x, "matmul.onnx")

