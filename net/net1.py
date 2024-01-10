import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import os

class Model(torch.nn.Module):
     def __init__(self):
         super().__init__()
         self.conv = nn.Conv2d(1, 1, 3, padding=1)
         self.relu = nn.ReLU()
         self.conv.weight.data.fill_(1)
         self.conv.bias.data.fill_(0)
     
     def forward(self, x):
         x = self.conv(x)
         x = self.relu(x)
         return x


