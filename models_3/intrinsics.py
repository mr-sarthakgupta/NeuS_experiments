import torch
import torch.nn as nn
import numpy as np
from models.datasetmm import Dataset, load_K_Rt_from_P
from pyhocon import ConfigFactory
import os

class LearnFocal(nn.Module):
    def __init__(self, num_cams, intrinsic_init):
        super(LearnFocal, self).__init__()
        self.param = nn.Parameter((torch.from_numpy(intrinsic_init)).float(), requires_grad=True)
        
    def forward(self, i):
        print(f"intrinsics sum = {torch.sum(self.param)}")
        return self.param[i]