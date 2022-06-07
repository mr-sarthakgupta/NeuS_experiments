import torch
import torch.nn as nn
import numpy as np
from models.datasetmm import Dataset, load_K_Rt_from_P
from pyhocon import ConfigFactory
import os

# class LearnFocal(nn.Module):
#     def __init__(self, H, W, req_grad, fx_only, order=2, init_focal=None):
#         super(LearnFocal, self).__init__()
#         self.H = H
#         self.W = W
#         self.fx_only = fx_only  # If True, output [fx, fx]. If False, output [fx, fy]
#         self.order = order  # check our supplementary section.

#         if self.fx_only:
#             if init_focal is None:
#                 self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
#             else:
#                 if self.order == 2:
#                     # a**2 * W = fx  --->  a**2 = fx / W
#                     coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
#                 elif self.order == 1:
#                     # a * W = fx  --->  a = fx / W
#                     coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
#                 else:
#                     print('Focal init order need to be 1 or 2. Exit')
#                     exit()
#                 self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
#         else:
#             if init_focal is None:
#                 self.fx = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
#                 self.fy = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
#                 self.v1 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
#                 self.v2 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=req_grad)  # (1, )
#             else:
#                 if self.order == 2:
#                     # a**2 * W = fx  --->  a**2 = fx / W
#                     coe_x = torch.tensor(np.sqrt(init_focal / float(W)), requires_grad=False).float()
#                     coe_y = torch.tensor(np.sqrt(init_focal / float(H)), requires_grad=False).float()
#                 elif self.order == 1:
#                     # a * W = fx  --->  a = fx / W
#                     coe_x = torch.tensor(init_focal / float(W), requires_grad=False).float()
#                     coe_y = torch.tensor(init_focal / float(H), requires_grad=False).float()
#                 else:
#                     print('Focal init order need to be 1 or 2. Exit')
#                     exit()
#                 self.fx = nn.Parameter(coe_x, requires_grad=req_grad)  # (1, )
#                 self.fy = nn.Parameter(coe_y, requires_grad=req_grad)  # (1, )

#     def forward(self, i=None):  # the i=None is just to enable multi-gpu training
#         if self.fx_only:
#             if self.order == 2:
#                 fxfy = torch.stack([self.fx ** 2 * self.W, self.fx ** 2 * self.W])
#             else:
#                 fxfy = torch.stack([self.fx * self.W, self.fx * self.W])
#         else:
#             if self.order == 2:
#                 fxfy = torch.stack([self.fx**2 * self.W, self.fy**2 * self.H])
#                 v = torch.stack([self.v1, self.v2])
#                 intrinsics_matrix = torch.stack([fxfy, v], dim = 1)
#             else:
#                 fxfy = torch.stack([self.fx * self.W, self.fy * self.H])
#                 intrinsics_matrix = torch.stack([fxfy, v], dim = 1)
#         return intrinsics_matrix


class LearnFocal(nn.Module):
    def __init__(self, num_cams, conf_path):
        super(LearnFocal, self).__init__()
        self.conf_path = conf_path
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', 'bmvs_bear')
        f.close()
        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', 'bmvs_bear')
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        data_obj = Dataset(self.conf['dataset'])
        intrinsic_np = data_obj.get_intrinsics()
        self.param = nn.Parameter((torch.from_numpy(intrinsic_np)).float(), requires_grad=True)
        # self.param = nn.Parameter(torch.rand([num_cams, 4, 4]), requires_grad=True)

    def forward(self, i):
        print(f"sum = {torch.sum(self.param)}")
        return self.param[i]