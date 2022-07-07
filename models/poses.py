import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as RotLib
from torch.utils.tensorboard import SummaryWriter
import logging.config
import os
import datetime

now = datetime.datetime.now()

save_path = f"logs_poses/logs_{now}"

import sys

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(save_path)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# logging.basicConfig(filename=save_path, 
# 					format='%(name)s - %(levelname)s - %(message)s', 
# 					filemode='w') 

logger = logging.getLogger('spam_application')
logger.setLevel(logging.DEBUG)

def convert3x4_4x4(input):
    """
    :param input:  (N, 3, 4) or (3, 4) torch or np
    :return:       (N, 4, 4) or (4, 4) torch or np
    """
    if torch.is_tensor(input):
        if len(input.shape) == 3:
            output = torch.cat([input, torch.zeros_like(input[:, 0:1])], dim=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = torch.cat([input, torch.tensor([[0,0,0,1]], dtype=input.dtype, device=input.device)], dim=0)  # (4, 4)
            # print("this is it")
    else:
        if len(input.shape) == 3:
            output = np.concatenate([input, np.zeros_like(input[:, 0:1])], axis=1)  # (N, 4, 4)
            output[:, 3, 3] = 1.0
        else:
            output = np.concatenate([input, np.array([[0,0,0,1]], dtype=input.dtype)], axis=0)  # (4, 4)
            output[3, 3] = 1.0
    return output


def vec2skew(v):
    """
    :param v:  (3, ) torch tensor
    :return:   (3, 3)
    """
    zero = torch.zeros(1, dtype=torch.float32, device=v.device)
    skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
    skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
    skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
    return skew_v  # (3, 3)

def Exp(r):
    """so(3) vector to SO(3) matrix
    :param r: (3, ) axis-angle, torch tensor
    :return:  (3, 3)
    """
    skew_r = vec2skew(r)  # (3, 3)
    norm_r = r.norm() + 1e-15
    eye = torch.eye(3, dtype=torch.float32, device=r.device)
    R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
    return R

def make_c2w(r, t):
    """
    :param r:  (3, ) axis-angle             torch tensor
    :param t:  (3, ) translation vector     torch tensor
    :return:   (4, 4)
    """
    R = Exp(r)  # (3, 3)
    c2w = torch.cat([R, t.unsqueeze(1)], dim=1)  # (3, 4)
    # print(c2w)
    c2w = convert3x4_4x4(c2w)  # (4, 4)
    # print(c2w)
    return c2w


class LearnPose(nn.Module):
    def __init__(self, num_cams, init_c2w=None):
        """
        :param num_cams:
        :param learn_R:  True/False
        :param learn_t:  True/False
        :param init_c2w: (N, 4, 4) torch tensor
        """
        super(LearnPose, self).__init__()
        self.num_cams = num_cams
        self.device = torch.device('cuda')
        self.init_c2w = init_c2w
        self.init_c2w.requires_grad = False
        random_noise = 0.08*torch.randn(3, 4)
        random_noise = torch.cat((random_noise, torch.tensor([[0, 0, 0, 0]])), dim = 0)
        self.init_c2w_noisy = init_c2w + random_noise
        if init_c2w is not None:
            self.init_c2w_noisy = nn.Parameter(self.init_c2w_noisy, requires_grad=False)


        self.r = nn.Parameter(torch.zeros(size=(self.num_cams, 3), dtype=torch.float32), requires_grad=True)  # (N, 3)
        self.t = nn.Parameter(torch.zeros(size=(self.num_cams, 3), dtype=torch.float32), requires_grad=True)  # (N, 3)

    def forward(self, cam_id):
        r = self.r[cam_id]  # (3, ) axis-angle
        t = self.t[cam_id]  # (3, )
        c2w = make_c2w(r, t)  # (4, 4)

        # learn a delta pose between init pose and target pose, if a init pose is provided
        if self.init_c2w is not None:
            c2w = c2w @ self.init_c2w_noisy[cam_id]
        logging.info(f"This is the {cam_id}th extrinsics matrix: {c2w}")
        logging.info(f"This is the {cam_id}th original matrix: {self.init_c2w[cam_id]}")
        return c2w