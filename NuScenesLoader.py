from nuscenes import NuScenes
import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rel_data_path = '/data/sets/nuscenes'
dataroot = BASE_DIR + rel_data_path

version = 'v1.0-mini'

# references: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#             https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/data

class NuScenesLoader(data.Dataset):
    def __init__(self, num_points, dataroot=dataroot, transforms=None, train=True):
        # TODO
        self.dataroot = dataroot
        self.transforms = transforms
        self.num_points = num_points
        # Probably does not belong here:
        nusc = NuScenes(version=version, dataroot=self.dataroot, verbose=True)

    def __getitem__(self, idx):
        # TODO
        pass

    def __len__(self):
        # TODO
        pass

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


dataset = NuScenesLoader(16, train=True)
