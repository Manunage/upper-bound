from nuscenes import NuScenes
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_gt
import os
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
import open3d as o3d
import h5py

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rel_data_path = '/data/sets/nuscenes'
dataroot = BASE_DIR + rel_data_path

# Test code with mini dataset
mini_testrun = True


# references: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#             https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/data

class NuScenesLoader(data.Dataset):
    def __init__(self, num_points, root=dataroot, transforms=None, train=True, dataset=None):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_points = num_points
        if mini_testrun:
            self.dataset = NuScenes(version='v1.0-mini', dataroot=self.root, verbose=True)
        else:
            #TODO prepare train and test datasets here
            pass

    def __getitem__(self, idx):
        annotation = self.dataset.sample_annotation[idx]
        print(annotation)
        sample = self.dataset.get('sample', annotation['sample_token'])
        print(sample)
        if annotation['num_lidar_pts'] > 0:
            pass
        if annotation['num_radar_pts'] > 0:
            pass

        return annotation

    # TODO Return number of annotations
    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


dataset = NuScenesLoader(16, train=True)

#print(dataset[46])
boxes = load_gt(nusc=dataset.dataset, eval_split='mini_train', box_cls=DetectionBox, verbose=True)
for b in boxes:
    print(b)
