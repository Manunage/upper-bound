from nuscenes import NuScenes
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
            f = h5py.File('pointcloud_data.h5', 'w')
            mini_data = f.create_dataset("unlimited", (10000, 2), maxshape=(None, 10))
            #TODO get pointclouds into dataset
            for sample_data in self.dataset.sample_data:
                if sample_data['fileformat'] == 'pcd' and sample_data['is_key_frame']:
                    print(sample_data)
                    file = sample_data['filename']
                    print(file)
                    # pcd = o3d.io.read_point_cloud(pcd_file)
                    # out_arr = np.asarray(pcd.points)
        else:
            #TODO prepare train and test datasets here
            pass

    def __getitem__(self, idx):
        sample = self.dataset.sample[idx]

    # Return number of samples
    # TODO Maybe return number of instances? Or something else?
    def __len__(self):
        return len(self.dataset.sample)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


dataset = NuScenesLoader(16, train=True)

print(len(dataset))
