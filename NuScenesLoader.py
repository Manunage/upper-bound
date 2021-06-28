from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.geometry_utils import points_in_box
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
print(BASE_DIR)
print(dataroot)


# references: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#             https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/data

class NuScenesLoader(data.Dataset):
    def __init__(self, num_points, root=dataroot, transforms=None, train=True, mini_testrun=True):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_points = num_points
        if mini_testrun:
            self.dataset = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)
            if train:
                # TODO maybe not use load_gt and instead go directly by self.dataset.annotation
                self.boxes = load_gt(nusc=self.dataset, eval_split='mini_train', box_cls=DetectionBox, verbose=True)
        else:
            #TODO load train and test datasets here
            pass

    def __getitem__(self, idx):
        annotation = self.dataset.sample_annotation[idx]
        sample_data_tokens = self.dataset.get('sample', annotation['sample_token'])['data']
        sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT', 'LIDAR_TOP']
        all_points = []
        for sensor_name in sensors:
            sample_data = self.dataset.get('sample_data', sample_data_tokens[sensor_name])
            filename = sample_data['filename']
            if sample_data['sensor_modality'] == 'radar':
                pointcloud = RadarPointCloud.from_file(osp.join(dataroot, filename))
            elif sample_data['sensor_modality'] == 'lidar':
                pointcloud = LidarPointCloud.from_file(osp.join(dataroot, filename))
            # Get only first 3 dimensions (coordinates)
            for point_data in pointcloud.points:
                all_points.append(point_data[:3])

        # TODO all_points has wrong dimensions for points_in_box. Tranpose? Maybe needs to be ndarray, not list.
        filter_mask = points_in_box(box=self.dataset.get_box(annotation['token']), points=all_points)
        filtered_points = all_points[filter_mask]
        label = annotation['category_name']
        return filtered_points, label

    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


dataset = NuScenesLoader(16, train=True)

print(dataset.boxes.__repr__)
