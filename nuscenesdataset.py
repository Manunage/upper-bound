import os.path as osp
import os
import shutil
import torch

from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, InMemoryDataset, extract_zip, Data
import torch_geometric.transforms as T
from torch_geometric.io import read_txt_array

from torch_points3d.datasets.base_dataset import BaseDataset
from torch_points3d.metrics.classification_tracker import ClassificationTracker
from torch_points3d.utils.download import download_url

import os
import os.path as osp

import numpy as np
import torch.utils.data as data
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

######
# Nuscenes-dataset loader for torch_points3d/datasets/classification
######

class NuScenesLoader(data.Dataset):
    def __init__(self, num_points, root=os.environ['NUSCENES_PATH'], transforms=None, train=True):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_points = num_points
        if train:
            self.dataset = NuScenes(version='v1.0-trainval', dataroot=root, verbose=True)
        else:
            self.dataset = NuScenes(version='v1.0-test', dataroot=root, verbose=True)

    def __getitem__(self, idx):
        this_annotation = self.dataset.sample_annotation[idx]
        sample_data_tokens = self.dataset.get('sample', this_annotation['sample_token'])['data']
        sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                   'LIDAR_TOP']

        # FIELDS x y z rad/lid intensity vx vy
        points_ego_frame = [[], [], [], [], [], [], []]
        for sensor_name in sensors:
            this_sample_data = self.dataset.get('sample_data', sample_data_tokens[sensor_name])
            this_calibrated_sensor = self.dataset.get('calibrated_sensor', this_sample_data['calibrated_sensor_token'])

            # Get box (sensor coordinate frame)
            _, boxes, _ = self.dataset.get_sample_data(sample_data_token=sample_data_tokens[sensor_name],
                                                       selected_anntokens=[this_annotation['token']])
            box = boxes[0]

            # Transform box from sensor coordinate frame to ego pose frame
            box.rotate(Quaternion(this_calibrated_sensor['rotation']))
            box.translate(np.array(this_calibrated_sensor['translation']))

            # Get points (sensor coordinate frame)
            filename = this_sample_data['filename']
            if this_sample_data['sensor_modality'] == 'radar':
                pointcloud = RadarPointCloud.from_file(osp.join(self.root, filename))
                is_lidar = 0
            else:
                pointcloud = LidarPointCloud.from_file(osp.join(self.root, filename))
                is_lidar = 1

            # Transform pointcloud from sensor coordinate frame to ego pose frame
            rotation_matrix = Quaternion(this_calibrated_sensor['rotation']).rotation_matrix
            pointcloud.rotate(rotation_matrix)
            pointcloud.translate(np.array(this_calibrated_sensor['translation']))

            # Rotate velocities to ego pose frame (for radar)
            if not is_lidar:
                velocities = pointcloud.points[8:10, :]
                velocities = np.vstack((velocities, np.zeros(pointcloud.points.shape[1])))
                pointcloud.points[8:10, :] = np.dot(rotation_matrix, velocities)[:2]

            # transpose for points_in_box and get only first 3 dimensions (coordinates)
            points_with_metadata = np.transpose(pointcloud.points)
            points = []
            for point_data in points_with_metadata:
                points.append(point_data[:3])
            points = np.transpose(points)
            # filter for points in box
            filter_mask = points_in_box(box=box, points=points)
            filtered_points = pointcloud.points[:, filter_mask]

            # print(sensor_name)
            # print(filtered_points)

            # add points from this sensor to the points from the other sensors (in ego pose frame)
            # points_ego_frame FIELDS: x y z rad/lid intensity vx vy
            for point in range(len(filtered_points[0])):
                for i in range(3):
                    points_ego_frame[i].append(filtered_points[i][point])
                points_ego_frame[3].append(is_lidar)
                if is_lidar:
                    points_ego_frame[4].append(filtered_points[3][point])
                    points_ego_frame[5].append(0)
                    points_ego_frame[6].append(0)
                else:
                    points_ego_frame[4].append(0)
                    points_ego_frame[5].append(filtered_points[8][point])
                    points_ego_frame[6].append(filtered_points[9][point])

        label = this_annotation['category_name']
        return points_ego_frame, label

    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


class NuScenesDataset(BaseDataset):

    def __init__(self):
        super().__init__()

        self.train_dataset = NuScenesLoader(
            self._data_path,
            train=True,
            transform=self.train_transform,
            pre_transform=self.pre_transform,
        )
        self.test_dataset = NuScenesLoader(
            self._data_path,
            train=False,
            transform=self.test_transform,
            pre_transform=self.pre_transform,
        )

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool):
        """Factory method for the tracker
        Arguments:
            wandb_log - Log using weight and biases
            tensorboard_log - Log using tensorboard
        Returns:
            [BaseTracker] -- tracker
        """
        return ClassificationTracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)
