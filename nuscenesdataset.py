import os.path as osp
import os
import shutil
import torch

from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader, Dataset, extract_zip, Data
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


class NuScenesLoader(Dataset):
    def len(self):
        return self.__len__()

    def download(self):
        pass

    @property
    def processed_file_names(self):
        # TODO
        return ["filler"]

    @property
    def raw_file_names(self):
        return ["filler"]

    def __init__(self, root, train=True, transform=None, pre_transform=None, pre_filter=None):
        self.nusc_source = os.environ['NUSCENES_SOURCE']
        if train:
            self.dataset = NuScenes(version='v1.0-trainval', dataroot=self.nusc_source, verbose=True)
        else:
            self.dataset = NuScenes(version='v1.0-test', dataroot=self.nusc_source, verbose=True)
        super(NuScenesLoader, self).__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.transforms = transform
        if train:
            path = self.processed_paths[0]
        else:
            path = self.processed_paths[1]

        self.data, self.slices = torch.load(path)

    def get_data(self, idx):
        this_annotation = self.dataset.sample_annotation[idx]
        sample_data_tokens = self.dataset.get('sample', this_annotation['sample_token'])['data']
        sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                   'LIDAR_TOP']

        # FIELDS x y z is_lidar intensity vx vy
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
                pointcloud = RadarPointCloud.from_file(osp.join(self.nusc_source, filename))
                is_lidar = 0
            else:
                pointcloud = LidarPointCloud.from_file(osp.join(self.nusc_source, filename))
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
            # points_ego_frame FIELDS: x y z is_lidar intensity vx vy
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

    def process(self):
        torch.save(self.process_set("train"), self.processed_paths[0])
        torch.save(self.process_set("test"), self.processed_paths[1])

    def process_set(self, dataset):
        categories = []
        for category in self.dataset.category:
            categories.append(category['name'])

        for i in range(len(self)):
            print("Currently processing point cloud number {} of {}".format(i, len(self)))
            points_array, sample_category = self.get_data(i)

            target = categories.index(sample_category)
            points_array = np.transpose(points_array)
            data = Data(x=points_array, y=torch.tensor([target]))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

    def __len__(self):
        return len(self.dataset.sample_annotation)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data


class NuScenesDataset(BaseDataset):

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)

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
