import os
import os.path as osp

import numpy as np
import torch
import torch.utils.data as data
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion


class NuScenesLoaderFeats(data.Dataset):
    def __init__(self, num_points, root=os.environ['NUSCENES_PATH'], transforms=None, train=True):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_points = num_points
        self.train = train
        self.dataset = NuScenes(version='v1.0-trainval', dataroot=self.root, verbose=True)
        self.categories = []
        for category in self.dataset.category:
            self.categories.append(category['name'])

    def __getitem__(self, idx):
        this_annotation = self.dataset.sample_annotation[idx]
        sample_data_tokens = self.dataset.get('sample', this_annotation['sample_token'])['data']
        sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                   'LIDAR_TOP']

        sample_category = this_annotation['category_name']
        target = self.categories.index(sample_category)

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

        points_ego_frame = np.float32(points_ego_frame)
        points_ego_frame = np.transpose(points_ego_frame)

        try:
            if self.transforms is not None:
                points_ego_frame = self.transforms(points_ego_frame)
            points_ego_frame = self.fixed_points(points_ego_frame)
        except ValueError:
            return None

        points_ego_frame = np.array(points_ego_frame, dtype=np.float32)

        return points_ego_frame, target

    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(1024, pts)

    # From torch geometric
    def fixed_points(self, points):
        starting_len = len(points)
        if starting_len >= self.num_points:
            choice = np.random.choice(starting_len, self.num_points, replace=False)
        else:
            choice = np.random.choice(starting_len, self.num_points, replace=True)
        result = []
        for idx in choice:
            result.append(points[idx])
        return result

    def get_good_indices(self):
        good_idxs = []
        for idx in range(len(self)):
            if idx % 1000 == 0:
                print('At index {}'.format(idx))
            if self[idx] is not None:
                good_idxs.append(idx)
        return good_idxs

