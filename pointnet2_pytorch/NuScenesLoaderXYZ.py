import os
import os.path as osp

import numpy as np
import torch
import torch.utils.data as data
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion


# references: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#             https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/data

class NuScenesLoaderXYZ(data.Dataset):
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

        # FIELDS x y z
        points_ego_frame = [[], [], []]
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
            else:
                pointcloud = LidarPointCloud.from_file(osp.join(self.root, filename))

            # Transform pointcloud from sensor coordinate frame to ego pose frame
            rotation_matrix = Quaternion(this_calibrated_sensor['rotation']).rotation_matrix
            pointcloud.rotate(rotation_matrix)
            pointcloud.translate(np.array(this_calibrated_sensor['translation']))

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
            # points_ego_frame FIELDS: x y z
            for point in range(len(filtered_points[0])):
                for i in range(3):
                    points_ego_frame[i].append(filtered_points[i][point])

        points_ego_frame = np.float32(points_ego_frame)
        points_ego_frame = np.transpose(points_ego_frame)

        try:
            if self.transforms is not None:
                points_ego_frame = self.transforms(points_ego_frame)
                if type(points_ego_frame) != list:
                    print('?')
            points_ego_frame = self.fixed_points(points_ego_frame)
        except ValueError:
            return None

        points_ego_frame = np.array(points_ego_frame, dtype=np.float32)
        #points_ego_frame = torch.from_numpy(points_ego_frame)

        return points_ego_frame, target

    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)

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


#dataset = NuScenesLoaderXYZ(4)

# data = np.asarray(dataset.get_good_indices())

# save to npy file
# np.save('goodIndices.npy', data)
# print('Done!')
