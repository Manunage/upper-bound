import os
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np
import torch.utils.data as data
from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import points_in_box
from pyquaternion import Quaternion

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
rel_data_path = '/data/sets/nuscenes'
dataroot = BASE_DIR + rel_data_path
print(BASE_DIR)
print(dataroot)


# references: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
#             https://github.com/erikwijmans/Pointnet2_PyTorch/tree/master/pointnet2/data

class NuScenesLoader(data.Dataset):
    def __init__(self, num_points, root=os.environ['NUSCENES_PATH'], transforms=None, train=True, mini_testrun=False):
        super().__init__()
        self.root = root
        self.transforms = transforms
        self.num_points = num_points
        if mini_testrun:
            self.dataset = NuScenes(version='v1.0-mini', dataroot=self.root, verbose=True)
            if train:
                pass
        else:
            if train:
                self.dataset = NuScenes(version='v1.0-trainval', dataroot=self.root, verbose=True)
            else:
                self.dataset = NuScenes(version='v1.0-test', dataroot=self.root, verbose=True)
            pass

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


dataset = NuScenesLoader(16, train=True, mini_testrun=False)

from collections import Counter
from matplotlib.pyplot import hist, hist2d, scatter
from matplotlib import pyplot as plt


# Prints stats about boxes without any points in them
def print_stats(dset):
    num_boxes = len(dset.dataset.sample_annotation)
    num_boxes_without_lidar = 0
    num_boxes_without_radar = 0
    num_boxes_without_any = 0
    total_num_points = 0
    max_points_in_one_annotation = 0
    points_numbers = []
    for annotation in dset.dataset.sample_annotation:
        points_in_annotation = annotation['num_lidar_pts'] + annotation['num_radar_pts']
        total_num_points = total_num_points + points_in_annotation
        points_numbers.append(points_in_annotation)
        if points_in_annotation > max_points_in_one_annotation:
            max_points_in_one_annotation = points_in_annotation
        if points_in_annotation == 0:
            num_boxes_without_any += 1
            num_boxes_without_radar += 1
            num_boxes_without_lidar += 1
        elif annotation['num_lidar_pts'] == 0:
            num_boxes_without_lidar += 1
        elif annotation['num_radar_pts'] == 0:
            num_boxes_without_radar += 1

    frequency_counter = Counter(points_numbers)
    frequency = []
    for key in range(max_points_in_one_annotation + 1):
        value = frequency_counter.get(key)
        frequency.append(0) if value is None else frequency.append(value)

    print('There are {} annotations total.'.format(num_boxes))

    print('There are {} annotations without any lidar points.'.format(num_boxes_without_lidar))
    print('That means the ratio of annotations without any lidar points to total annotations is {}'.format(
        num_boxes_without_lidar / num_boxes))

    print('There are {} annotations without any radar points.'.format(num_boxes_without_radar))
    print('That means the ratio of annotations without any radar points to total annotations is {}'.format(
        num_boxes_without_radar / num_boxes))

    print('{} annotations contain no points at all!'.format(num_boxes_without_any))
    print('That means the ratio of annotations without any points to total annotations is {}'.format(
        num_boxes_without_any / num_boxes))

    print('There are {} points total.'.format(total_num_points))
    print('That means the average number of points per annotation is {}'.format(total_num_points / num_boxes))

    print('The maximum number of points in one annotation is {}'.format(max_points_in_one_annotation))

    print(frequency)

    max_value = 500
    bins = list(range(max_value))
    bins.append(max_value - 0.0001)
    histogram = hist(points_numbers, bins=bins, log=True, histtype='bar')

    plt.title('Point frequency distribution (annotations)')
    plt.xlabel('Number of points')
    plt.ylabel('Occurrences')


# print_stats(dataset)


def lidar_stats(dset):
    # FIELDS distance intensity
    point_stats = [[], []]
    # Number of points to look at. One million takes up to half a minute
    point_limit = 1000000

    point_counter = 0
    for sample_data in dset.dataset.sample_data:  # Get points (sensor coordinate frame)
        if point_counter < point_limit:
            if sample_data['sensor_modality'] == 'lidar' and sample_data['is_key_frame']:
                filename = sample_data['filename']
                pointcloud = LidarPointCloud.from_file(osp.join(dset.root, filename))
                points = pointcloud.points
                for point_index in range(len(pointcloud.points[0])):
                    point_counter = point_counter + 1
                    x = points[0][point_index]
                    y = points[1][point_index]
                    z = points[2][point_index]
                    distance = np.sqrt(np.power(x, 2) + np.power(y, 2) + np.power(z, 2))
                    intensity = points[3][point_index]
                    point_stats[0].append(distance)
                    point_stats[1].append(intensity)

    distance, intensity = point_stats[0], point_stats[1]

    histogram = hist(intensity, bins=100, log=True)
    plt.title('Intensity frequency distribution')
    plt.xlabel('Intensity')
    plt.ylabel('Occurrences')

    # histogram = hist2d(x=distance, y=intensity, bins=256, cmin=10)
    # histogram = hist2d(x=distance, y=intensity, bins=32, cmin=5)
    # histogram = hist2d(x=distance, y=intensity, bins=64, cmin=5, range=((0, 10), (0, 250)))
    # scatterplot = scatter(x=distance, y=intensity, s=0.1, alpha=0.5)

    # plt.title('Lidar points: intensity and distance')
    # plt.xlabel('Distance from sensor')
    # plt.ylabel('Intensity')


# lidar_stats(dataset)
