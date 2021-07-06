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

show_empty_boxes_stats = False


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
                # self.boxes = load_gt(nusc=self.dataset, eval_split='mini_train', box_cls=DetectionBox, verbose=True)
                pass
        else:
            # TODO load train and test datasets here
            pass


    def __getitem__(self, idx):
        this_annotation = self.dataset.sample_annotation[idx]
        sample_data_tokens = self.dataset.get('sample', this_annotation['sample_token'])['data']
        sensors = ['RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT',
                   'LIDAR_TOP']
        # all_points = []
        for sensor_name in sensors:
            sample_data = self.dataset.get('sample_data', sample_data_tokens[sensor_name])
            # Get box (sensor coordinate frame)
            _, boxes, _ = self.dataset.get_sample_data(sample_data_token=sample_data_tokens[sensor_name],
                                                       selected_anntokens=[this_annotation['token']])
            box = boxes[0]
            filename = sample_data['filename']
            # Get points (sensor coordinate frame?)
            if sample_data['sensor_modality'] == 'radar':
                pointcloud = RadarPointCloud.from_file(osp.join(dataroot, filename))
            elif sample_data['sensor_modality'] == 'lidar':
                pointcloud = LidarPointCloud.from_file(osp.join(dataroot, filename))

            # transpose for points_in_box and get only first 3 dimensions (coordinates) and
            points_with_metadata = np.transpose(pointcloud.points)
            points = []
            for point_data in points_with_metadata:
                points.append(point_data[:3])
            points = np.transpose(points)
            filter_mask = points_in_box(box=box, points=points)
            filtered_points = points[:, filter_mask]
            print(filtered_points)
            # TODO Works for sensors individually. Transform each to ego pose to get common frame?

        label = this_annotation['category_name']
        return filtered_points, label


    def __len__(self):
        return len(self.dataset.sample_annotation)

    def set_num_points(self, pts):
        self.num_points = min(int(1e4), pts)


# Prints stats about boxes without any points in them
def print_empty_boxes_stats(dset):
    number_of_boxes = len(dset.dataset.sample_annotation)
    num_boxes_without_lidar = 0
    num_boxes_without_radar = 0
    num_boxes_without_any = 0
    for annotation in dset.dataset.sample_annotation:
        if annotation['num_lidar_pts'] == 0 and annotation['num_radar_pts'] == 0:
            num_boxes_without_any += 1
            num_boxes_without_radar += 1
            num_boxes_without_lidar += 1
        elif annotation['num_lidar_pts'] == 0:
            num_boxes_without_lidar += 1
        elif annotation['num_radar_pts'] == 0:
            num_boxes_without_radar += 1
    print('There are {} annotations total.'.format(number_of_boxes))
    print('There are {} annotations without any lidar points.'.format(num_boxes_without_lidar))
    print('There are {} annotations without any radar points.'.format(num_boxes_without_radar))
    print('{} annotations contain no points at all!'.format(num_boxes_without_any))


dataset = NuScenesLoader(16, train=True)

# print_empty_boxes_stats(dataset)
