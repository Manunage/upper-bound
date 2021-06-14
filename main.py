from nuscenes import NuScenes
import os
import os.path as osp
import numpy as np


script_path = osp.dirname(__file__)
rel_data_path = '/data/sets/nuscenes'
dataroot = script_path + rel_data_path

# nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)


#################
# playing around with nuscenes schema
#first_scene = nusc.scene[0]
#print("first_scene:")
#print(first_scene)
#first_sample_token = first_scene['first_sample_token']
#print("first_sample_token:")
#print(first_sample_token)
#first_sample = nusc.get('sample', first_sample_token)
#print("first_sample:")
#print(first_sample)
#first_sample_data = nusc.get('sample_data', first_sample['data']['RADAR_FRONT_RIGHT'])
#print("first_sample_data:")
#print(first_sample_data)
#next_sample_token = first_sample_data['next']
#print(next_sample_token)
#################


# access pointclouds by traversing sample_data and filtering for 'pcd' datatype
def get_pointclouds():
    for data in nusc.sample_data:
        if data['fileformat'] == 'pcd':
            # do something with pointclouds here, e.g. access file at path data['filename']
            pass

get_pointclouds()
