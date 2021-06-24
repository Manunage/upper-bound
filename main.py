from nuscenes import NuScenes

# import os
# import os.path as osp
# import numpy as np

nusc = NuScenes(version='v1.0-mini', dataroot='/data/sets/nuscenes', verbose=True)

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
for sample_annotation in nusc.sample_annotation:
    print(sample_annotation)


#################


# access pointclouds by traversing sample_data and filtering for 'pcd' datatype
def get_pointclouds():
    for data in nusc.sample_data:
        if data['fileformat'] == 'pcd':
            # do something with pointclouds here, e.g. access file at path data['filename']
            print(data)
            sample = nusc.get('sample', data['sample_token'])
            print(sample)


get_pointclouds()
