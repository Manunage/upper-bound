from pointnet2.data.NuScenesLoaderXYZ import NuScenesLoaderXYZ
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

'''
data = [[], []]
for idx in indices:
    if idx % 1000 == 0:
        print('At index {}'.format(idx))
    elem = dataset[idx]
    data[0].append(len(elem[0]))
    data[1].append(elem[1])


idxs_not_none = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices.npy')
pts = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/pointsNumData.npy')
pts = np.asarray(pts)

pts_nbrs = [8, 16, 32, 64, 128]
for min_pts in pts_nbrs:
    data = []
    for count, idx in enumerate(idxs_not_none):
        if idx % 100000 == 0:
            print('At index {}'.format(idx))
        if pts[0][count] >= min_pts:
            data.append(idx)

    # save to npy file
    np.save('../models/goodIndices{}.npy'.format(min_pts), data)

'''

points_data = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/pointsNumData.npy')
points_data = np.asarray(points_data)

idxs = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices.npy')
idxs_8 = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices8.npy')
idxs_16 = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices16.npy')
idxs_32 = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices32.npy')
idxs_64 = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices64.npy')
idxs_128 = np.load('/home/mnagel/PycharmProjects/Pointnet2_PyTorch/goodIndices128.npy')

print(len(idxs))
print(len(idxs_8))
print(len(idxs_16))
print(len(idxs_32))
print(len(idxs_64))
print(len(idxs_128))


def get_labels(data, min_pts):
    labels = []
    for idx in range(len(data[0])):
        if data[0][idx] >= min_pts:
            labels.append(data[1][idx])
    return labels


def number_to_category(numbers):
    categories = ['human.pedestrian.adult', 'human.pedestrian.child', 'human.pedestrian.wheelchair',
                  'human.pedestrian.stroller', 'human.pedestrian.personal_mobility', 'human.pedestrian.police_officer',
                  'human.pedestrian.construction_worker', 'animal', 'vehicle.car', 'vehicle.motorcycle',
                  'vehicle.bicycle', 'vehicle.bus.bendy', 'vehicle.bus.rigid', 'vehicle.truck', 'vehicle.construction',
                  'vehicle.emergency.ambulance', 'vehicle.emergency.police', 'vehicle.trailer',
                  'movable_object.barrier', 'movable_object.trafficcone', 'movable_object.pushable_pullable',
                  'movable_object.debris', 'static_object.bicycle_rack']
    result = []
    for num in numbers:
        result.append(categories[num])
    return result


def coefficient_of_variation(data):
    standard_deviation = np.std(data)
    mean = np.mean(data)
    return 100*standard_deviation / mean


def get_label_frequencies(min_points=0):
    labels = get_labels(points_data, min_points)
    counter = Counter(labels)
    occs = []
    for i in range(23):
        occs.append(counter[i])
    return occs

'''
pts_coeff_var = [[], []]
for min_pts in [1, 8, 16, 32, 64, 128]:
    freqs = get_label_frequencies(min_pts)
    coefficient = coefficient_of_variation(freqs)
    pts_coeff_var[0].append(min_pts)
    pts_coeff_var[1].append(coefficient)
print(pts_coeff_var)

plt.plot(pts_coeff_var[0], pts_coeff_var[1], 'ro')
plt.xlabel('Minimum number of points filtered for')
plt.ylabel('CV')
plt.show()
'''

pts_anns = [[], []]
for min_pts in [1, 8, 16, 32, 64, 128]:
    num_anns = len(get_labels(points_data, min_pts))
    pts_anns[0].append(min_pts)
    pts_anns[1].append(num_anns)

plt.plot(pts_anns[0], pts_anns[1], 'ro')
plt.xlabel('Minimum number of points filtered for')
plt.ylabel('Number of remaining annotations')
plt.show()
