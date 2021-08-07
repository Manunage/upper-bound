# Source: https://colab.research.google.com/github/nicolas-chaulet/torch-points3d/blob/master/notebooks/ObjectClassificationRSConv.ipynb
# 2021-07-27

# Needed for remote rendering
import os
import sys
from omegaconf import OmegaConf
import pyvista as pv
import torch
import time
import datetime
from torch_points3d.applications.rsconv import RSConv

os.environ["DISPLAY"] = ":99.0"
os.environ["PYVISTA_OFF_SCREEN"] = "true"
os.environ["PYVISTA_PLOT_THEME"] = "true"
os.environ["PYVISTA_USE_PANEL"] = "true"
os.environ["PYVISTA_AUTO_CLOSE"] = "false"
os.system("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")

nusc_path = os.environ['NUSCENES_PATH']

USE_NORMAL = True
DIR = os.path.dirname(os.path.realpath(__file__))  # data will go in DIR/data.

# The dataset will be downloaded the first time this cell is run
from torch_points3d.datasets.classification.nuscenesdataset import NuScenesLoader
import torch_points3d.core.data_transform as T3D
import torch_geometric.transforms as T

pre_transform = T.Compose([T.NormalizeScale(), T3D.GridSampling3D(0.02)])
dataset = NuScenesLoader(train=True, transform=None,
                          pre_transform=pre_transform, pre_filter=None)


class RSConvClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = RSConv("encoder", input_nc=3 * USE_NORMAL, output_nc=23, num_layers=4)
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    @property
    def conv_type(self):
        """ This is needed by the dataset to infer which batch collate should be used"""
        return self.encoder.conv_type

    def get_output(self):
        """ This is needed by the tracker to get access to the ouputs of the network"""
        return self.output

    def get_labels(self):
        """ Needed by the tracker in order to access ground truth labels"""
        return self.labels

    def get_current_losses(self):
        """ Entry point for the tracker to grab the loss """
        return {"loss_class": float(self.loss_class)}

    def forward(self, data):
        # Set labels for the tracker
        self.labels = data.y.squeeze()

        # Forward through the network
        data_out = self.encoder(data)
        self.output = self.log_softmax(data_out.x.squeeze())

        # Set loss for the backward pass
        self.loss_class = torch.nn.functional.nll_loss(self.output, self.labels)

    def backward(self):
        self.loss_class.backward()


model = RSConvClassifier()

from torch_points3d.datasets.batch import SimpleBatch

NUM_WORKERS = 4
BATCH_SIZE = 12

transform = T.FixedPoints(2048)
dataset = NuScenesLoader(train=True, transform=transform,
                          pre_transform=pre_transform, pre_filter=None)

collate_function = lambda datalist: SimpleBatch.from_data_list(datalist)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_function
)
next(iter(train_loader))

yaml_config = """
task: classification
class: nuscenesdataset.NuScenesDataset
name: nuscenesdataset
dataroot: %s
number: %s
pre_transforms:
    - transform: NormalizeScale
    - transform: GridSampling3D
      lparams: [0.02]
train_transforms:
    - transform: FixedPoints
      lparams: [2048]
    - transform: RandomNoise
    - transform: RandomRotate
      params:
        degrees: 180
        axis: 2
    - transform: AddFeatsByKeys
      params:
        feat_names: [norm]
        list_add_to_x: [%r]
        delete_feats: [True]
test_transforms:
    - transform: FixedPoints
      lparams: [2048]
    - transform: AddFeatsByKeys
      params:
        feat_names: [norm]
        list_add_to_x: [%r]
        delete_feats: [True]
""" % (os.environ['NUSCENES_PATH'], USE_NORMAL, USE_NORMAL)

from omegaconf import OmegaConf

params = OmegaConf.create(yaml_config)

# Instantiate dataset
from torch_points3d.datasets.classification.nuscenesdataset import NuScenesDataset

dataset = NuScenesDataset(params)
print(dataset)

# Setup the data loaders
dataset.create_dataloaders(
    model,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    precompute_multi_scale=False
)

next(iter(dataset.test_dataloaders[0]))

# Setup the tracker and actiavte tensorboard loging
logdir = os.path.join(DIR, "outputs")  # Replace with your own path
logdir = os.path.join(logdir, str(datetime.datetime.now()))
os.mkdir(logdir)
os.chdir(logdir)
tracker = dataset.get_tracker(False, True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

from torch_points3d.metrics.colored_tqdm import Coloredtqdm as Ctq


def train_epoch(device):
    model.to(device)
    model.train()
    tracker.reset("train")
    train_loader = dataset.train_dataloader
    iter_data_time = time.time()
    with Ctq(train_loader) as tq_train_loader:
        for i, data in enumerate(tq_train_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            optimizer.zero_grad()
            data.to(device)
            model.forward(data)
            model.backward()
            optimizer.step()
            if i % 10 == 0:
                tracker.track(model)

            tq_train_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()


def test_epoch(device):
    model.to(device)
    model.eval()
    tracker.reset("test")
    test_loader = dataset.test_dataloaders[0]
    iter_data_time = time.time()
    with Ctq(test_loader) as tq_test_loader:
        for i, data in enumerate(tq_test_loader):
            t_data = time.time() - iter_data_time
            iter_start_time = time.time()
            data.to(device)
            model.forward(data)
            tracker.track(model)

            tq_test_loader.set_postfix(
                **tracker.get_metrics(),
                data_loading=float(t_data),
                iteration=float(time.time() - iter_start_time),
            )
            iter_data_time = time.time()

EPOCHS = 50
for i in range(EPOCHS):
  print("=========== EPOCH %i ===========" % i)
  time.sleep(0.5)
  train_epoch('cuda')
  tracker.publish(i)
  test_epoch('cuda')
  tracker.publish(i)
