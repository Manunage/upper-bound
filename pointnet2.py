import torch

from torch_points3d.applications.pointnet2 import PointNet2


class PointNet2Classifier(torch.nn.Module):
    def __init__(self):
        USE_NORMAL = True
        NUM_CATEGORIES = 10  # Nuscenes:23 (32 if background classes are included)
        super().__init__()
        self.encoder = PointNet2("encoder", input_nc=3 * USE_NORMAL, output_nc=NUM_CATEGORIES, num_layers=4)
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
