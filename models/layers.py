import torch
from torch import nn
import numpy as np

from utils.miscellaneous_utils import get_device


class SepConvTemp3D(nn.Module):
    def __init__(self, depth_wise_parameters, point_wise_parameters, input_dim):
        super(SepConvTemp3D, self).__init__()
        in_channels = depth_wise_parameters["in_channels"]
        kernel_size = depth_wise_parameters["kernel_size"]
        out_channels = depth_wise_parameters["out_channels"]
        padding = depth_wise_parameters["padding"]

        # We create input_dim number of layers for depth convolution
        depth_wise_layers = []
        for i in range(input_dim):
            layer_i = nn.Conv3d(in_channels=in_channels, kernel_size=kernel_size, out_channels=out_channels,
                                padding=padding, device=get_device(), bias=False)
            depth_wise_layers.append(layer_i)
        self.conv3d_temp_depth_wise = nn.ModuleList(depth_wise_layers)
        self.__weights_initialization_depth_wise(kernel_size)

        in_channels = point_wise_parameters["in_channels"]
        kernel_size = point_wise_parameters["kernel_size"]
        out_channels = point_wise_parameters["out_channels"]

        # We create a single layer for point convolution
        self.conv3d_temp_point_wise = nn.Conv3d(in_channels=in_channels, kernel_size=kernel_size,
                                                out_channels=out_channels,
                                                device=get_device(), bias=False)
        self.__weights_initialization_point_wise([input_dim, input_dim])

    def __weights_initialization_depth_wise(self, filter_size):
        # we use an identity filter
        filter_size = list(filter_size)
        filter_initializer = np.zeros([1, 1] + filter_size, dtype=np.float32)
        filter_initializer[0, 0, int(filter_size[0] / 2.0), int(filter_size[1] / 2.0), int(filter_size[2] / 2.0)] = 1.0

        # iterate through each layer and manually set weights
        with torch.no_grad():
            for layer in self.conv3d_temp_depth_wise:
                layer.weight = nn.Parameter(torch.as_tensor(filter_initializer, device=get_device()))

    def __weights_initialization_point_wise(self, input_size):
        # we use an identity filter
        filter_initializer = np.zeros(input_size + [1, 1, 1], dtype=np.float32)
        for i in range(input_size[0]):
            filter_initializer[i, i, :, :, :] = 1.0

        # manually set weights
        with torch.no_grad():
            self.conv3d_temp_point_wise.weight = nn.Parameter(torch.as_tensor(filter_initializer,
                                                                              device=get_device()))

    def forward(self, features):
        no_features = features.shape[1]

        # Introduce a fake "batch dimension", previous dimension becomes time
        # -1 means to preserve the current dim
        curr_features = features.expand(1, -1, -1, -1, -1)
        curr_features = torch.reshape(curr_features,
                                      (curr_features.shape[0], curr_features.shape[2], curr_features.shape[1],
                                       curr_features.shape[3], curr_features.shape[4]))

        # In the following part we will do a depthwise convolution
        curr_features = list(torch.split(curr_features, 1, dim=1))

        for channel_no in range(no_features):
            curr_features[channel_no] = self.conv3d_temp_depth_wise[channel_no](curr_features[channel_no])

        # Stack the channels together
        curr_features = torch.cat(curr_features, dim=1)

        # Now we do the pointwise convolution
        curr_features = self.conv3d_temp_point_wise(curr_features)

        # Remove the fake dimension and switch no_channels and batch_size back
        curr_features = torch.reshape(curr_features,
                                      (curr_features.shape[0], curr_features.shape[2], curr_features.shape[1],
                                       curr_features.shape[3], curr_features.shape[4]))
        curr_features = torch.squeeze(curr_features, dim=0)

        return curr_features
