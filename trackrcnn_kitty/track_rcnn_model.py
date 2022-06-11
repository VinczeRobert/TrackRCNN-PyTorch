import warnings
from collections import OrderedDict

import torch.jit
import torchvision
from torch import nn, Tensor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import misc as misc_nn_ops, FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from trackrcnn_kitty.layers import SepConvTemp3D
from trackrcnn_kitty.losses import compute_association_loss
from trackrcnn_kitty.region_proposal_network_creator import RegionProposalNetworkCreator
from trackrcnn_kitty.roi_heads_creator import RoIHeadsCreator
from trackrcnn_kitty.utils import check_for_degenerate_boxes, validate_and_build_stacked_boxes


class TrackRCNN(nn.Module):
    def __init__(self, num_classes, backbone_output_dim=2048, batch_size=4):
        super(TrackRCNN, self).__init__()
        # Create a Transform object which will be responsible on applying transformations on the image
        # These parameters are taken from Pytorch code
        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        # We create the backbone: we'll use a pretrained Resnet101
        # that has been trained on the COCO dataset
        resnet101 = torchvision.models.resnet101(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.backbone = resnet101

        # Initialize Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024, 2048, 2048, 2048],
            256,
            LastLevelMaxPool()
        )

        # We create our two depth-wise separable Conv3D layers
        conv3d_parameters_1 = {
            "in_channels": 1,
            "kernel_size": (3, 3, 3),  # value used by the authors of TrackRCNN for the Conv3d layers
            "out_channels": 1,
            "padding": (1, 1, 1)
        }
        conv3d_parameters_2 = {
            "in_channels": backbone_output_dim,
            "kernel_size": (1, 1, 1),
            "out_channels": backbone_output_dim,
            "padding": None
        }
        self.conv3d_temp_1 = SepConvTemp3D(conv3d_parameters_1, conv3d_parameters_2, backbone_output_dim)
        self.conv3d_temp_2 = SepConvTemp3D(conv3d_parameters_1, conv3d_parameters_2, backbone_output_dim)

        self.relu = nn.ReLU()

        # Create the region proposal network
        self.rpn = RegionProposalNetworkCreator().get_instance()

        # Create RoI heads
        self.roi_heads = RoIHeadsCreator(num_classes).get_instance()

        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of detections
        # and the number of outputs was set by the authors to 128
        self.association_head = nn.Linear(in_features=batch_size, out_features=128)

        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        stacked_boxes = validate_and_build_stacked_boxes(targets)

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        check_for_degenerate_boxes(targets)

        # Run the images through the backbone (resnet101)
        feature_dict, features = self.__forward_backbone(images.tensors)

        # The next step is to send our features through our Conv3D layers
        features = self.conv3d_temp_1.forward(features)
        features = self.relu(features)
        feature_dict['4'] = features
        features = self.conv3d_temp_2.forward(features)
        features = self.relu(features)
        feature_dict['5'] = features

        feature_dict = self.fpn(feature_dict)

        if isinstance(feature_dict, Tensor):
            feature_dict = OrderedDict([("0", feature_dict)])

        proposals, proposel_losses = self.rpn(images, feature_dict, targets)
        detections, detector_losses = self.roi_heads(feature_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # The association head gets proposals as inputs
        associations = self.association_head(stacked_boxes)

        # compute association loss
        association_loss = compute_association_loss(associations, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposel_losses)
        losses.update({"association_loss": torch.tensor(association_loss)})

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        else:
            return self.eager_outputs(losses, detections)

    @torch.jit.unused
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses

        return detections

    def __forward_backbone(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        out = OrderedDict()

        x = self.backbone.layer1(x)
        out['0'] = x
        x = self.backbone.layer2(x)
        out['1'] = x
        x = self.backbone.layer3(x)
        out['2'] = x
        x = self.backbone.layer4(x)
        out['3'] = x

        return out, x
