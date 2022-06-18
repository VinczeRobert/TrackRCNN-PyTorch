from collections import OrderedDict

import numpy as np
import torch
from torch import nn, Tensor

from references.maskrcnn.utils import compute_overlaps
from trackrcnn_kitty.losses import compute_association_loss
from trackrcnn_kitty.models.layers import SepConvTemp3D
from trackrcnn_kitty.models.mask_rcnn import CustomMaskRCNN
from trackrcnn_kitty.utils import validate_and_build_stacked_boxes, check_for_degenerate_boxes


class TrackRCNN(CustomMaskRCNN):
    def __init__(self,
                 num_classes,
                 backbone,
                 pretrain_only_backbone=False,
                 maskrcnn_params=None,
                 fixed_size=None,
                 **kwargs):
        super(TrackRCNN, self).__init__(num_classes,
                                        backbone,
                                        pretrain_only_backbone,
                                        maskrcnn_params,
                                        fixed_size,
                                        **kwargs)

        backbone_output_dim = backbone.out_channels

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

        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of detections
        # and the number of outputs was set by the authors to 128
        self.association_head = nn.Linear(in_features=4, out_features=128, bias=False)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passes")

        stacked_boxes = validate_and_build_stacked_boxes(targets, self.training)

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        check_for_degenerate_boxes(targets)

        # Run the images through the backbone (resnet101) and fpn
        feature_dict = self.backbone(images.tensors)

        # Run the images through the Conv3D Layers for tracking
        features = self.conv3d_temp_1.forward(feature_dict["pool"])  # DON'T HARDCODE THIS
        features = self.relu(features)
        feature_dict[str(len(feature_dict) + 1)] = features
        features = self.conv3d_temp_2.forward(features)
        features = self.relu(features)
        feature_dict[len(feature_dict) + 1] = features

        if isinstance(feature_dict, Tensor):
            feature_dict = OrderedDict([("0", feature_dict)])

        proposals, proposel_losses = self.rpn(images, feature_dict, targets)
        detections, detector_losses = self.roi_heads(feature_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        # This for loop gets the predicted tracking ids for the region proposals
        # by checking their overlap with the ground-truth objects
        proposal_track_ids = []
        for idx, reg_prop_for_time_frame in enumerate(proposals):
            overlaps = compute_overlaps(reg_prop_for_time_frame.cpu(), targets[idx]["boxes"].cpu())
            proposal_ids = np.argmax(overlaps, axis=1)
            track_ids = torch.stack([targets[idx]["object_ids"][id] for id in proposal_ids], axis=0)
            proposal_track_ids.append(track_ids)

        # get the association vectors and compute the loss
        stacked_proposals = torch.cat(proposals, axis=0)
        associations = self.association_head(stacked_proposals)
        assocation_loss = compute_association_loss(associations.cpu(), proposal_track_ids)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposel_losses)
        losses.update({'association_loss': assocation_loss})

        if self.training:
            return losses

        return detections
