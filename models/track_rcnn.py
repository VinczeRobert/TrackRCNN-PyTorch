from collections import OrderedDict

from torchvision.models.detection.image_list import ImageList
from torchvision.ops import MultiScaleRoIAlign
import torch
from models.association_head import AssociationHead
from models.layers import SepConvTemp3D
from models.mask_rcnn import CustomMaskRCNN
from models.roi_heads import RoIHeadsCustom
from utils.miscellaneous_utils import check_for_degenerate_boxes


class TrackRCNN(CustomMaskRCNN):
    def __init__(self, backbone, config, is_dataset_resized):
        super(TrackRCNN, self).__init__(backbone, config, is_dataset_resized)

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
        self.relu = torch.nn.ReLU()

        association_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of detections
        # and the number of outputs was set by the authors to 128
        resolution = self.roi_heads.box_roi_pool.output_size[0]
        representation_size = 128
        association_head = AssociationHead(backbone.out_channels * resolution ** 2, representation_size)

        # # Override the RoI heads to have access to custom forward method
        num_classes = config.num_pretrained_classes
        self.roi_heads = RoIHeadsCustom(backbone.out_channels,
                                        num_classes,
                                        self.roi_heads.mask_roi_pool,
                                        self.roi_heads.mask_head,
                                        self.roi_heads.mask_predictor,
                                        association_roi_pool,
                                        association_head)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passes")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))
        check_for_degenerate_boxes(targets)

        if self.is_dataset_resized:
            # the transforms have been already applied on the dataset
            # ImageList object needs to be created manually
            images = torch.stack(images, dim=0)
            images = ImageList(images, [(1024, 1024) for _ in range(len(images))])
        else:
            images, targets = self.transform(images, targets)

        # Run the images through the backbone (resnet50/resnet101) and fpn
        feature_dict = self.backbone(images.tensors)

        # Run the inputs through the Conv3D Layers for tracking
        for layer_name, x in feature_dict.items():
            temp = self.conv3d_temp_1(x)
            temp = self.relu(temp)
            temp = self.conv3d_temp_2(temp)
            temp = self.relu(temp)
            feature_dict[layer_name] = temp

        if isinstance(feature_dict, torch.Tensor):
            feature_dict = OrderedDict([("0", feature_dict)])

        proposals, proposel_losses = self.rpn(images, feature_dict, targets)
        detections, detector_losses = self.roi_heads(feature_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposel_losses)

        if self.training:
            return losses

        return detections
