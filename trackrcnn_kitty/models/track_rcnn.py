from collections import OrderedDict

from torch import nn, Tensor

from trackrcnn_kitty.models.layers import SepConvTemp3D
from trackrcnn_kitty.models.mask_rcnn import CustomMaskRCNN
from trackrcnn_kitty.models.roi_heads import RoIHeadsCustom
from trackrcnn_kitty.utils import check_for_degenerate_boxes


class TrackRCNN(CustomMaskRCNN):
    def __init__(self,
                 num_classes,
                 backbone,
                 config):
        super(TrackRCNN, self).__init__(num_classes,
                                        backbone,
                                        config)

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

        if config.pytorch_pretrained_model:
            num_classes = 91

        # # Override the RoI heads to have access to custom forward method
        self.roi_heads = RoIHeadsCustom(backbone.out_channels,
                                        num_classes,
                                        self.roi_heads.mask_roi_pool,
                                        self.roi_heads.mask_head,
                                        self.roi_heads.mask_predictor)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passes")

        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        check_for_degenerate_boxes(targets)

        # Run the inputs through the backbone (resnet101) and fpn
        feature_dict = self.backbone(images.tensors)

        # Run the inputs through the Conv3D Layers for tracking
        for layer_name, x in feature_dict.items():
            temp = self.conv3d_temp_1(x)
            temp = self.relu(temp)
            temp = self.conv3d_temp_2(temp)
            temp = self.relu(temp)
            feature_dict[layer_name] = temp

        if isinstance(feature_dict, Tensor):
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
