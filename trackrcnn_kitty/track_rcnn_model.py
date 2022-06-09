import warnings
from collections import OrderedDict

import numpy as np
import torch.jit
import torchvision
from torch import nn, Tensor, where, split, reshape, cat, squeeze, stack, eq, masked_select, div, logical_not, max, \
    min, tensor, sum
from torch.nn import Conv3d
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import misc as misc_nn_ops, MultiScaleRoIAlign, FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

from trackrcnn_kitty.utils import as_numpy


class TrackRCNN(nn.Module):
    def __init__(self, num_classes, device):
        super(TrackRCNN, self).__init__()
        # Create a Transform object which will be responsible on applying transformations on the image
        # These parameters are taken from Pytorch code
        min_size = 800
        max_size = 1333
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

        # We create the backbone: we'll use a Resnet101
        # that has been trained on the COCO dataset
        resnet101 = torchvision.models.resnet101(pretrained=True, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
        self.backbone = resnet101
        # For now we are not going to use BackbonewithFPN, but we might later

        # Initialize Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork(
            [256, 512, 1024, 2048, 2048, 2048],
            256,
            LastLevelMaxPool()
        )

        # We create our two depth-wise separable Conv3D layers

        self.conv_temp1_depth_wise = [Conv3d(in_channels=1, kernel_size=(3, 3, 3), out_channels=1, padding=(1, 1, 1),
                                             device=device)] * 2048
        filter_initializer = np.zeros([1, 1, 3, 3, 3], dtype=np.float32)
        filter_size = [3, 3, 3]
        filter_initializer[0, 0, :, int(filter_size[1] / 2.0), int(filter_size[2] / 2.0)] = 1.0 / filter_size[0]
        for layer in self.conv_temp1_depth_wise:
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.as_tensor(filter_initializer, device=device))

        point_wise_weights = np.zeros([2048, 2048, 1, 1, 1], dtype=np.float32)
        for i in range(2048):
            point_wise_weights[i, i, :, :, :] = 1.0
        self.conv_temp1_point_wise = Conv3d(in_channels=2048, kernel_size=(1, 1, 1), out_channels=2048, device=device)
        with torch.no_grad():
            self.conv_temp1_point_wise.weight = nn.Parameter(torch.as_tensor(point_wise_weights, device=device))

        self.conv_temp2_depth_wise = [Conv3d(in_channels=1, kernel_size=(3, 3, 3), out_channels=1, padding=(1, 1, 1),
                                             device=device)] * 2048
        for layer in self.conv_temp2_depth_wise:
            with torch.no_grad():
                layer.weight = nn.Parameter(torch.as_tensor(filter_initializer, device=device))
        self.conv_temp2_point_wise = Conv3d(in_channels=2048, kernel_size=(1, 1, 1), out_channels=2048, device=device)
        with torch.no_grad():
            self.conv_temp1_point_wise.weight = nn.Parameter(torch.as_tensor(point_wise_weights, device=device))

        self.relu = nn.ReLU()

        # Create the region proposal network
        anchor_sizes = ((32,), (64,), (128,), (256,), (512,), (512,), (512,))
        aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

        rpn_head = RPNHead(256, rpn_anchor_generator.num_anchors_per_location()[0])
        rpn_fg_iou_thresh = 0.7
        rpn_bg_iou_thresh = 0.3
        rpn_batch_size_per_image = 256
        rpn_positive_fraction = 0.5
        rpn_pre_nms_top_n = {'training': 2000, 'testing': 1000}
        rpn_post_nms_top_n = {'training': 2000, 'testing': 1000}
        rpn_nms_thresh = 0.7
        rpn_score_thresh = 0.0
        self.rpn = RegionProposalNetwork(
            rpn_anchor_generator,
            rpn_head,
            rpn_fg_iou_thresh,
            rpn_bg_iou_thresh,
            rpn_batch_size_per_image,
            rpn_positive_fraction,
            rpn_pre_nms_top_n,
            rpn_post_nms_top_n,
            rpn_nms_thresh,
            score_thresh=rpn_score_thresh

        )

        # Create RoI heads
        mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)

        mask_layers = (256, 256, 256, 256)
        mask_dilation = 1
        mask_head = MaskRCNNHeads(256, mask_layers, mask_dilation)

        mask_predictor_in_channels = 256
        mask_dim_reduced = 256
        mask_predictor = MaskRCNNPredictor(mask_predictor_in_channels, mask_dim_reduced, num_classes)

        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(256 * resolution ** 2, representation_size)

        representation_size = 1024
        box_predictor = FastRCNNPredictor(representation_size, num_classes)

        box_fg_iou_thresh = 0.5
        box_bg_iou_thresh = 0.5
        box_batch_size_per_image = 512
        box_positive_fraction = 0.25
        bbox_reg_weights = None
        box_score_thresh = 0.05
        box_nms_thresh = 0.5
        box_detections_per_img = 100
        self.roi_heads = RoIHeads(
            box_roi_pool,
            box_head,
            box_predictor,
            box_fg_iou_thresh,
            box_bg_iou_thresh,
            box_batch_size_per_image,
            box_positive_fraction,
            bbox_reg_weights,
            box_score_thresh,
            box_nms_thresh,
            box_detections_per_img,
        )

        self.roi_heads.mask_roi_pool = mask_roi_pool
        self.roi_heads.mask_head = mask_head
        self.roi_heads.mask_predictor = mask_predictor

        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of regions proposed by the RPN
        # and the number of outputs was set by the authors to 128
        self.association_head = nn.Linear(in_features=4, out_features=128)

        # used only on torchscript mode
        self._has_warned = False

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        # needed for the association head
        stacked_boxes = []

        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                            raise ValueError(
                                f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
                else:
                    raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

                stacked_boxes.extend(boxes)
        stacked_boxes = stack(stacked_boxes)
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)
        self.__check_for_degenerate_boxes(targets)

        # Run the images through the backbone (resnet101)
        feature_dict, features = self.__forward_backbone(images.tensors)

        # The next step is to send our features through our Conv3D layers
        features = self.__forward_separable_conv3d(features, self.conv_temp1_depth_wise, self.conv_temp1_point_wise)
        feature_dict['4'] = features
        features = self.__forward_separable_conv3d(features, self.conv_temp2_depth_wise, self.conv_temp2_point_wise)
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
        association_loss = self.__compute_association_loss(associations, targets)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposel_losses)

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

    # For a valid box the height and width have to be positive numbers
    @staticmethod
    def __check_for_degenerate_boxes(targets):
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

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

    def __forward_separable_conv3d(self, features, conv_temp_depth_wise_layers, conv_temp_point_wise_layer):
        no_features = features.shape[1]

        # Introduce a fake "batch dimension", previous dimension becomes time
        # -1 means to preserve the current dim
        curr_features = features.expand(1, -1, -1, -1, -1)
        curr_features = reshape(curr_features, (curr_features.shape[0], curr_features.shape[2], curr_features.shape[1],
                                                curr_features.shape[3], curr_features.shape[4]))

        # In the following part we will do a depthwise convolution
        curr_features = list(split(curr_features, 1, dim=1))

        for channel_no in range(no_features):
            curr_features[channel_no] = conv_temp_depth_wise_layers[channel_no](curr_features[channel_no])

        # Stack the channels together
        curr_features = cat(curr_features, dim=1)

        # Now we do the pointwise convolution
        curr_features = conv_temp_point_wise_layer(curr_features)

        curr_features = self.relu(curr_features)

        # Remove the fake dimension and switch no_channels and batch_size back
        curr_features = reshape(curr_features, (curr_features.shape[0], curr_features.shape[2], curr_features.shape[1],
                                                curr_features.shape[3], curr_features.shape[4]))
        curr_features = squeeze(curr_features, dim=0)

        return curr_features

    @staticmethod
    def __compute_associations_loss_for_detection(detection_id, detection_distances, detection_ids_axis_0,
                                                  detection_ids_axis_1):
        detection_id_mask_axis_0 = eq(detection_ids_axis_0, detection_id)
        detection_id_mask_axis_1 = eq(detection_ids_axis_1, detection_id)
        distances_for_current_detection = detection_distances * detection_id_mask_axis_0.int().float()
        all_detections_class_ids = div(detection_ids_axis_1, 1000, rounding_mode='floor')
        current_detection_class_id = div(detection_id, 1000, rounding_mode='floor')
        detection_id_by_class_mask = eq(all_detections_class_ids, current_detection_class_id)
        distances_for_current_detection = masked_select(distances_for_current_detection, detection_id_by_class_mask)
        detection_id_mask_axis_1 = masked_select(detection_id_mask_axis_1, detection_id_by_class_mask)

        same_ids = masked_select(distances_for_current_detection, detection_id_mask_axis_1)
        different_ids = masked_select(distances_for_current_detection, logical_not(detection_id_mask_axis_1))

        hard_pos = max(same_ids)
        hard_neg = min(different_ids)

        if len(same_ids) > 0 and len(different_ids) > 0:

            margin = 0.2
            hard_pos = np.asscalar(as_numpy(hard_pos))
            hard_neg = np.asscalar(as_numpy(hard_neg))
            triplet_loss = margin + hard_pos - hard_neg
            loss = triplet_loss if triplet_loss > 0 else 0
            # normalization = len(loss)

            return loss, 1
        else:
            return 0, 1

    @staticmethod
    def __compute_association_loss(associations, targets):
        # Create a tensor of dim (D), D being the number of detections
        all_detection_ids = []
        for target in targets:
            all_detection_ids.append(target['object_ids'])
        all_detection_ids = cat(all_detection_ids, dim=0)

        # associations is a tensor of dim (D, 128), D being the number of detections
        # compute euclidean distance between every pair of detections from this batch
        detection_distances = torch.cdist(associations, associations)

        unique_detection_ids = torch.unique(all_detection_ids)

        loss = 0
        normalization = 0
        for detection_id in unique_detection_ids:
            loss_per_id, normalization_per_id = TrackRCNN.__compute_associations_loss_for_detection(detection_id,
                                                                                                    detection_distances,
                                                                                                    all_detection_ids,
                                                                                                    all_detection_ids)
            loss += loss_per_id
            normalization += normalization_per_id

        loss = loss / normalization

        return loss
