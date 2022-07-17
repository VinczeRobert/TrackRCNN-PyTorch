import numpy as np
import torch
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.ops import MultiScaleRoIAlign

from trackrcnn_kitty.losses import AssociationLoss
from trackrcnn_kitty.models.association_head import AssociationHead
from trackrcnn_kitty.utils import compute_overlaps


class RoIHeadsCustom(RoIHeads):

    def __init__(self,
                 out_channels,
                 num_classes,
                 # Mask
                 mask_roi_pool=None,
                 mask_head=None,
                 mask_predictor=None,
                 keypoint_roi_pool=None,
                 keypoint_head=None,
                 keypoint_predictor=None,
                 ):

        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)

        resolution = box_roi_pool.output_size[0]
        representation_size = 1024
        box_head = TwoMLPHead(
            out_channels * resolution ** 2,
            representation_size
        )

        representation_size = 1024
        box_predictor = FastRCNNPredictor(
            representation_size,
            num_classes)

        fg_iou_thresh = 0.5
        bg_iou_thresh = 0.5
        batch_size_per_image = 512
        positive_fraction = 0.25
        bbox_reg_weights = None
        score_thresh = 0.05
        nms_thresh = 0.5
        detections_per_img = 100

        super(RoIHeadsCustom, self).__init__(box_roi_pool,
                                             box_head,
                                             box_predictor,
                                             fg_iou_thresh, bg_iou_thresh,
                                             batch_size_per_image, positive_fraction,
                                             bbox_reg_weights,
                                             score_thresh,
                                             nms_thresh,
                                             detections_per_img,
                                             mask_roi_pool,
                                             mask_head,
                                             mask_predictor,
                                             keypoint_roi_pool,
                                             keypoint_head,
                                             keypoint_predictor)

        self.mask_roi_pool = mask_roi_pool
        self.mask_head = mask_head
        self.mask_predictor = mask_predictor

        self.association_roi_pool = MultiScaleRoIAlign(
            featmap_names=['0', '1', '2', '3'],
            output_size=7,
            sampling_ratio=2)
        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of detections
        # and the number of outputs was set by the authors to 128
        resolution = box_roi_pool.output_size[0]
        representation_size = 128
        self.association_head = AssociationHead(out_channels * resolution ** 2, representation_size)
        self.association_loss = AssociationLoss()

    def forward(self, features,
                proposals,
                image_shapes,
                targets=None
                ):
        detections, detector_losses = super(RoIHeadsCustom, self).forward(features, proposals, image_shapes, targets)

        association_features = self.association_roi_pool(features, proposals, image_shapes)
        association_features = self.association_head(association_features)

        # This for loop gets the predicted tracking ids for the region proposals
        # by checking their overlap with the ground-truth objects
        proposal_track_ids = []
        for idx, reg_prop_for_time_frame in enumerate(proposals):
            overlaps = compute_overlaps(reg_prop_for_time_frame.cpu(), targets[idx]["boxes"].cpu())
            proposal_ids = np.argmax(overlaps, axis=1)
            track_ids = torch.stack([targets[idx]["obj_ids"][id] for id in proposal_ids], axis=0)
            proposal_track_ids.append(track_ids)

        # Compute the association loss
        association_loss = self.association_loss(association_features.cpu(), proposal_track_ids)
        association_loss.requires_grad = True
        detector_losses.update({"loss_association": association_loss})
        return detections, detector_losses
