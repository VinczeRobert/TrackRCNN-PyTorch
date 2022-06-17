from collections import OrderedDict

import torch.jit
from torch import Tensor
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from trackrcnn_kitty.models.roi_heads import RoIHeadsCustom
from trackrcnn_kitty.utils import check_for_degenerate_boxes, validate_and_build_stacked_boxes

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

COCO_DATASET_CLASSES = 91
RPN_BATCH_SIZE_PER_IMG_DEFAULT = 256


class CustomMaskRCNN(MaskRCNN):
    def __init__(self,
                 num_classes,
                 backbone,
                 pretrain_only_backbone,
                 maskrcnn_params,
                 fixed_size=(1024, 309),
                 **kwargs):
        # In some cases we create a new anchor generator to use smaller anchors (normally,
        # when the images and objects are too small)
        rpn_anchor_generator = None
        rpn_batch_size_per_image = None
        if maskrcnn_params is not None and isinstance(maskrcnn_params, dict):
            if "anchor_sizes" in maskrcnn_params and "aspect_ratios" in maskrcnn_params:
                anchor_sizes = tuple([(size,) for size in maskrcnn_params["anchor_sizes"]])
                aspect_ratios = (tuple(maskrcnn_params["aspect_ratios"]),) * len(anchor_sizes)
                rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
                rpn_batch_size_per_image = maskrcnn_params.get("rpn_batch_size_per_image",
                                                               RPN_BATCH_SIZE_PER_IMG_DEFAULT)

        # The number of classes of the COCO dataset that the backbone is pretrained one is 91
        # Also we want to use 32 ROIs per image because the images don't have many objects
        super(CustomMaskRCNN, self).__init__(backbone, COCO_DATASET_CLASSES, rpn_anchor_generator=rpn_anchor_generator,
                                             rpn_batch_size_per_image=rpn_batch_size_per_image,
                                             rpn_pre_nms_top_n_train=1000,
                                             rpn_pre_nms_top_n_test=500,
                                             rpn_post_nms_top_n_train=1000,
                                             rpn_post_nms_top_n_test=500,
                                             **kwargs)
        # Override the transform class to perform resize with fixed size the way it is described in the paper
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 800
        max_size = 1333
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, fixed_size=fixed_size)

        # Override the RoI heads to have access to custom forward method
        self.roi_heads = RoIHeadsCustom(backbone.out_channels,
                                        num_classes,
                                        self.roi_heads.mask_roi_pool,
                                        self.roi_heads.mask_head,
                                        self.roi_heads.mask_predictor)

        self.finetune(num_classes)
        if pretrain_only_backbone is False:
            self.load_weights_pretrained_on_coco()

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

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

    # TODO: Fix this method or remove it
    def load_weights_pretrained_on_coco(self):
        # state_dict = load_state_dict_from_url(model_urls["maskrcnn_resnet50_fpn_coco"], progress=True)
        # self.load_state_dict(state_dict)
        # overwrite_eps(self, 0.0)
        state_dict = torch.load("mask_rcnn_coco.pth")
        self.load_state_dict(state_dict, strict=False)
        overwrite_eps(self, 0.0)

    def finetune(self, num_classes):
        # get number of input features for the classifier
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = self.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        self.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
