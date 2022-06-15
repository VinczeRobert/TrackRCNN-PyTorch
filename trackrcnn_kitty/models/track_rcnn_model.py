import warnings
from collections import OrderedDict

import torch.jit
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from trackrcnn_kitty.losses import compute_association_loss
from trackrcnn_kitty.utils import check_for_degenerate_boxes, validate_and_build_stacked_boxes

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

COCO_DATASET_CLASSES = 91
RPN_BATCH_SIZE_PER_IMG_DEFAULT = 256


class TrackRCNN(MaskRCNN):
    def __init__(self,
                 num_classes,
                 backbone,
                 do_tracking,
                 pretrain_only_backbone,
                 maskrcnn_params,
                 fixed_size=(1024, 309),
                 **kwargs):
        # In some cases we create a new anchor generator to use smaller anchors (normally,
        # when the images and objects are too small)
        rpn_anchor_generator = None
        if "anchor_sizes" in maskrcnn_params and "aspect_ratios" in maskrcnn_params:
            anchor_sizes = tuple([(size,) for size in maskrcnn_params["anchor_sizes"]])
            aspect_ratios = (tuple(maskrcnn_params["aspect_ratios"]),) * len(anchor_sizes)
            rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        rpn_batch_size_per_image = maskrcnn_params.get("rpn_batch_size_per_image", RPN_BATCH_SIZE_PER_IMG_DEFAULT)

        # The number of classes of the COCO dataset that the backbone is pretrained one is 91
        # Also we want to use 32 ROIs per image because the images don't have many objects
        super(TrackRCNN, self).__init__(backbone, COCO_DATASET_CLASSES, rpn_anchor_generator=rpn_anchor_generator,
                                        rpn_batch_size_per_image=rpn_batch_size_per_image, **kwargs)
        # Override the transform class to perform resize with fixed size the way it is described in the paper
        image_mean = [0.485, 0.456, 0.406]
        image_std = [0.229, 0.224, 0.225]
        min_size = 800
        max_size = 1333
        self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, fixed_size=fixed_size)

        if pretrain_only_backbone is False:
            self.load_weights_pretrained_on_coco()

        self.do_tracking = do_tracking
        self.finetune(num_classes)

        backbone_output_dim = 1024  # TODO: don't hardcode this

        # # We create our two depth-wise separable Conv3D layers
        # conv3d_parameters_1 = {
        #     "in_channels": 1,
        #     "kernel_size": (3, 3, 3),  # value used by the authors of TrackRCNN for the Conv3d layers
        #     "out_channels": 1,
        #     "padding": (1, 1, 1)
        # }
        # conv3d_parameters_2 = {
        #     "in_channels": backbone_output_dim,
        #     "kernel_size": (1, 1, 1),
        #     "out_channels": backbone_output_dim,
        #     "padding": None
        # }
        # # self.conv3d_temp_1 = SepConvTemp3D(conv3d_parameters_1, conv3d_parameters_2, backbone_output_dim)
        # # self.conv3d_temp_2 = SepConvTemp3D(conv3d_parameters_1, conv3d_parameters_2, backbone_output_dim)

        # self.relu = nn.ReLU()

        # Finally we create the new association head, which is basically a fully connected layer
        # the number of inputs is equal to the number of detections
        # and the number of outputs was set by the authors to 128
        # self.association_head = nn.Linear(in_features=batch_size, out_features=128)

        # used only on torchscript mode
        self._has_warned = False

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

        # Run the images through the backbone (resnet101)
        feature_dict = self.backbone(images.tensors)

        # The next step is to send our features through our Conv3D layers
        if self.do_tracking:
            features = self.conv3d_temp_1.forward(feature_dict["pool"])  # DON'T HARDCODE THIS
            features = self.relu(features)
            feature_dict[str(len(feature_dict) + 1)] = features
            features = self.conv3d_temp_2.forward(features)
            features = self.relu(features)
            feature_dict[len(feature_dict) + 1] = features

        # feature_dict = self.fpn(feature_dict)

        if isinstance(feature_dict, Tensor):
            feature_dict = OrderedDict([("0", feature_dict)])

        proposals, proposel_losses = self.rpn(images, feature_dict, targets)
        detections, detector_losses = self.roi_heads(feature_dict, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        losses = {}
        losses.update(detector_losses)
        losses.update(proposel_losses)

        if self.do_tracking:
            # The association head gets proposals as inputs
            associations = self.association_head(stacked_boxes)

            # compute association loss
            association_loss = compute_association_loss(associations, targets)
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

    def load_weights_pretrained_on_coco(self):
        state_dict = load_state_dict_from_url(model_urls["maskrcnn_resnet50_fpn_coco"], progress=True)
        self.load_state_dict(state_dict)
        overwrite_eps(self, 0.0)
