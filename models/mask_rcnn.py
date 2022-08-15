import os.path
import sys
from collections import OrderedDict

import torch.jit
from torch import Tensor
from torch.hub import load_state_dict_from_url
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection._utils import overwrite_eps
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.image_list import ImageList
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform

from utils.miscellaneous_utils import check_for_degenerate_boxes

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


class CustomMaskRCNN(MaskRCNN):
    def __init__(self, backbone, config, is_dataset_resized):
        self.is_dataset_resized = is_dataset_resized

        rpn_anchor_generator = None
        if config.maskrcnn_params is not None and isinstance(config.maskrcnn_params, dict):
            if "anchor_sizes" in config.maskrcnn_params and "aspect_ratios" in config.maskrcnn_params:
                anchor_sizes = tuple([(size,) for size in config.maskrcnn_params["anchor_sizes"]])
                aspect_ratios = (tuple(config.maskrcnn_params["aspect_ratios"]),) * len(anchor_sizes)
                rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

                config.maskrcnn_params.pop("anchor_sizes", None)
                config. maskrcnn_params.pop("aspect_ratios", None)

        # The number of classes of the COCO dataset that the backbone is pretrained one is 91
        super(CustomMaskRCNN, self).__init__(backbone,
                                             config.num_pretrained_classes,
                                             rpn_anchor_generator=rpn_anchor_generator,
                                             **config.maskrcnn_params)

        if config.fixed_image_size:
            self.train_image_size = config.train_image_size
            self.test_image_size = config.test_image_size
            # Override the transform class to perform resize with fixed size the way it is described in the paper
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            min_size = 800
            max_size = 1333
            # If validation is done instead of training, fixed_size will be changed later
            self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std,
                                                      fixed_size=self.train_image_size)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

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
            images = ImageList(images, [self.transform.fixed_size for _ in range(len(images))])
        else:
            images, targets = self.transform(images, targets)

        # Run the images through the backbone (resnet50/resnet101) and fpn
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

    def load_weights(self, weights_path, load_weights, use_resnet101=False):
        try:
            if os.path.exists(weights_path) and load_weights:
                try:
                    state_dict = torch.load(weights_path)
                except RuntimeError:
                    # Probably because of attempting to deserialize object on A CUDA device but
                    # torch.cuda.is_available() is False.
                    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))

                self.load_state_dict(state_dict["model_state"], strict=False)
                print('Successfully loaded custom weights!')

            else:
                # If we don't have a valid weights path we are going to try
                # to download one from the internet
                # Unfortunately on PyTorch pretrained weights on MaskRCNN are available only for ResNet50 backbone
                if use_resnet101 is False:
                    state_dict = load_state_dict_from_url(model_urls["maskrcnn_resnet50_fpn_coco"])
                    self.load_state_dict(state_dict, strict=False)
                    overwrite_eps(self, 0.0)
                    print('Successfully loaded pytorch pretrained weights from COCO Resnet50!')
                else:
                    print('Maskrcnn with Resnet101 does not have pretrained weights. Stopping the program!')
                    sys.exit(-1)

        except RuntimeError as e:
            print("There is no valid set of weights that can be loaded."
                  "Training will start with newly initialized weights!")
            print(e)

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
