import os.path
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
from trackrcnn_kitty.utils import check_for_degenerate_boxes

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

COCO_DATASET_CLASSES = 91


class CustomMaskRCNN(MaskRCNN):
    def __init__(self,
                 num_classes,
                 backbone,
                 config):

        rpn_anchor_generator = None
        if config.maskrcnn_params is not None and isinstance(config.maskrcnn_params, dict):
            if "anchor_sizes" in config.maskrcnn_params and "aspect_ratios" in config.maskrcnn_params:
                anchor_sizes = tuple([(size,) for size in config.maskrcnn_params["anchor_sizes"]])
                aspect_ratios = (tuple(config.maskrcnn_params["aspect_ratios"]),) * len(anchor_sizes)
                rpn_anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

                config.maskrcnn_params.pop("anchor_sizes", None)
                config. maskrcnn_params.pop("aspect_ratios", None)

        # The number of classes of the COCO dataset that the backbone is pretrained one is 91
        super(CustomMaskRCNN, self).__init__(backbone, COCO_DATASET_CLASSES, rpn_anchor_generator=rpn_anchor_generator,
                                             **config.maskrcnn_params)

        if config.fixed_image_size:
            self.train_image_size = config.train_image_size
            self.test_image_size = config.test_image_size
            # Override the transform class to perform resize with fixed size the way it is described in the paper
            image_mean = [0.485, 0.456, 0.406]
            image_std = [0.229, 0.224, 0.225]
            min_size = 800
            max_size = 1333
            # If validation is done instead of training, fixed_size will be changed
            self.transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std,
                                                      fixed_size=self.train_image_size)

        if config.pytorch_pretrained_model is False:
            self.finetune(num_classes)

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

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

    def __preprocess_coco_weights(self, state_dict):
        converted_parameters = OrderedDict()

        # # First we get the weights for RPN if sizes match
        # if self.rpn.head.conv.in_channels == state_dict["rpn.conv_shared.weight"].shape[0]:
        #     converted_parameters["rpn.head.conv.weight"] = state_dict["rpn.conv_shared.weight"]
        #     converted_parameters["rpn.head.conv.bias"] = state_dict["rpn.conv_shared.bias"]
        #     converted_parameters["rpn.head.cls_logits.weight"] = state_dict["rpn.conv_class.weight"]
        #     converted_parameters["rpn.head.cls_logits.bias"] = state_dict["rpn.conv_class.bias"]
        #     converted_parameters["rpn.head.bbox_pred.weight"] = state_dict["rpn.conv_bbox.weight"]
        #     converted_parameters["rpn.head.bbox_pred.bias"] = state_dict["rpn.conv_bbox.weight"]

        # Next come the mask head parameters
        # converted_parameters["roi_heads.mask_head.mask_fcn1.weight"] = state_dict["mask.conv1.weight"]
        # converted_parameters["roi_heads.mask_head.mask_fcn1.bias"] = state_dict["mask.conv1.bias"]
        # converted_parameters["roi_heads.mask_head.mask_fcn2.weight"] = state_dict["mask.conv2.weight"]
        # converted_parameters["roi_heads.mask_head.mask_fcn2.bias"] = state_dict["mask.conv2.bias"]
        # converted_parameters["roi_heads.mask_head.mask_fcn3.weight"] = state_dict["mask.conv3.weight"]
        # converted_parameters["roi_heads.mask_head.mask_fcn3.bias"] = state_dict["mask.conv3.bias"]
        # converted_parameters["roi_heads.mask_head.mask_fcn4.weight"] = state_dict["mask.conv4.weight"]
        # converted_parameters["roi_heads.mask_head.mask_fcn4.bias"] = state_dict["mask.conv4.bias"]
        # converted_parameters["roi_heads.mask_predictor.conv5_mask.weight"] = state_dict["mask.deconv.weight"]
        # converted_parameters["roi_heads.mask_predictor.conv5_mask.bias"] = state_dict["mask.deconv.bias"]
        # converted_parameters["roi_heads.mask_predictor.mask_fcn_logits.weight"] = state_dict["mask.conv5.weight"]
        # converted_parameters["roi_heads.mask_predictor.mask_fcn_logits.bias"] = state_dict["mask.conv5.bias"]

        # Next come the box head parameters
        # converted_parameters["roi_heads.box_head.fc6.weight"] = state_dict["classifier.conv1.weight"].reshape(
        #     (1024, 256 * 7 * 7))
        # converted_parameters["roi_heads.box_head.fc6.bias"] = state_dict["classifier.conv1.bias"]
        # converted_parameters["roi_heads.box_head.fc7.weight"] = state_dict["classifier.conv2.weight"].reshape(
        #     (1024, 1024))
        # converted_parameters["roi_heads.box_head.fc7.bias"] = state_dict["classifier.conv2.bias"]
        # converted_parameters["roi_heads.box_predictor.cls_score.weight"] = state_dict["classifier.linear_class.weight"]
        # converted_parameters["roi_heads.box_predictor.cls_score.bias"] = state_dict["classifier.linear_class.bias"]
        # converted_parameters["roi_heads.box_predictor.bbox_pred.weight"] = state_dict["classifier.linear_bbox.weight"]
        # converted_parameters["roi_heads.box_predictor.bbox_pred.bias"] = state_dict["classifier.linear_bbox.bias"]

        # Next come the FPN parameters
        # converted_parameters["backbone.fpn.inner_blocks.0.weight"] = state_dict["fpn.P2_conv1.weight"]
        # converted_parameters["backbone.fpn.inner_blocks.0.bias"] = state_dict["fpn.P2_conv1.bias"]
        # converted_parameters["backbone.fpn.inner_blocks.1.weight"] = state_dict["fpn.P3_conv1.weight"]
        # converted_parameters["backbone.fpn.inner_blocks.1.bias"] = state_dict["fpn.P3_conv1.bias"]
        # converted_parameters["backbone.fpn.inner_blocks.2.weight"] = state_dict["fpn.P4_conv1.weight"]
        # converted_parameters["backbone.fpn.inner_blocks.2.bias"] = state_dict["fpn.P4_conv1.bias"]
        # converted_parameters["backbone.fpn.inner_blocks.3.weight"] = state_dict["fpn.P5_conv1.weight"]
        # converted_parameters["backbone.fpn.inner_blocks.3.bias"] = state_dict["fpn.P5_conv1.bias"]
        # converted_parameters["backbone.fpn.layer_blocks.0.weight"] = state_dict["fpn.P2_conv2.1.weight"]
        # converted_parameters["backbone.fpn.layer_blocks.0.bias"] = state_dict["fpn.P2_conv2.1.bias"]
        # converted_parameters["backbone.fpn.layer_blocks.1.weight"] = state_dict["fpn.P3_conv2.1.weight"]
        # converted_parameters["backbone.fpn.layer_blocks.1.bias"] = state_dict["fpn.P3_conv2.1.bias"]
        # converted_parameters["backbone.fpn.layer_blocks.2.weight"] = state_dict["fpn.P4_conv2.1.weight"]
        # converted_parameters["backbone.fpn.layer_blocks.2.bias"] = state_dict["fpn.P4_conv2.1.bias"]
        # converted_parameters["backbone.fpn.layer_blocks.3.weight"] = state_dict["fpn.P5_conv2.1.weight"]
        # converted_parameters["backbone.fpn.layer_blocks.3.bias"] = state_dict["fpn.P5_conv2.1.bias"]

        # Lastly we convert the backbone parameters which are by far the most
        converted_parameters["backbone.body.conv1.weight"] = state_dict["fpn.C1.0.weight"]

        # 10 layers in ResNet layer1
        converted_parameters["backbone.body.layer1.0.conv1.weight"] = state_dict["fpn.C2.0.conv1.weight"]
        converted_parameters["backbone.body.layer1.0.conv2.weight"] = state_dict["fpn.C2.0.conv2.weight"]
        converted_parameters["backbone.body.layer1.0.conv3.weight"] = state_dict["fpn.C2.0.conv3.weight"]
        converted_parameters["backbone.body.layer1.1.conv1.weight"] = state_dict["fpn.C2.1.conv1.weight"]
        converted_parameters["backbone.body.layer1.1.conv2.weight"] = state_dict["fpn.C2.1.conv2.weight"]
        converted_parameters["backbone.body.layer1.1.conv3.weight"] = state_dict["fpn.C2.1.conv3.weight"]
        converted_parameters["backbone.body.layer1.2.conv1.weight"] = state_dict["fpn.C2.2.conv1.weight"]
        converted_parameters["backbone.body.layer1.2.conv2.weight"] = state_dict["fpn.C2.2.conv2.weight"]
        converted_parameters["backbone.body.layer1.2.conv3.weight"] = state_dict["fpn.C2.2.conv3.weight"]
        converted_parameters["backbone.body.layer1.0.downsample.0.weight"] = state_dict["fpn.C2.0.downsample.0.weight"]

        # 13 layers in ResNet layer2
        converted_parameters["backbone.body.layer2.0.conv1.weight"] = state_dict["fpn.C3.0.conv1.weight"]
        converted_parameters["backbone.body.layer2.0.conv2.weight"] = state_dict["fpn.C3.0.conv2.weight"]
        converted_parameters["backbone.body.layer2.0.conv3.weight"] = state_dict["fpn.C3.0.conv3.weight"]
        converted_parameters["backbone.body.layer2.1.conv1.weight"] = state_dict["fpn.C3.1.conv1.weight"]
        converted_parameters["backbone.body.layer2.1.conv2.weight"] = state_dict["fpn.C3.1.conv2.weight"]
        converted_parameters["backbone.body.layer2.1.conv3.weight"] = state_dict["fpn.C3.1.conv3.weight"]
        converted_parameters["backbone.body.layer2.2.conv1.weight"] = state_dict["fpn.C3.2.conv1.weight"]
        converted_parameters["backbone.body.layer2.2.conv2.weight"] = state_dict["fpn.C3.2.conv2.weight"]
        converted_parameters["backbone.body.layer2.2.conv3.weight"] = state_dict["fpn.C3.2.conv3.weight"]
        converted_parameters["backbone.body.layer2.3.conv1.weight"] = state_dict["fpn.C3.3.conv1.weight"]
        converted_parameters["backbone.body.layer2.3.conv2.weight"] = state_dict["fpn.C3.3.conv2.weight"]
        converted_parameters["backbone.body.layer2.3.conv3.weight"] = state_dict["fpn.C3.3.conv3.weight"]
        converted_parameters["backbone.body.layer2.0.downsample.0.weight"] = state_dict["fpn.C3.0.downsample.0.weight"]

        # 70 layers in ResNet layer3
        converted_parameters["backbone.body.layer3.0.conv1.weight"] = state_dict["fpn.C4.0.conv1.weight"]
        converted_parameters["backbone.body.layer3.0.conv2.weight"] = state_dict["fpn.C4.0.conv2.weight"]
        converted_parameters["backbone.body.layer3.0.conv3.weight"] = state_dict["fpn.C4.0.conv3.weight"]
        converted_parameters["backbone.body.layer3.1.conv1.weight"] = state_dict["fpn.C4.1.conv1.weight"]
        converted_parameters["backbone.body.layer3.1.conv2.weight"] = state_dict["fpn.C4.1.conv2.weight"]
        converted_parameters["backbone.body.layer3.1.conv3.weight"] = state_dict["fpn.C4.1.conv3.weight"]
        converted_parameters["backbone.body.layer3.2.conv1.weight"] = state_dict["fpn.C4.2.conv1.weight"]
        converted_parameters["backbone.body.layer3.2.conv2.weight"] = state_dict["fpn.C4.2.conv2.weight"]
        converted_parameters["backbone.body.layer3.2.conv3.weight"] = state_dict["fpn.C4.2.conv3.weight"]
        converted_parameters["backbone.body.layer3.3.conv1.weight"] = state_dict["fpn.C4.3.conv1.weight"]
        converted_parameters["backbone.body.layer3.3.conv2.weight"] = state_dict["fpn.C4.3.conv2.weight"]
        converted_parameters["backbone.body.layer3.3.conv3.weight"] = state_dict["fpn.C4.3.conv3.weight"]
        converted_parameters["backbone.body.layer3.4.conv1.weight"] = state_dict["fpn.C4.4.conv1.weight"]
        converted_parameters["backbone.body.layer3.4.conv2.weight"] = state_dict["fpn.C4.4.conv2.weight"]
        converted_parameters["backbone.body.layer3.4.conv3.weight"] = state_dict["fpn.C4.4.conv3.weight"]
        converted_parameters["backbone.body.layer3.5.conv1.weight"] = state_dict["fpn.C4.5.conv1.weight"]
        converted_parameters["backbone.body.layer3.5.conv2.weight"] = state_dict["fpn.C4.5.conv2.weight"]
        converted_parameters["backbone.body.layer3.5.conv3.weight"] = state_dict["fpn.C4.5.conv3.weight"]
        converted_parameters["backbone.body.layer3.6.conv1.weight"] = state_dict["fpn.C4.6.conv1.weight"]
        converted_parameters["backbone.body.layer3.6.conv2.weight"] = state_dict["fpn.C4.6.conv2.weight"]
        converted_parameters["backbone.body.layer3.6.conv3.weight"] = state_dict["fpn.C4.6.conv3.weight"]
        converted_parameters["backbone.body.layer3.7.conv1.weight"] = state_dict["fpn.C4.7.conv1.weight"]
        converted_parameters["backbone.body.layer3.7.conv2.weight"] = state_dict["fpn.C4.7.conv2.weight"]
        converted_parameters["backbone.body.layer3.7.conv3.weight"] = state_dict["fpn.C4.7.conv3.weight"]
        converted_parameters["backbone.body.layer3.8.conv1.weight"] = state_dict["fpn.C4.8.conv1.weight"]
        converted_parameters["backbone.body.layer3.8.conv2.weight"] = state_dict["fpn.C4.8.conv2.weight"]
        converted_parameters["backbone.body.layer3.8.conv3.weight"] = state_dict["fpn.C4.8.conv3.weight"]
        converted_parameters["backbone.body.layer3.9.conv1.weight"] = state_dict["fpn.C4.9.conv1.weight"]
        converted_parameters["backbone.body.layer3.9.conv2.weight"] = state_dict["fpn.C4.9.conv2.weight"]
        converted_parameters["backbone.body.layer3.9.conv3.weight"] = state_dict["fpn.C4.9.conv3.weight"]
        converted_parameters["backbone.body.layer3.10.conv1.weight"] = state_dict["fpn.C4.10.conv1.weight"]
        converted_parameters["backbone.body.layer3.10.conv2.weight"] = state_dict["fpn.C4.10.conv2.weight"]
        converted_parameters["backbone.body.layer3.10.conv3.weight"] = state_dict["fpn.C4.10.conv3.weight"]
        converted_parameters["backbone.body.layer3.11.conv1.weight"] = state_dict["fpn.C4.11.conv1.weight"]
        converted_parameters["backbone.body.layer3.11.conv2.weight"] = state_dict["fpn.C4.11.conv2.weight"]
        converted_parameters["backbone.body.layer3.11.conv3.weight"] = state_dict["fpn.C4.11.conv3.weight"]
        converted_parameters["backbone.body.layer3.12.conv1.weight"] = state_dict["fpn.C4.12.conv1.weight"]
        converted_parameters["backbone.body.layer3.12.conv2.weight"] = state_dict["fpn.C4.12.conv2.weight"]
        converted_parameters["backbone.body.layer3.12.conv3.weight"] = state_dict["fpn.C4.12.conv3.weight"]
        converted_parameters["backbone.body.layer3.13.conv1.weight"] = state_dict["fpn.C4.13.conv1.weight"]
        converted_parameters["backbone.body.layer3.13.conv2.weight"] = state_dict["fpn.C4.13.conv2.weight"]
        converted_parameters["backbone.body.layer3.13.conv3.weight"] = state_dict["fpn.C4.13.conv3.weight"]
        converted_parameters["backbone.body.layer3.14.conv1.weight"] = state_dict["fpn.C4.14.conv1.weight"]
        converted_parameters["backbone.body.layer3.14.conv2.weight"] = state_dict["fpn.C4.14.conv2.weight"]
        converted_parameters["backbone.body.layer3.14.conv3.weight"] = state_dict["fpn.C4.14.conv3.weight"]
        converted_parameters["backbone.body.layer3.15.conv1.weight"] = state_dict["fpn.C4.15.conv1.weight"]
        converted_parameters["backbone.body.layer3.15.conv2.weight"] = state_dict["fpn.C4.15.conv2.weight"]
        converted_parameters["backbone.body.layer3.15.conv3.weight"] = state_dict["fpn.C4.15.conv3.weight"]
        converted_parameters["backbone.body.layer3.16.conv1.weight"] = state_dict["fpn.C4.16.conv1.weight"]
        converted_parameters["backbone.body.layer3.16.conv2.weight"] = state_dict["fpn.C4.16.conv2.weight"]
        converted_parameters["backbone.body.layer3.16.conv3.weight"] = state_dict["fpn.C4.16.conv3.weight"]
        converted_parameters["backbone.body.layer3.17.conv1.weight"] = state_dict["fpn.C4.17.conv1.weight"]
        converted_parameters["backbone.body.layer3.17.conv2.weight"] = state_dict["fpn.C4.17.conv2.weight"]
        converted_parameters["backbone.body.layer3.17.conv3.weight"] = state_dict["fpn.C4.17.conv3.weight"]
        converted_parameters["backbone.body.layer3.18.conv1.weight"] = state_dict["fpn.C4.18.conv1.weight"]
        converted_parameters["backbone.body.layer3.18.conv2.weight"] = state_dict["fpn.C4.18.conv2.weight"]
        converted_parameters["backbone.body.layer3.18.conv3.weight"] = state_dict["fpn.C4.18.conv3.weight"]
        converted_parameters["backbone.body.layer3.19.conv1.weight"] = state_dict["fpn.C4.19.conv1.weight"]
        converted_parameters["backbone.body.layer3.19.conv2.weight"] = state_dict["fpn.C4.19.conv2.weight"]
        converted_parameters["backbone.body.layer3.19.conv3.weight"] = state_dict["fpn.C4.19.conv3.weight"]
        converted_parameters["backbone.body.layer3.20.conv1.weight"] = state_dict["fpn.C4.20.conv1.weight"]
        converted_parameters["backbone.body.layer3.20.conv2.weight"] = state_dict["fpn.C4.20.conv2.weight"]
        converted_parameters["backbone.body.layer3.20.conv3.weight"] = state_dict["fpn.C4.20.conv3.weight"]
        converted_parameters["backbone.body.layer3.21.conv1.weight"] = state_dict["fpn.C4.21.conv1.weight"]
        converted_parameters["backbone.body.layer3.21.conv2.weight"] = state_dict["fpn.C4.21.conv2.weight"]
        converted_parameters["backbone.body.layer3.21.conv3.weight"] = state_dict["fpn.C4.21.conv3.weight"]
        converted_parameters["backbone.body.layer3.22.conv1.weight"] = state_dict["fpn.C4.22.conv1.weight"]
        converted_parameters["backbone.body.layer3.22.conv2.weight"] = state_dict["fpn.C4.22.conv2.weight"]
        converted_parameters["backbone.body.layer3.22.conv3.weight"] = state_dict["fpn.C4.22.conv3.weight"]
        converted_parameters["backbone.body.layer3.0.downsample.0.weight"] = state_dict["fpn.C4.0.downsample.0.weight"]

        # 10 layers in ResNet layer1
        converted_parameters["backbone.body.layer4.0.conv1.weight"] = state_dict["fpn.C5.0.conv1.weight"]
        converted_parameters["backbone.body.layer4.0.conv2.weight"] = state_dict["fpn.C5.0.conv2.weight"]
        converted_parameters["backbone.body.layer4.0.conv3.weight"] = state_dict["fpn.C5.0.conv3.weight"]
        converted_parameters["backbone.body.layer4.1.conv1.weight"] = state_dict["fpn.C5.1.conv1.weight"]
        converted_parameters["backbone.body.layer4.1.conv2.weight"] = state_dict["fpn.C5.1.conv2.weight"]
        converted_parameters["backbone.body.layer4.1.conv3.weight"] = state_dict["fpn.C5.1.conv3.weight"]
        converted_parameters["backbone.body.layer4.2.conv1.weight"] = state_dict["fpn.C5.2.conv1.weight"]
        converted_parameters["backbone.body.layer4.2.conv2.weight"] = state_dict["fpn.C5.2.conv2.weight"]
        converted_parameters["backbone.body.layer4.2.conv3.weight"] = state_dict["fpn.C5.2.conv3.weight"]
        converted_parameters["backbone.body.layer4.0.downsample.0.weight"] = state_dict["fpn.C5.0.downsample.0.weight"]

        # Get names of all bn layers from state_dict
        bn_layer_names = [p for p in state_dict if 'bn' in p and 'mask' not in p and 'classifier' not in p]
        bn_layer_names.extend([p for p in state_dict if 'downsample' in p])
        # Perform some name changes on them
        new_layer_names = [p.replace('fpn.', '').
                               replace('C2', 'backbone.body.layer1').
                               replace('C3', 'backbone.body.layer2').
                               replace('C4', 'backbone.body.layer3').
                               replace('C5', 'backbone.body.layer4')
                           for p in bn_layer_names]

        for idx in range(len(bn_layer_names)):
            converted_parameters[new_layer_names[idx]] = state_dict[bn_layer_names[idx]]

        # These are not needed
        converted_parameters.pop('backbone.body.layer1.0.downsample.0.bias')
        converted_parameters.pop('backbone.body.layer2.0.downsample.0.bias')
        converted_parameters.pop('backbone.body.layer3.0.downsample.0.bias')
        converted_parameters.pop('backbone.body.layer4.0.downsample.0.bias')

        converted_parameters["backbone.body.bn1.running_mean"] = state_dict["fpn.C1.1.running_mean"]
        converted_parameters["backbone.body.bn1.running_var"] = state_dict["fpn.C1.1.running_var"]
        converted_parameters["backbone.body.bn1.weight"] = state_dict["fpn.C1.1.weight"]
        converted_parameters["backbone.body.bn1.bias"] = state_dict["fpn.C1.1.bias"]

        return converted_parameters

    def load_weights(self, weights_path, preprocess_weights=False, use_resnet101=False):
        try:
            if os.path.exists(weights_path):
                state_dict = torch.load(weights_path)

                if preprocess_weights:
                    # This project includes some unofficial weights that are obtained
                    # from pretraining MaskRCNN on Coco, but they require some changes
                    # in order to load them in Pytorch
                    state_dict = self.__preprocess_coco_weights(state_dict)
                    self.load_state_dict(state_dict, strict=False)
                else:
                    self.load_state_dict(state_dict["model_state"], strict=False)

            else:
                # If we don't have a valid weights path we are going to try
                # to download one from the internet
                # Unfortunately on PyTorch pretrained weights are available only for ResNet50
                if use_resnet101 is False:
                    state_dict = load_state_dict_from_url(model_urls["maskrcnn_resnet50_fpn_coco"])
                    self.load_state_dict(state_dict, strict=False)
                    overwrite_eps(self, 0.0)

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
