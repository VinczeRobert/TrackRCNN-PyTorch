import os

import torch

from references.pytorch_detection.engine import train_one_epoch, evaluate
from references.pytorch_detection.utils import MetricLogger
from trackrcnn_kitty.creators.backbone_with_fpn_creator import BackboneWithFPNCreator
from trackrcnn_kitty.creators.data_loader_creator import get_data_loaders
from trackrcnn_kitty.datasets.dataset_factory import get_dataset
from trackrcnn_kitty.datasets.transforms import get_transforms
from trackrcnn_kitty.models.mask_rcnn import CustomMaskRCNN
from trackrcnn_kitty.models.track_rcnn import TrackRCNN

from trackrcnn_kitty.utils import write_detection_to_file, write_gt_to_file

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


class TrainEngine:
    def __init__(self, config):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.config = config

        train = True if self.config.task in ["train", "train+val"] else False
        transforms = get_transforms(self.config.transforms_list, train)

        self.dataset = get_dataset(config, transforms, train)
        self.data_loaders = get_data_loaders(self.dataset, self.config)

        backbone = BackboneWithFPNCreator(trainable_backbone_layers=self.config.trainable_backbone_layers,
                                          use_resnet101=self.config.use_resnet101,
                                          pretrained_backbone=self.config.pretrained_backbone,
                                          freeze_batchnorm=self.config.freeze_batchnorm).get_instance()

        if self.config.add_associations:
            self.model = TrackRCNN(num_classes=self.dataset.num_classes,
                                   backbone=backbone,
                                   pretrained_backbone=self.config.pretrained_backbone,
                                   maskrcnn_params=self.config.maskrcnn_params)
        else:
            self.model = CustomMaskRCNN(num_classes=self.dataset.num_classes,
                                        backbone=backbone,
                                        pretrained_backbone=self.config.pretrained_backbone,
                                        maskrcnn_params=self.config.maskrcnn_params)

            # If the backbone was no pretrained weights, we are going to try to use
            # pretrained weights for the whole model
            if self.config.pretrained_backbone is False:
                self.model.load_weights(self.config.weights_path, self.config.preprocess_weights,
                                        self.config.use_resnet101)

        # self.model.finetune(3)

        self.model.to(self.device)

    def training(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer = torch.optim.SGD(params, lr=self.config.learning_rate,
        #                             weight_decay=self.config.weight_decay,
        #                             momentum=self.config.momentum)
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.data_loaders["train"], self.device, epoch, print_freq=10)
            # evaluate(self.model, self.test_data_loaders["train"], device=self.device)

        checkpoint = {
            "epoch": self.config.num_epochs,
            "model_state": self.model.state_dict(),
            "optim_state": optimizer.state_dict()
        }

        torch.save(checkpoint, "experiment4.pth")

        print("Training complete.")

    def evaluate(self):
        evaluate(self.model, self.data_loaders["test"], device=self.device)

    def training_and_evaluating(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.data_loaders["train"], self.device, epoch, print_freq=10)
            evaluate(self.model, self.data_loaders["test"], device=self.device)

        checkpoint = {
            "epoch": self.config.num_epochs,
            "model_state": self.model.state_dict(),
            "optim_state": optimizer.state_dict()
        }

        torch.save(checkpoint, self.config.weights_path)

        print("Training complete.")

    def evaluate_and_save_results(self, directory):
        self.model.load_state_dict(torch.load(self.config.weights_path)["model_state"])
        self.model.eval()
        metric_logger = MetricLogger(delimiter=" ")
        current_index = 0
        for images, targets, in metric_logger.log_every(self.data_loaders["test"], 100, "Test:"):
            torch.cuda.empty_cache()
            images = list(img.to(self.device) for img in images)

            torch.cuda.synchronize()
            outputs = self.model(images)

            current_inner_index = current_index
            # Store ground truth detections
            for target in targets:
                gt_file_name = os.path.join(directory, "ground_truth", f"image_{current_inner_index:06}.txt")
                write_gt_to_file(target, gt_file_name)
                current_inner_index = current_inner_index + 1

            # Store predicted detections
            for output in outputs:
                dets_file_name = os.path.join(directory, "detected", f"image_{current_index:06}.txt")
                write_detection_to_file(output, dets_file_name)
                # masks_file_name = os.path.join(directory, "masks", f"{current_index:06}.txt")
                # write_segmentation_mask_to_file(output["masks"], masks_file_name)
                print(current_index)
                current_index = current_index + 1
