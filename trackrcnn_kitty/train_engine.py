import os

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from references.pytorch_detection.engine import train_one_epoch, evaluate
from references.pytorch_detection.utils import MetricLogger
from trackrcnn_kitty.adnotate import adnotate
from trackrcnn_kitty.create_tracks import adnotate_first_image, find_tracks_for_one_image
from trackrcnn_kitty.creators.backbone_with_fpn_creator import BackboneWithFPNCreator
from trackrcnn_kitty.creators.data_loader_creator import get_data_loaders
from trackrcnn_kitty.models.mask_rcnn import CustomMaskRCNN
from trackrcnn_kitty.models.track_rcnn import TrackRCNN

from trackrcnn_kitty.utils import write_detection_to_file, write_gt_to_file, get_device, compute_overlaps_masks

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}

CAR_CONFIDENCE_THRESH = 0.84698
PEDESTRIAN_CONFIDENCE_THRESH = 0.93688


class TrainEngine:
    def __init__(self, config):
        self.device = get_device()
        self.config = config
        self.data_loaders, num_classes = get_data_loaders(self.config)
        self.writer = SummaryWriter("tensorboard/robert-maskrcnn-exp1")

        backbone = BackboneWithFPNCreator(trainable_backbone_layers=self.config.trainable_backbone_layers,
                                          use_resnet101=self.config.use_resnet101,
                                          pretrained_backbone=self.config.pretrained_backbone,
                                          freeze_batchnorm=self.config.freeze_batchnorm,
                                          fpn_out_channels=self.config.fpn_out_channels,
                                          add_last_layer=self.config.add_last_layer).get_instance()

        if self.config.add_associations:
            self.model = TrackRCNN(num_classes=num_classes,
                                   backbone=backbone,
                                   config=self.config,
                                   )
        else:
            self.model = CustomMaskRCNN(num_classes=num_classes,
                                        backbone=backbone,
                                        config=self.config)

        # If the backbone was no pretrained weights, we are going to try to use
        # pretrained weights for the whole model
        if self.config.pretrained_backbone is False:
            self.model.load_weights(self.config.weights_path, self.config.preprocess_weights,
                                    self.config.use_resnet101)

        if self.config.pytorch_pretrained_model:
            self.model.finetune(num_classes)

        self.model.to(self.device)

    def training(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        # optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)
        optimizer = torch.optim.SGD(params, lr=self.config.learning_rate, momentum=0.9, weight_decay=0.005)

        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.data_loaders["train"], self.device, epoch, print_freq=10)
            lr_scheduler.step()

        checkpoint = {
            "epoch": self.config.num_epochs,
            "model_state": self.model.state_dict(),
            "optim_state": optimizer.state_dict()
        }

        torch.save(checkpoint, "loss_hopefully_correct.pth")

        print("Training complete.")

    def evaluate(self):
        self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
        evaluate(self.model, self.data_loaders["test"], device=self.device)

    def training_and_evaluating(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=self.config.learning_rate,
                                    momentum=0.9, weight_decay=0.0005)
        # and a learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            self.model.transform.fixed_size = self.config.train_image_size if self.config.fixed_image_size else None
            train_one_epoch(self.model, optimizer, self.data_loaders["train"], self.device, epoch, print_freq=10)
            lr_scheduler.step()

            # By default we validate every 5 epochs
            if (epoch + 1) % self.config.epochs_to_validate == 0:
                self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
                evaluate(self.model, self.data_loaders["test"], device=self.device)

            checkpoint = {
                "epoch": self.config.num_epochs,
                "model_state": self.model.state_dict(),
                "optim_state": optimizer.state_dict()
            }

            try:
                torch.save(checkpoint, f"night{epoch}.pth")
            except OSError:
                print("Error at saving!")
                continue

        print("Training complete.")

    def evaluate_and_save_results(self, directory):
        self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
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

    def __filter_masks(self, output):
        masks_to_draw = []

        # Keep only masks that have a chance of being selected
        scores = output["scores"]
        scores = scores[scores >= CAR_CONFIDENCE_THRESH]
        masks = output["masks"]
        masks = masks[:len(scores)]

        # Because pedestrians have a higher threshold, they need an extra check
        labels = output["labels"]
        labels = labels[:len(scores)]

        for i, label in enumerate(labels):
            if label == 1 or scores[i] >= PEDESTRIAN_CONFIDENCE_THRESH:
                masks_to_draw.append(masks[i])

        if len(masks_to_draw) == 0:
            return masks_to_draw, labels

        masks_to_draw = [mask.detach().cpu().reshape((mask.shape[1], mask.shape[2])) for mask in masks_to_draw]
        masks_to_draw = torch.stack(masks_to_draw, dim=0)
        masks_to_draw = masks_to_draw > 0.5

        return masks_to_draw, labels

    def __edit_output_for_tracking(self, output, image_path):
        # artificially add the image_path to outputs to simplify logic
        output["image_path"] = image_path
        # Keep only the masks that have a chance of being selected from the current frame
        masks_jm, labels = self.__filter_masks(output)

        if len(masks_jm) == 0:
            return {}

        output["masks"] = masks_jm
        output["labels"] = labels

        return output

    def annotate_results(self):
        self.model.eval()
        obj_id = 1
        for images, targets in self.data_loaders["test"]:
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)
            for idx, output in enumerate(outputs):
                masks_to_draw, _ = self.__filter_masks(output)

                if len(masks_to_draw) == 0:
                    continue

                objects_ids = [i for i in range(obj_id, obj_id + len(masks_to_draw))]
                colours = [np.random.choice(range(256), size=3) for _ in range(len(objects_ids))]
                adnotate(targets[idx]["image_path"], masks_to_draw, objects_ids, colours)
                obj_id = obj_id + len(objects_ids)

    def annotate_results_with_tracking(self):
        self.model.eval()
        obj_id_count = 0

        is_first_batch = True
        last_output_from_batch = None
        last_target_from_batch = None
        association_dict = dict()

        for images, targets in self.data_loaders["test"]:
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)

            if is_first_batch:
                is_first_batch = False

                output = self.__edit_output_for_tracking(outputs[0], targets[0]["image_path"])

                if len(output) == 0:
                    continue

                association_dict, obj_id_count = adnotate_first_image(output, association_dict, obj_id_count)
            else:
                outputs = list(outputs)
                targets = list(targets)
                outputs.insert(0, last_output_from_batch)
                targets.insert(0, last_target_from_batch)

            for idx in range(1, len(outputs)):
                outputs[idx] = self.__edit_output_for_tracking(outputs[idx], targets[idx]["image_path"])

                if len(outputs[idx]) == 0:
                    continue

                if len(outputs[idx - 1]) == 0:
                    association_dict, obj_id_count = adnotate_first_image(outputs[idx], association_dict, 0)
                    continue

                # Continue tracks for current image
                association_dict, obj_id_count = find_tracks_for_one_image(outputs[idx], outputs[idx - 1],
                                                                           association_dict,
                                                                           obj_id_count)
            else:
                last_output_from_batch = outputs[-1]
                last_target_from_batch = targets[-1]

    def calculate_metrics(self):
        self.model.eval()
        # Initializing values needed
        soft_TP = 0
        TP = 0
        IDS = 0
        FP = 0
        M = 0

        for images, targets in self.data_loaders["test"]:
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)
            for idx, output in enumerate(outputs):
                masks_to_keep = []

                # Keep only masks that have a chance of being selected
                scores = output["scores"]
                scores = scores[scores >= CAR_CONFIDENCE_THRESH]
                masks = output["masks"]
                masks = masks[:len(scores)]

                # Because pedestrians have a higher threshold, they need an extra check
                labels = output["labels"]
                labels = labels[:len(scores)]

                for i, label in enumerate(labels):
                    if label == 1 or scores[i] >= PEDESTRIAN_CONFIDENCE_THRESH:
                        masks_to_keep.append(masks[i])

                if len(masks_to_keep) == 0:
                    continue

                masks_to_keep = [mask.detach().cpu().reshape((mask.shape[1], mask.shape[2])) for mask in masks_to_keep]
                masks_to_keep = torch.stack(masks_to_keep, dim=0)
                masks_to_keep = masks_to_keep > 0.5

                # Get IoU scores
                mask_overlaps = compute_overlaps_masks(targets[idx]["masks"].numpy(), masks_to_keep.numpy())

                max_ious = np.amax(mask_overlaps, axis=1)
                max_ious = max_ious[max_ious > 0.5]
                #
                # try:
                #     indexes = [idx for idx in np.unique(np.argmax(mask_overlaps, axis=1)) if max_ious[idx] > 0.5]
                # except IndexError:
                #     indexes = [idx] if max_ious[idx] > 0.5 else []

                # Update values from formula
                M = M + len(targets[idx]["masks"])
                TP = TP + len(max_ious)
                # FP = FP + (mask_overlaps.shape[0] - len(max_ious))
                soft_TP = soft_TP + np.sum(max_ious)

        MOTSA = (TP - FP - IDS) / M
        sMOTSA = (soft_TP - FP - IDS) / M
        print("MOTSA score is: " + str(MOTSA))
        print("sMOTSA score is: " + str(sMOTSA))
