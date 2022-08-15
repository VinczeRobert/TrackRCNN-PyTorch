import os

import cv2 as cv
import numpy as np
import torch

from creators.backbone_with_fpn_creator import BackboneWithFPNCreator
from datasets.data_loader_creator import get_data_loaders
from models.mask_rcnn import CustomMaskRCNN
from models.track_rcnn import TrackRCNN
from references.pytorch_detection.engine import train_one_epoch, evaluate
from references.pytorch_detection.utils import MetricLogger
from utils.io_utils import write_gt_to_file, write_detection_to_file, save_tracking_prediction_for_batch, \
    save_detections_for_batch, load_tracking_predictions
from utils.metrics_utils import compute_overlaps_masks
from utils.miscellaneous_utils import get_device
from utils.tracking_utils import track_sequence, visualize_tracks, make_tracks_disjoint

model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
}


class TrackRCNNPyTorchEngine:
    def __init__(self, config):
        self.device = get_device()
        self.config = config
        self.data_loaders, num_classes, is_dataset_resized = get_data_loaders(self.config)

        backbone = BackboneWithFPNCreator(trainable_backbone_layers=self.config.trainable_backbone_layers,
                                          use_resnet101=self.config.use_resnet101,
                                          pretrain_only_backbone=self.config.pretrain_only_backbone,
                                          freeze_batchnorm=self.config.freeze_batchnorm,
                                          fpn_out_channels=self.config.fpn_out_channels,
                                          add_last_layer=self.config.add_last_layer).get_instance()

        if self.config.add_associations:
            self.model = TrackRCNN(backbone=backbone, config=self.config, is_dataset_resized=is_dataset_resized)
        else:
            self.model = CustomMaskRCNN(backbone=backbone, config=self.config, is_dataset_resized=is_dataset_resized)

        # If the backbone was no pretrained weights, we are going to try to use
        # pretrained weights for the whole model
        if self.config.pretrain_only_backbone is False:
            self.model.load_weights(self.config.weights_path, self.config.load_weights, self.config.use_resnet101)

        if num_classes != self.config.num_pretrained_classes:
            self.model.finetune(num_classes)

        self.model.to(self.device)

    def train(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config.optimizer_parameters["name"] == "sgd":
            optimizer = torch.optim.SGD(params, lr=self.config.learning_rate,
                                        momentum=self.config.optimizer_parameters.get("momentum", 0.9),
                                        weight_decay=self.config.optimizer_parameters.get("weight_decay", 0.005))
        else:
            optimizer = torch.optim.Adam(params, lr=self.config.learning_rate)

        lr_scheduler = None
        if self.config.lr_scheduler_step_size is not None and self.config.lr_scheduler_gamma is not None:
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                           step_size=self.config.lr_scheduler_step_size,
                                                           gamma=self.config.lr_scheduler_gamma)

        for epoch in range(self.config.num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.data_loaders["train"], self.device, epoch, print_freq=10)

            if lr_scheduler:
                lr_scheduler.step()

            if self.config.epochs_to_save_weights and ((epoch + 1) % self.config.epochs_to_save_weights == 0):
                checkpoint = {
                    "epoch": self.config.num_epochs,
                    "model_state": self.model.state_dict(),
                    "optim_state": optimizer.state_dict()
                }

                try:
                    torch.save(checkpoint, f"{self.config.saved_weights_name}_epoch={epoch}.pth")
                except OSError:
                    print("Error at saving!")
                    continue

        print("Training complete.")

    def evaluate(self):
        self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
        evaluate(self.model, self.data_loaders["test"], device=self.device)

    # TODO: currently this is not used because it gives an error right before the second validation. Reasons not
    #  known yet. It should be fixed or removed.
    def train_and_evaluate(self):
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

            if self.config.epochs_to_save_weights and ((epoch + 1) % self.config.epochs_to_save_weights == 0):
                checkpoint = {
                    "epoch": self.config.num_epochs,
                    "model_state": self.model.state_dict(),
                    "optim_state": optimizer.state_dict()
                }

                try:
                    torch.save(checkpoint, f"{self.config.saved_weights_path}_epoch={epoch}.pth")
                except OSError:
                    print("Error at saving!")
                    continue

        print("Training complete.")

    def save_bounding_box_results(self, dataset_path):
        # If the evaluate method of this class (which uses pycocotools) is too slow, an
        # alternative for mAP on bounding boxes is: https://github.com/Cartucho/mAP.
        # In order to use this repo, bounding boxes need to be saved in a certain format.

        directory = os.path.join("predictions", os.path.basename(dataset_path))
        self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
        self.model.eval()
        metric_logger = MetricLogger(delimiter=" ")
        current_index = 0
        for images, targets, in metric_logger.log_every(self.data_loaders["test"], 100, "Test:"):
            # torch.cuda.empty_cache()
            images = list(img.to(self.device) for img in images)
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

    def forward_predictions_for_tracking(self, sequence_number):
        results_dir_path = os.path.join("predictions", sequence_number)
        torch.cuda.empty_cache()
        self.model.transform.fixed_size = self.config.test_image_size if self.config.fixed_image_size else None
        self.model.eval()
        os.makedirs(results_dir_path, exist_ok=True)
        results_path = os.path.join(results_dir_path, os.path.basename(results_dir_path) + ".txt")
        if os.path.exists(results_path):
            os.remove(results_path)
        metric_logger = MetricLogger(delimiter=" ")
        track = 0
        for images, _, in metric_logger.log_every(self.data_loaders["test"], 20, "Test:"):
            images = list(img.to(self.device) for img in images)
            outputs = self.model(images)
            track = save_tracking_prediction_for_batch(outputs, results_path, track)

    def annotate_results_without_tracking(self):
        self.model.eval()
        obj_id = 1

        out_folder = os.path.join("annotations_created", self.config.sequence_number)
        os.makedirs(out_folder, exist_ok=True)

        for images, targets in self.data_loaders["test"]:
            images = list(img.to(self.device) for img in images if img is not None)
            targets = list(target for target in targets if target is not None)
            if len(images) == 0:
                continue
            outputs = self.model(images)
            obj_id = save_detections_for_batch(outputs, targets, self.config.confidence_threshold_car,
                                               self.config.confidence_threshold_pedestrian, out_folder, obj_id)

    def annotate_results_with_tracking(self, sequence_number):
        detections_import_path = os.path.join("predictions", sequence_number, sequence_number + ".txt")
        boxes, scores, association_vectors, classes, masks = load_tracking_predictions(detections_import_path)

        while len(self.data_loaders["test"]) > len(boxes):
            boxes.append([])
            scores.append([])
            association_vectors.append([])
            classes.append([])
            masks.append([])

        # transform into numpy arrays
        for t in range(len(boxes)):
            if len(boxes[t]) > 0:
                boxes[t] = np.vstack(boxes[t])
                scores[t] = np.array(scores[t])
                classes[t] = np.array(classes[t])
                association_vectors[t] = np.vstack(association_vectors[t])

        tracker_options = {
            "confidence_threshold_car": self.config.confidence_threshold_car,
            "reid_weight_car": self.config.reid_weight_car,
            "association_threshold_car": self.config.association_threshold_car,
            "keep_alive_car": self.config.keep_alive_car,
            "reid_euclidean_offset_car": self.config.reid_euclidean_offset_car,
            "reid_euclidean_scale_car": self.config.reid_euclidean_scale_car,
            "confidence_threshold_pedestrian": self.config.confidence_threshold_pedestrian,
            "reid_weight_pedestrian": self.config.reid_weight_pedestrian,
            "association_threshold_pedestrian": self.config.association_threshold_pedestrian,
            "keep_alive_pedestrian": self.config.keep_alive_pedestrian,
            "reid_euclidean_offset_pedestrian": self.config.reid_euclidean_offset_pedestrian,
            "reid_euclidean_scale_pedestrian": self.config.reid_euclidean_scale_pedestrian
        }

        tracks = track_sequence(tracker_options, boxes, scores, association_vectors, classes, masks)

        # This method solves the issue of overlapping pixels.
        # If a pixel is covered by more than one object mask, it will be assigned to the one with higher score
        tracks = make_tracks_disjoint(tracks)

        # Gather images
        sequence_root_path = os.path.join(self.config.dataset_path, "images", "validation", self.config.sequence_number)
        base_paths = os.listdir(sequence_root_path)
        image_paths = [os.path.join(sequence_root_path, base_path) for base_path in base_paths]
        all_images = []
        for path in image_paths:
            image = cv.imread(path, -1)
            all_images.append(image)
        visualize_tracks(self.config.sequence_number, tracks, all_images)

    # TODO: This is currently incorrect. Needs to be fixed in the end.
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
                scores = scores[scores >= self.config.confidence_threshold_car]
                masks = output["masks"]
                masks = masks[:len(scores)]

                # Because pedestrians have a higher threshold, they need an extra check
                labels = output["labels"]
                labels = labels[:len(scores)]

                for i, label in enumerate(labels):
                    if label == 1 or scores[i] >= self.config.confidence_threshold_pedestrian:
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
