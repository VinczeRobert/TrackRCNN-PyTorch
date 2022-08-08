"""
This class loads images from Mapillary that already went through both the user-defined transforms and
the ones inside MaskRCNN (normalization + resizing)
"""
import json
import os

from datasets.resized_dataset import ResizedDataset


class MapillaryInstSegResizedDataset(ResizedDataset):
    def __init__(self, root_path, all_classes=True, train=True):
        self.root_path = root_path
        if all_classes:
            self.num_classes = 37 + 1
        else:
            self.num_classes = 11 + 1
        self.train = train

        self.labels_with_instances_coco_overlap = [
            0,  # Bird
            19,  # Person
            20,  # Bicyclist
            21,  # Motorcyclist
            33,  # Bench
            38,  # Fire Hydrant
            48,  # Traffic Light
            54,  # Bus
            55,  # Car
            57,  # Motorcycle
            61  # Truck
        ]

        # read the config file
        # We are only interested in v2 and only in the labels
        with open(os.path.join(self.root_path, "config_v1.2.json")) as config_file:
            config = json.load(config_file)
        self.labels = config["labels"]
        self.labels_with_instances = [idx for idx, d in enumerate(self.labels) if d["instances"]]

        self.classes_dict = {c: idx for idx, c, in enumerate(self.labels_with_instances)}

        self.classes_coco_dict = {c: idx for idx, c, in enumerate(self.labels_with_instances_coco_overlap)}

        if self.train:
            data_path = os.path.join(self.root_path, "training")
        else:
            data_path = os.path.join(self.root_path, "validation")

        batch_folders = list(os.listdir(data_path))

        images = []
        targets = []
        for folder_name in batch_folders:
            images_root_path = os.path.join(data_path, folder_name, "images")
            targets_root_path = os.path.join(data_path, folder_name, "targets")

            images_paths = list(sorted(os.listdir(images_root_path)))
            targets_paths = list(sorted(os.listdir(targets_root_path)))

            for image_path, target_path in zip(images_paths, targets_paths):
                images.append(os.path.join(images_root_path, image_path))
                targets.append(os.path.join(targets_root_path, target_path))

        super().__init__(sorted(images), sorted(targets))
