"""
This script does filtering on the Mapillary Vistas training or validation dataset based on mask sizes and classes.
Parameters in lines 113-127 need to be set by the user.
"""

import os
import json
import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset


class MapillaryToBeFilteredDataset(Dataset):
    def __init__(self, root_path, save_path, mask_threshold=0, all_classes=True, train=True):
        """
        root_path: path to the dataset that we want to preprocess
        save_path: path where the preprocessed dataset will be saved at
        mask_threshold: minimum area of the detection masks that will be kept. Masks below this value won't be kept.
        The value is 0 by default.
        all_classes: if is it True, then all 37+1 classes of the Mapillary Vistas dataset wil be used. if it is false,
        then only the 11+1 classes that overlap with the classes of the COCO dataset will be used
        train: if True it is assumed the training data is preprocessed, otherwise the validation data.
        """
        self.root_path = root_path
        self.save_path = save_path
        self.mask_threshold = mask_threshold
        self.all_classes = all_classes
        self.train = train

        with open(os.path.join(self.root_path, "config.json")) as config_file:
            config = json.load(config_file)
        labels = config["labels"]

        # Parts of the image that don't belong to any classes will be considered to be part of the background
        if self.all_classes:
            self.background_value = 65
            self.kept_labels = [idx for idx, d in enumerate(labels) if d["instances"]]
        else:
            self.background_value = 27
            self.kept_labels = [
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

        if self.train:
            self.image_path = "training/images"
            self.mask_path = "training/instances"
        else:
            self.image_path = "validation/images"
            self.mask_path = "validation/instances"

        images_root_path = os.path.join(self.root_path, self.image_path)
        masks_root_path = os.path.join(self.root_path, self.mask_path)
        self.images = list(sorted(os.listdir(images_root_path)))
        self.masks = list(sorted(os.listdir(masks_root_path)))

        print('Reading data is completed')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        full_image_path = os.path.join(self.root_path, self.image_path, self.images[idx])
        full_mask_path = os.path.join(self.root_path, self.mask_path, self.masks[idx])

        image = Image.open(full_image_path)
        mask = Image.open(full_mask_path)

        mask_array = np.array(mask)
        obj_ids, counts = np.unique(mask_array, return_counts=True)

        filtered_obj_ids = []

        for idx, obj_id in enumerate(obj_ids):
            if obj_id // 256 in self.kept_labels and counts[idx] > self.mask_threshold:
                filtered_obj_ids.append(obj_id)

        if len(filtered_obj_ids) == 0:
            return None

        # Put the background value at every pixel that is not considered to be part of any detection objects
        boolean_array = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.bool)
        for id in filtered_obj_ids:
            int_bool_array = mask_array == id
            boolean_array = np.logical_or(int_bool_array, boolean_array)
        mask_array[np.logical_not(boolean_array)] = self.background_value
        new_mask_image = Image.fromarray(mask_array)

        image_save_path = os.path.join(self.save_path, "training/images", os.path.basename(full_image_path))
        mask_save_path = os.path.join(self.save_path, "training/instances", os.path.basename(full_mask_path))
        image.save(image_save_path)
        new_mask_image.save(mask_save_path)

        return None


def collate_fn(batch):
    return None


if __name__ == '__main__':
    root_path = "/Users/robert.vincze/Downloads/Mapillary"

    # Expected directory of saved path:
    # SAVED_PATH:
    # -training
    # -------images
    # -------instances
    # -validation
    # -------images
    # -------instances
    save_path = "/Users/robert.vincze/Downloads/Mapillary-testing"

    mask_threshold = 32 * 32
    all_classes = False
    train = True

    mapillary_dataset = MapillaryToBeFilteredDataset(root_path, save_path, mask_threshold, all_classes, train)
    data_loader = torch.utils.data.DataLoader(
        mapillary_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    batch_count = 0
    num_batches = len(data_loader)
    for result in data_loader:
        if batch_count % 10 == 0:
            print(f"Filtering status: {batch_count}/{num_batches}")
        batch_count = batch_count + 1
