import os
import json
import numpy as np
import torch

from PIL import Image
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import Dataset


class Mapillary32x32CocoClasses(Dataset):
    def __init__(self, root_path, train=True):
        self.root_path = root_path
        self.train = train

        with open(os.path.join(self.root_path, "config.json")) as config_file:
            config = json.load(config_file)
        self.labels = config["labels"]

        self.SAVE_PATH = "/Users/robert.vincze/Downloads/Mapillary-32x32-resize_coco_classes"
        self.MASK_THRESHOLD = 32 * 32
        self.SPECIAL_BACKGROUND_VALUE = 27  # (SKY)

        self.labels_with_instances_coco_overlap = [
            0,   # Bird
            19,  # Person
            20,  # Bicyclist
            21,  # Motorcyclist
            33,  # Bench
            38,  # Fire Hydrant
            48,  # Traffic Light
            54,  # Bus
            55,  # Car
            57,  # Motorcycle
            61   # Truck
        ]
        self.classes_dict = {c: idx for idx, c, in enumerate(self.labels_with_instances_coco_overlap)}

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
            if obj_id // 256 in self.labels_with_instances_coco_overlap and counts[idx] > self.MASK_THRESHOLD:
                filtered_obj_ids.append(obj_id)

        if len(filtered_obj_ids) == 0:
            return None

        boolean_array = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.bool)
        for id in filtered_obj_ids:
            int_bool_array = mask_array == id
            boolean_array = np.logical_or(int_bool_array, boolean_array)

        mask_array[np.logical_not(boolean_array)] = self.SPECIAL_BACKGROUND_VALUE
        new_mask_image = Image.fromarray(mask_array)

        image_save_path = os.path.join(self.SAVE_PATH, "training/images", os.path.basename(full_image_path))
        mask_save_path = os.path.join(self.SAVE_PATH, "training/instances", os.path.basename(full_mask_path))
        image.save(image_save_path)
        new_mask_image.save(mask_save_path)

        return None

def collate_fn(batch):
    return None


if __name__ == '__main__':
    ROOT_PATH = "/Users/robert.vincze/Downloads/Mapillary"

    min_size = 800
    max_size = 1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    # transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    mapillary_dataset = Mapillary32x32CocoClasses(ROOT_PATH)
    data_loader = torch.utils.data.DataLoader(
        mapillary_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    count = 0
    for result in data_loader:
        count = count + 1
        print(count)
