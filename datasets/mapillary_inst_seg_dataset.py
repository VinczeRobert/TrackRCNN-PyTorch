import os
import json
import torch
import numpy as np

from PIL import Image
from torch.utils.data import Dataset

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class MapillaryInstSegDataset(Dataset):
    def __init__(self, root_path, transforms, all_classes=True, train=True):
        self.root_path = root_path
        self.transforms = transforms

        # read the config file
        # We are only interested in v2 and only in the labels
        with open(os.path.join(self.root_path, "config_v1.2.json")) as config_file:
            config = json.load(config_file)
        labels = config["labels"]

        if all_classes:
            self.num_classes = 37 + 1
            self.kept_labels = [idx for idx, d in enumerate(labels) if d["instances"]]
        else:
            self.num_classes = 11 + 1
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
        self.classes_dict = {c: idx for idx, c, in enumerate(self.kept_labels)}

        self.train = train
        if self.train:
            images_root_path = os.path.join(self.root_path, "training/images")
            masks_root_path = os.path.join(self.root_path, "training/instances")
        else:
            images_root_path = os.path.join(self.root_path, "validation/images")
            masks_root_path = os.path.join(self.root_path, "validation/instances")

        self.images = list(sorted(os.listdir(images_root_path)))
        self.masks = list(sorted(os.listdir(masks_root_path)))

        print('Reading data is completed...')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.train:
            image_path = os.path.join(self.root_path, "training/images", self.images[idx])
            mask_path = os.path.join(self.root_path, "training/instances", self.masks[idx])
        else:
            image_path = os.path.join(self.root_path, "validation/images",  self.images[idx])
            mask_path = os.path.join(self.root_path, "validation/instances", self.masks[idx])

        image = Image.open(image_path)
        mask = Image.open(mask_path)
        mask_array = np.array(mask)
        obj_ids = np.unique(mask_array)
        obj_ids = obj_ids[1:]

        masks = mask_array == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        areas = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])

            area = (xmax - xmin) * (ymax - ymin)

            # ignore bounding boxes whit area=0
            if area > 0:
                boxes.append([xmin, ymin, xmax, ymax])
                areas.append(area)
            else:
                num_objs = num_objs - 1

        labels = []
        for obj_id in obj_ids:
            labels.append(self.classes_dict[obj_id // 256])

        # Convert everything into a tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.bool)
        obj_ids = torch.tensor(obj_ids, dtype=torch.int16)

        # Suppose all instances are not crowd
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "areas": areas,
            "iscrowd": is_crowd,
            "obj_ids": obj_ids,
        }

        if self.transforms is not None and len(target["boxes"]) > 0:
            image, target = self.transforms(image, target)

        return image, target
