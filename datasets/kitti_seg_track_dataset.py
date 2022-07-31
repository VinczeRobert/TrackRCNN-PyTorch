import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KITTISegTrackDataset(Dataset):
    def __init__(self, root_path, transforms, train=True, sequence_number=None):
        """
        If sequence_no is not None then this class is going to load
        the images only for that particular sequence number.
        """
        self.root_path = root_path
        self.transforms = transforms
        self.num_classes = 3
        self.train = train
        self.sequence_number = sequence_number

        if self.train:
            images_root_path = os.path.join(self.root_path, "images/training")
            masks_root_path = os.path.join(self.root_path, "annotations/instances")
        else:
            images_root_path = os.path.join(self.root_path, "images/validation")
            masks_root_path = os.path.join(self.root_path, "annotations/val-instances")

        if self.sequence_number:
            images_root_path = os.path.join(images_root_path, sequence_number)
            masks_root_path = os.path.join(masks_root_path, sequence_number)
            self.images = list(sorted(os.listdir(images_root_path)))
            self.masks = list(sorted(os.listdir(masks_root_path)))
        else:
            sequence_numbers = list(sorted(os.listdir(images_root_path)))
            # only for Mac
            if sequence_numbers[0] == ".DS_Store":
                sequence_numbers = sequence_numbers[1:]

            self.images = []
            self.masks = []
            for seq in sequence_numbers:
                image_seq_root_path = os.path.join(images_root_path, seq)
                masks_seq_root_path = os.path.join(masks_root_path, seq)

                images = list(sorted(os.listdir(image_seq_root_path)))
                masks = list(sorted(os.listdir(masks_seq_root_path)))
                images = [os.path.join(seq, image) for image in images]
                masks = [os.path.join(seq, mask) for mask in masks]
                self.images.extend(images)
                self.masks.extend(masks)

        print('Reading data is completed...')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.train:
            image_path = os.path.join(self.root_path, "images/training", self.images[idx])
            mask_path = os.path.join(self.root_path, "annotations/instances", self.masks[idx])
        else:
            image_path = os.path.join(self.root_path, "images/validation", self.images[idx])
            mask_path = os.path.join(self.root_path, "annotations/val-instances", self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        # convert the PIL Image into a numpy array
        mask_array = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask_array)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        # remove object_ids that are equal to 10000
        # those are ignore regions
        obj_ids = np.asarray(list(filter(lambda x: x != 10000, obj_ids)))
        if len(obj_ids) == 0:
            # It means the current image has no detections in it
            # so we will return None and filter it out at training time

            # Normally these should be filtered out before loading the data
            # but here we do tracking and the relationships between consecutive images matter
            # More specifically, when calculating the association loss, we want it to be calculated for
            # batch_size adjacent frames, even if a good chunk of those frames might not have detections in it.
            return None, None

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

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

        if len(num_objs) == 0:
            # Images with no valid detections will be also filtered out at training time
            return None, None

        # there are two classes (excluding background)
        # we can get the class of an object by doing
        # floor division between the object id and 1000
        # 1 for car
        # 2 for pedestrian
        labels = []
        for obj_id in obj_ids:
            labels.append(obj_id // 1000)

        # Convert everything into a tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        masks = torch.as_tensor(masks, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        obj_ids = torch.as_tensor(obj_ids, dtype=torch.int16)

        # suppose all instances are not crowd because we explicitly eliminated ignore regions
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "masks": masks,
            "areas": areas,
            "labels": labels,
            "obj_ids": obj_ids,
            "iscrowd": is_crowd
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target
