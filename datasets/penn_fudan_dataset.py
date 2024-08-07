import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class PennFudanDataset(Dataset):
    def __init__(self, root, transforms, train):
        self.root = root
        self.transforms = transforms
        self.num_classes = 2
        self.train = train

        # sort the images to make sure they are aligned
        if train:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages", "training"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PNGMasks", "training"))))
        else:
            self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages", "validation"))))
            self.masks = list(sorted(os.listdir(os.path.join(root, "PNGMasks", "validation"))))

    def __getitem__(self, idx):
        # load images and masks
        if self.train:
            img_path = os.path.join(self.root, "PNGImages", "training", self.imgs[idx])
            mask_path = os.path.join(self.root, "PNGMasks", "training", self.masks[idx])
        else:
            img_path = os.path.join(self.root, "PNGImages", "validation", self.imgs[idx])
            mask_path = os.path.join(self.root, "PNGMasks", "validation", self.masks[idx])

        img = Image.open(img_path).convert("RGB")

        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background
        mask = Image.open(mask_path)
        # convert the PIL Image into a numpy array
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

        targets = {"boxes": boxes, "labels": labels, "masks": masks, "image_id": image_id, "area": area,
                   "iscrowd": is_crowd}

        if self.transforms is not None:
            img, targets = self.transforms(img, targets)

        return img, targets

    def __len__(self):
        return len(self.imgs)
