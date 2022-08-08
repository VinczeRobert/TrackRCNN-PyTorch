import numpy as np
import pycocotools.mask as cocomask
import torch
from torch.utils.data import Dataset


class ResizedDataset(Dataset):
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # load image
        with open(self.images[idx], 'rb') as f:
            image = np.load(f)

        with open(self.targets[idx]) as f:
            content = f.readlines()

        obj_ids = []
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        masks = []

        for line in content:
            entries = line.split(' ')
            obj_ids.append(int(entries[0]))
            boxes.append([float(entries[1]), float(entries[2]), float(entries[3]), float(entries[4])])
            labels.append(int(entries[5]))
            areas.append(float(entries[6]))
            iscrowd.append(int(entries[7]))
            mask_coco = {"size": [int(entries[8]), int(entries[9])],
                         "counts": entries[10].strip().encode(encoding='UTF-8')}
            masks.append(cocomask.decode(mask_coco))

        boxes = torch.tensor(np.array(boxes), dtype=torch.float32)
        masks = torch.tensor(np.array(masks), dtype=torch.uint8)
        areas = torch.tensor(np.array(areas), dtype=torch.float32)
        labels = torch.tensor(np.array(labels), dtype=torch.int64)
        obj_ids = torch.tensor(np.array(obj_ids), dtype=torch.int16)
        iscrowd = torch.tensor(np.array(iscrowd), dtype=torch.uint8)

        target = {
            "boxes": boxes,
            "masks": masks,
            "areas": areas,
            "labels": labels,
            "obj_ids": obj_ids,
            "iscrowd": iscrowd
        }

        image = torch.tensor(np.array(image))

        return image, target
