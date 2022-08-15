"""
This script does just resizing. It assumes the objects already have the 32x32 masks filtered out and that there are
only classes from COCO taken into consideration
"""

import os
import numpy as np
import cv2 as cv
import torch
import datasets.transforms as T
import pycocotools.mask as cocomask
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torch.utils.data import Dataset

from datasets.mapillary_inst_seg_dataset import MapillaryInstSegDataset
from references.pytorch_detection.utils import collate_fn

if __name__ == '__main__':
    ROOT_PATH = "D:\Robert\Mapillary-32x32"
    SAVE_PATH = "D:\Robert\Mapillary-resize-training"
    transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])

    min_size = 800
    max_size = 1100
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    fixed_size = (1024, 1024)
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, fixed_size=fixed_size)

    mapillary_dataset = MapillaryInstSegDataset(ROOT_PATH, transforms)
    data_loader = torch.utils.data.DataLoader(
        mapillary_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    count = 0
    for images, targets in data_loader:
        images, targets = transform(images, targets)
        batch_path = os.path.join(SAVE_PATH, str(count))
        os.mkdir(batch_path)
        images_path = os.path.join(SAVE_PATH, str(count), "images")
        targets_path = os.path.join(SAVE_PATH, str(count), "targets")
        os.mkdir(images_path)
        os.mkdir(targets_path)

        for image, target in zip(images.tensors, targets):
            # Save the image
            image_numpy = image.detach().cpu().numpy()
            with open(os.path.join(images_path, "%06d.npy" % count), 'wb') as f:
                np.save(f, image_numpy)

            # Save the targets
            obj_ids = target["obj_ids"].tolist()
            boxes = target["boxes"].tolist()
            labels = target["labels"].tolist()
            areas = target["area"].tolist()
            iscrowd = target["iscrowd"].tolist()
            masks = target["masks"].cpu().numpy()
            masks = [cocomask.encode(np.asfortranarray(m.squeeze(axis=0), dtype=np.uint8))
                     for m in np.vsplit(masks, len(boxes))]

            with open(os.path.join(targets_path, "%06d.txt" % count), "w") as f:
                for obj_id, bbox, label, area, iscrowd, mask in zip(obj_ids, boxes, labels, areas, iscrowd, masks):
                    print(obj_id, *bbox, label, area, iscrowd, *mask['size'], mask['counts'].decode(encoding='UTF-8'),
                          file=f)

            print(count)
            count = count + 1
