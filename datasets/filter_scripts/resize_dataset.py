"""
This script gathers the images and targets (in the same format as during training with MapillaryInstSegDataset)
and resizes them. Images are all resized to a fixed size (1024, 1024) as numpy files.
Targets are saved as text files.
Training Mapillary with data already resized improves the training speed a lot.
Parameters in lines 22-36 need to be set by the user.
"""

import os

import numpy as np
import pycocotools.mask as cocomask
import torch
from torch.utils.data import Dataset
from torchvision.models.detection.transform import GeneralizedRCNNTransform

import datasets.transforms as T
from datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from datasets.mapillary_inst_seg_dataset import MapillaryInstSegDataset
from references.pytorch_detection.utils import collate_fn

if __name__ == '__main__':
    root_path = "/Users/robert.vincze/Downloads/Mapillary-testing"
    save_path = "/Users/robert.vincze/Downloads/Mapillary-testing-resize"

    all_classes = False
    train = True
    transforms = T.Compose([T.ToTensor(), T.RandomHorizontalFlip(0.5)])
    dataset_name = "Mapillary"  # can be "Mapillary or Kitti

    min_size = 800
    max_size = 1100
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    fixed_size = (1024, 309)
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std, fixed_size=fixed_size)

    if dataset_name == "Mapillary":
        dataset = MapillaryInstSegDataset(root_path, transforms, all_classes, train)
    else:
        dataset = KITTISegTrackDataset(root_path, transforms, train)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    count = 0
    batch_count = 0
    num_batches = len(data_loader)
    for images, targets in data_loader:
        images, targets = transform(images, targets)
        batch_path = os.path.join(save_path, str(count))
        os.mkdir(batch_path)
        images_path = os.path.join(save_path, str(count), "images")
        targets_path = os.path.join(save_path, str(count), "targets")
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
            areas = target["areas"].tolist()
            iscrowd = target["iscrowd"].tolist()
            masks = target["masks"].cpu().numpy()
            masks = [cocomask.encode(np.asfortranarray(m.squeeze(axis=0), dtype=np.uint8))
                     for m in np.vsplit(masks, len(boxes))]

            with open(os.path.join(targets_path, "%06d.txt" % count), "w") as f:
                for obj_id, bbox, label, area, iscrowd, mask in zip(obj_ids, boxes, labels, areas, iscrowd, masks):
                    print(obj_id, *bbox, label, area, iscrowd, *mask['size'], mask['counts'].decode(encoding='UTF-8'),
                          file=f)
            count = count + 1

        if batch_count % 10 == 0:
            print(f"Filtering status: {batch_count}/{num_batches}")
        batch_count = batch_count + 1
