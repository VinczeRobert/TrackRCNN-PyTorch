import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class KITTISegTrackDataset(Dataset):
    def __init__(self, root_path, transforms):
        self.root_path = root_path
        self.transforms = transforms

        self.image_paths = []
        self.mask_paths = []
        self.targets = []

        self.__build_list()
        print('Reading data is completed...')

    def __getitem__(self, idx):
        # load images and targets
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        target = self.targets[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        target["masks"] = masks
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self):
        return len(self.image_paths)

    def __build_list(self):
        masks_root_path = os.path.join(self.root_path, "annotations/instances")
        for root, dirs, files in os.walk(masks_root_path):
            for file in files:
                mask_path = os.path.join(root, file)

                mask_img = Image.open(mask_path)

                # convert the PIL Image into a numpy array
                mask = np.array(mask_img)
                # instances are encoded as different colors
                obj_ids = np.unique(mask)
                # first id is the background, so remove it
                obj_ids = obj_ids[1:]
                # remove object_ids that are equal to 10000
                # those are ignore regions
                obj_ids = np.asarray(list(filter(lambda x: x != 10000, obj_ids)))
                if len(obj_ids) == 0:
                    # it means the current image has no detections in it
                    # this will cause problems in Dataloader, so we skip it
                    continue

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
                labels = torch.as_tensor(labels, dtype=torch.int64)
                object_ids = torch.as_tensor(obj_ids, dtype=torch.int64)

                # suppose all instances are not crowd
                is_crowd = torch.zeros((num_objs,), dtype=torch.int64)

                # Suppose all instances are not crowd
                self.targets.append({
                    "boxes": boxes,
                    "labels": labels,
                    "object_ids": object_ids,
                    "iscrowd": is_crowd
                })

                image_path = mask_path.replace("annotations/instances", "images/training")
                self.image_paths.append(image_path)
                self.mask_paths.append(mask_path)
