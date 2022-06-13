import random

import numpy as np
import torch
from torchvision.transforms import functional as F, ToPILImage


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class Resize(object):
    def __init__(self, resize_shape):
        self.resize_shape = resize_shape

    def __call__(self, image, target):
        x = image.shape[2]
        y = image.shape[1]
        pil_image = ToPILImage()(image)
        resized_image = pil_image.resize(self.resize_shape)
        resized_image = np.array(resized_image)
        resized_image = resized_image.reshape((resized_image.shape[2],
                                               resized_image.shape[1],
                                               resized_image.shape[0]))
        resized_image = torch.tensor(resized_image, dtype=torch.float32)

        resized_masks = []
        for idx, mask in enumerate(target["masks"]):
            mask = ToPILImage()(mask)
            resized_mask = mask.resize(self.resize_shape)
            resized_mask = np.array(resized_mask)
            resized_masks.append(torch.as_tensor(resized_mask))
        target["masks"] = torch.as_tensor(np.stack(resized_masks, axis=0))

        x_scale = self.resize_shape[1] / x
        y_scale = self.resize_shape[0] / y

        resized_boxes = []
        for idx, bb in enumerate(target["boxes"]):
            x = bb[0] * x_scale
            y = bb[1] * y_scale
            x_max = bb[2] * x_scale
            y_max = bb[3] * y_scale
            resized_boxes.append(torch.as_tensor([x, y, x_max, y_max]))
        target["boxes"] = torch.as_tensor(np.stack(resized_boxes, axis=0))

        return resized_image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
