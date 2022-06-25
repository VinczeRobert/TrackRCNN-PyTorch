import torchvision.transforms.functional as TF
import random


# Gamma correction:
# 1. Is applied only on the image
# 2. It is applied after normalization so we can't include it the initial transforms list
class GammaCorrection(object):
    def __init__(self, g_range):
        self.g_range = g_range

    def __call__(self, image):
        gamma = random.uniform(self.g_range[0], self.g_range[1])

        if gamma == 0:
            return image
        else:
            return TF.adjust_gamma(image, gamma, gain=1)


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
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = TF.to_tensor(image)
        return image, target
