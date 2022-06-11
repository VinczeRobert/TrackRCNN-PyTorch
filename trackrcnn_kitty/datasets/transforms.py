from torchvision.transforms import ToTensor, RandomHorizontalFlip, Compose
from torchvision.transforms.functional import adjust_gamma


def get_transforms(transforms_list):
    transforms = [ToTensor()]
    if "flip" in transforms_list:
        transforms.append(RandomHorizontalFlip(0.5))
    if "gamma_correction" in transforms_list:
        transforms.append(adjust_gamma)

    return Compose(transforms)
