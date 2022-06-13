import torchvision.transforms.functional as TF

import references.pytorch_detection.transforms as T


# There seems to be a gamma correction in the main
# example of TrackRCNN but it is not known yet what values they used
# so we don't use this calss yet
class GammaCorrection:
    def __init__(self):
        self.range = (-0.05, 0.05)

    def __call__(self, x):
        x = TF.adjust_gamma(x, self.range[0])
        x = TF.adjust_gamma(x, self.range[1])
        return x


def get_transforms(transforms_list, train):
    transforms = [T.ToTensor()]
    if "flip" in transforms_list and train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.Resize((309, 1024)))
    return T.Compose(transforms)
