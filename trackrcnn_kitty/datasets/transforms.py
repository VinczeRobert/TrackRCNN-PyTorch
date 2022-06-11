import references.detection.transforms as T


def get_transforms(transforms_list):
    transforms = [T.ToTensor()]
    if "flip" in transforms_list:
        transforms.append(T.RandomHorizontalFlip(0.5))
    # if "gamma_correction" in transforms_list:
    #     transforms.append(adjust_gamma)

    return T.Compose(transforms)
