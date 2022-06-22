import sys

import torch

from references.pytorch_detection.utils import collate_fn
from trackrcnn_kitty.datasets.dataset_factory import get_dataset
from trackrcnn_kitty.datasets.transforms import get_transforms


def __get_data_loader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


def get_data_loaders(config):
    data_loaders = dict()
    if config.task == "train":
        transforms = get_transforms(config.transforms_list, True)
        training_dataset = get_dataset(config, transforms, True)
        data_loaders["train"] = __get_data_loader(
            training_dataset,
            config.train_batch_size,
            True,
            4
        )
        num_classes = training_dataset.num_classes
    elif config.task in ["val", "save_preds"]:
        transforms = get_transforms(config.transforms_list, False)
        testing_dataset = get_dataset(config, transforms, False)
        data_loaders["test"] = __get_data_loader(
            testing_dataset,
            config.test_batch_size,
            False,
            1
        )
        num_classes = testing_dataset.num_classes
    elif config.task == "train+val":
        train_transforms = get_transforms(config.transforms_list, True)
        test_transforms = get_transforms(config.transforms_list, False)
        training_dataset = get_dataset(config, train_transforms, True)
        testing_dataset = get_dataset(config, test_transforms, False)
        data_loaders["train"] = __get_data_loader(
            training_dataset,
            config.train_batch_size,
            True,
            4
        )
        data_loaders["test"] = __get_data_loader(
            testing_dataset,
            config.test_batch_size,
            False,
            1
        )
        num_classes = training_dataset.num_classes
    else:
        print("Invalid task in configuration file! Stopping program.")
        sys.exit(-1)

    return data_loaders, num_classes
