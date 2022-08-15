import sys

import torch

from datasets.resized_dataset import ResizedDataset
from references.pytorch_detection.utils import collate_fn
from datasets.dataset_factory import get_dataset
import datasets.transforms as T


def __get_data_loader(dataset, batch_size, shuffle, num_workers):
    return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn
        )


def __get_transforms(transforms_list, train):
    transforms = [T.ToTensor()]

    if train:
        if "flip" in transforms_list:
            transforms.append(T.RandomHorizontalFlip(0.5))
        if "gamma" in transforms:
            transforms.append(T.GammaCorrection((-0.05, 0.05)))

    return T.Compose(transforms)


def __get_test_transforms():
    return T.ToTensor()


def get_data_loaders(config):
    data_loaders = dict()
    if config.task == "train":
        transforms = __get_transforms(config.transforms_list, True)
        training_dataset = get_dataset(config, transforms, True)
        data_loaders["train"] = __get_data_loader(
            training_dataset,
            config.train_batch_size,
            not config.add_associations,
            config.num_workers
        )
        num_classes = training_dataset.num_classes
        is_dataset_resized = isinstance(training_dataset, ResizedDataset)
    elif config.task in ["val", "save_preds", "annotate", "metrics", "save_preds_coco", "annotate_seq"]:
        transforms = __get_transforms(config.transforms_list, False)
        testing_dataset = get_dataset(config, transforms, False)
        data_loaders["test"] = __get_data_loader(
            testing_dataset,
            config.test_batch_size,
            not config.add_associations,
            config.num_workers
        )
        num_classes = testing_dataset.num_classes
        is_dataset_resized = isinstance(testing_dataset, ResizedDataset)
    elif config.task == "train+val":
        train_transforms = __get_transforms(config.transforms_list, True)
        test_transforms = __get_transforms(config.transforms_list, False)
        training_dataset = get_dataset(config, train_transforms, True)
        testing_dataset = get_dataset(config, test_transforms, False)
        data_loaders["train"] = __get_data_loader(
            training_dataset,
            config.train_batch_size,
            not config.add_associations,
            config.num_workers
        )
        data_loaders["test"] = __get_data_loader(
            testing_dataset,
            config.test_batch_size,
            False,
            config.num_workers
        )
        num_classes = training_dataset.num_classes
        # Here there is an assumption that either both datasets are resized or none of them are
        is_dataset_resized = isinstance(training_dataset, ResizedDataset)
    else:
        print("Invalid task in configuration file! Stopping program.")
        sys.exit(-1)

    return data_loaders, num_classes, is_dataset_resized
