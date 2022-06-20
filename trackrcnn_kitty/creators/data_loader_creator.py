import torch
import copy
import json

from references.pytorch_detection.utils import collate_fn
from trackrcnn_kitty.datasets.transforms import get_transforms


VAL_DATASET_DIM = 50


def get_data_loaders_for_penn_fudan(dataset, task, train_batch_size, test_batch_size, transforms_list):
    data_loaders = dict()
    test_dataset = copy.deepcopy(dataset)
    test_dataset.transforms = get_transforms(transforms_list, False)

    # We need to store the indices so we know which ones were used for testing
    # when we evaluate
    if task in ["train", "train+val"]:
        indices = torch.randperm(len(dataset)).tolist()
        train_indices = indices[:-VAL_DATASET_DIM]
        test_indices = indices[-VAL_DATASET_DIM:]

        with open("util/pfd_indices.json", "w") as f:
            json.dump({
                "train_indices": train_indices,
                "test_indices": test_indices
            }, f)
    else:
        with open("util/pfd_indices.json") as f:
            indices = json.load(f)
        train_indices = indices["train_indices"]
        test_indices = indices["test_indices"]

    dataset = torch.utils.data.Subset(dataset, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset, test_indices)

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    data_loader_test = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )

    data_loaders["test"] = data_loader_test
    data_loaders["train"] = data_loader_train

    return data_loaders


def get_data_loaders_for_kitti(dataset, task, train_batch_size, test_batch_size):
    # For KITTITrackSegDataset we either do only train or eval or save_preds (we don't do train+val)
    # That means we can assume that whether we do train or val the dataset is already
    # correctly loaded

    shuffle, batch_size, task = (True, train_batch_size, "train") if task == "train" else\
        (False, test_batch_size, "test")

    return {task: torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn
    )}


def get_data_loaders(dataset, config):
    if config.dataset == "KITTISegTrack":
        return get_data_loaders_for_kitti(dataset, config.task, config.train_batch_size, config.test_batch_size)
    else:
        return get_data_loaders_for_penn_fudan(dataset, config.task, config.train_batch_size, config.test_batch_size,
                                               config.transforms_list)
