import torch
import copy
from references.detection.utils import collate_fn
from trackrcnn_kitty.datasets.transforms import get_transforms


def get_data_loaders_for_penn_fudan(dataset, task, train_batch_size, test_batch_size, transforms_list):
    # For PennFudanDataset we either do only train or train+val
    # We actually fetch both data_loaders no matter what because logic
    # of this class has not been written het

    data_loaders = dict()
    test_dataset = copy.deepcopy(dataset)
    test_dataset.transforms = get_transforms(transforms_list, False)

    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    test_dataset = torch.utils.data.Subset(test_dataset, indices[-50:])

    data_loader_train = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=True,
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
    # For KITTI we either do only train or eval or save_preds (we don't do train+val)
    # That means we can assume that weither we do train or val the dataset is already
    # correctly loaded

    shuffle, batch_size, task = (True, train_batch_size, "train") if task == "train" else\
        (False, test_batch_size, "test")

    return {task: torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        collate_fn=collate_fn
    )}


def get_data_loaders(dataset, dataset_name, task, train_batch_size, test_batch_size, transforms_list):
    if dataset_name == "KITTISegTrack":
        return get_data_loaders_for_kitti(dataset, task, train_batch_size, test_batch_size)
    else:
        return get_data_loaders_for_penn_fudan(dataset, task, train_batch_size, test_batch_size, transforms_list)
