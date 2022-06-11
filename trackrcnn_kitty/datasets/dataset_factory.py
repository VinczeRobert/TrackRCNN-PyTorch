from trackrcnn_kitty.datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.datasets.penn_fudan_dataset import PennFudanDataset


def get_dataset(dataset_name, dataset_path, transforms):
    if dataset_name == "KITTISegTrack":
        return KITTISegTrackDataset(dataset_path, transforms)
    elif dataset_name == "PennFudan":
        return PennFudanDataset(dataset_path, transforms)
