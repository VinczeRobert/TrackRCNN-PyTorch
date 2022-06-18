from trackrcnn_kitty.datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.datasets.penn_fudan_dataset import PennFudanDataset


def get_dataset(config, transforms, train):
    if config.dataset == "KITTISegTrack":
        return KITTISegTrackDataset(config.dataset_path, transforms, train)
    elif config.dataset == "PennFudan":
        return PennFudanDataset(config.dataset_path, transforms)
