from datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from datasets.mapillary_inst_seg_dataset import MapillaryInstSegDataset
from datasets.penn_fudan_dataset import PennFudanDataset


def get_dataset(config, transforms, train):
    if config.dataset == "KITTISegTrack":
        return KITTISegTrackDataset(config.dataset_path, transforms, train, config.sequence_number)
    elif config.dataset == "PennFudan":
        return PennFudanDataset(config.dataset_path, transforms, train)
    elif config.dataset == "MapillaryInstSeg":
        return MapillaryInstSegDataset(config.dataset_path, transforms, train)
