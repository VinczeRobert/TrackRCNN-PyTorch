from datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from datasets.kitti_seg_track_resized_dataset import KITTISegTrackResizedDataset
from datasets.mapillary_inst_seg_dataset import MapillaryInstSegDataset
from datasets.mapillary_inst_seg_resized_dataset import MapillaryInstSegResizedDataset
from datasets.penn_fudan_dataset import PennFudanDataset


def get_dataset(config, transforms, train):
    if config.dataset == "KITTISegTrack":
        return KITTISegTrackDataset(config.dataset_path, transforms, train, config.sequence_number)
    elif config.dataset == "PennFudan":
        return PennFudanDataset(config.dataset_path, transforms, train)
    elif config.dataset == "MapillaryInstSeg":
        return MapillaryInstSegDataset(config.dataset_path, transforms, config.all_classes, train)
    elif config.dataset == "MapillaryInstSegResized":
        return MapillaryInstSegResizedDataset(config.dataset_path, config.all_classes, train)
    elif config.dataset == "KITTISegTrackResized":
        return KITTISegTrackResizedDataset(config.dataset_path, train)
    else:
        assert False, "Invalid dataset name"
