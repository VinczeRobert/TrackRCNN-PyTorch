"""
For now this script is only going to loop through one KITTI sequence
and link the ground truths in time, without any training happening.
"""
import munkres
import numpy as np
import torch
import torch.nn.functional as F
import cv2 as cv

import trackrcnn_kitty.datasets.transforms as T
from references.pytorch_detection.utils import collate_fn
from trackrcnn_kitty.adnotate import adnotate
from trackrcnn_kitty.datasets.kitti_seg_track_dataset import KITTISegTrackDataset


MATCHING_THRESHOLD_CAR = 0.25
MATCHING_THRESHOLD_PEDESTRIAN = 0.15


def adnotate_first_image(target, association_dict, obj_id_count):
    colors = []
    obj_ids = []
    for i in range(len(target["masks"])):
        random_color = np.random.choice(range(256), size=3)
        colors.append(random_color)
        obj_ids.append(obj_id_count)
        association_dict[obj_id_count] = (random_color, obj_id_count)
        obj_id_count = obj_id_count + 1
    adnotate(target["image_path"], target["masks"], obj_ids, colors)

    return association_dict, obj_id_count


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow = -flow
    flow[:, :, 0] += np.arange(w)
    flow[:, :, 1] += np.arange(h)[:, np.newaxis]
    res = cv.remap(img, flow, None, cv.INTER_LINEAR)
    return res


def find_tracks_for_one_image(targets_j, targets_jm1, association_dict, obj_id_count):
    associations_j = targets_j["association_vectors"]
    associations_jm1 = targets_jm1["association_vectors"]
    hungarian_algorithm = munkres.Munkres()

    # Get the cost matrix which will be used by the Hungarian algorithm
    av_distances = torch.cdist(associations_jm1, associations_j)

    # Get difference between rows and columns
    orig_columns_numbers = av_distances.shape[1]
    row_column_diff = av_distances.shape[0] - av_distances.shape[1]
    extra_columns = []
    if row_column_diff > 0:
        # There are more rows than columns, columns need to be padded
        av_distances = F.pad(input=av_distances, pad=(0, row_column_diff, 0, 0), mode='constant', value=0)
        extra_columns = [k for k in range(orig_columns_numbers, orig_columns_numbers + row_column_diff)]
    elif row_column_diff < 0:
        # There are more columns than rows, rows need to be added
        av_distances = F.pad(input=av_distances, pad=(0, 0, 0, abs(row_column_diff)), mode='constant')

    # this method works directly on the inputs
    # it returns a list of tuples of dim 2
    pairs = hungarian_algorithm.compute(av_distances.detach().cpu().numpy().copy())

    non_matching_pairs = []
    # eliminate pairs which have an IoU smaller than MATCHING_THRESHOLD
    for i in range(len(pairs) - 1, -1, -1):
        try:
            class_jm1 = targets_j["labels"][pairs[i][0]].item()
        except IndexError:
            class_jm1 = -1
        try:
            class_jm = targets_jm1["labels"][pairs[i][1]].item()
        except IndexError:
            class_jm = -1

        # eliminate matches where one element was added artificially
        if (av_distances[pairs[i][0]][pairs[i][1]] == 0) or \
                (class_jm != class_jm1 and class_jm != -1 and class_jm1 != -1):
            non_matching_pairs.append(pairs[i])
            del pairs[i]

    next_association_dict = dict()
    obj_ids = [0] * av_distances.shape[1]
    colors = [0] * av_distances.shape[1]

    # Loop through the non matching pairs and if the column is not one
    # that was artificially added for the Jaccard score, then add it
    for idx in non_matching_pairs:
        if idx[1] not in extra_columns:
            obj_ids[idx[1]] = obj_id_count
            random_color = np.random.choice(range(256), size=3)
            colors[idx[1]] = random_color
            next_association_dict[idx[1]] = (random_color, obj_id_count)
            obj_id_count = obj_id_count + 1

    for index in pairs:
        obj_id = association_dict[index[0]][1]
        color = association_dict[index[0]][0]
        obj_ids[index[1]] = obj_id
        colors[index[1]] = color
        next_association_dict[index[1]] = (color, obj_id)

    # delete detections from previous frame that are not present here
    # The new fake columns would always be at the end of this list
    if len(extra_columns) > 0:
        obj_ids = obj_ids[:-len(extra_columns)]
        colors = colors[:-len(extra_columns)]

    del association_dict
    association_dict = next_association_dict
    adnotate(targets_j["image_path"], targets_j["masks"], obj_ids, colors)

    return association_dict, obj_id_count


def main():
    root_path = "D:\\Robert\\KITTITrackSegDataset"
    transforms = T.ToTensor()
    reduced_dataset = KITTISegTrackDataset(root_path, transforms, False, "0006")
    batch_size = 8
    data_loader = torch.utils.data.DataLoader(
        reduced_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    is_first_batch = True
    last_target_from_batch = None
    association_dict = dict()

    obj_id_count = 0  # each newly detected object will get a new id, the others should keep the id

    for images, targets in data_loader:
        # We will link detections between each consecutive pair of frames
        # The last frame of every batch will have to be linked with the
        # first one from the next batch
        if is_first_batch:
            # The very first image can't be compared to any previous frame
            is_first_batch = False
            association_dict, obj_id_count = adnotate_first_image(targets[0], association_dict, obj_id_count)
        else:
            # Insert the last element of the previous batch at the beginning of the list
            targets = list(targets)
            targets.insert(0, last_target_from_batch)

        for j in range(1, len(targets)):
            # get masks for current and previous images
            association_dict, obj_id_count = find_tracks_for_one_image(targets[j], targets[j - 1], association_dict,
                                                                       obj_id_count)
        else:
            # Save last value for the next batch
            last_target_from_batch = targets[-1]


if __name__ == '__main__':
    main()
