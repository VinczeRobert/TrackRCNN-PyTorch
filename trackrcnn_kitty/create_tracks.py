"""
For now this script is only going to loop through one KITTI sequence
and link the ground truths in time, without any training happening.
"""
import munkres
import numpy as np
import torch

import trackrcnn_kitty.datasets.transforms as T
from references.pytorch_detection.utils import collate_fn
from trackrcnn_kitty.adnotate import adnotate
from trackrcnn_kitty.datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.utils import compute_overlaps_masks


MATCHING_THRESHOLD_CAR = 0.25
MATCHING_THRESHOLD_PEDESTRIAN = 0.15


def main():
    root_path = "/Users/robert.vincze/Downloads/KITTITrackSegValDataset/"
    transforms = T.ToTensor()
    reduced_dataset = KITTISegTrackDataset(root_path, transforms, False, "0002")
    batch_size = 8
    data_loader = torch.utils.data.DataLoader(
        reduced_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
    )

    is_first_batch = True
    last_pair_from_batch = None
    hungarian_algorithm = munkres.Munkres()
    association_dict = dict()

    obj_id_count = 0  # each newly detected object will get a new id, the others should keep the id

    for images, targets in data_loader:
        # We will link detections between each consecutive pair of frames
        # The last frame of every batch will have to be linked with the
        # first one from the next batch
        if is_first_batch:
            # The very first image can't be compared to any previous frame
            is_first_batch = False
            colors = []
            obj_ids = []
            for i in range(len(targets[0]["masks"])):
                random_color = np.random.choice(range(256), size=3)
                colors.append(random_color)
                obj_ids.append(obj_id_count)
                association_dict[obj_id_count] = (random_color, obj_id_count)
                obj_id_count = obj_id_count + 1
            adnotate(targets[0]["image_path"], targets[0]["masks"], obj_ids, colors)
        else:
            # Insert the last element of the previous batch at the beginning of the list
            targets = list(targets)
            targets.insert(0, last_pair_from_batch[1])

        for j in range(1, len(targets)):
            # get masks for current and previous images
            masks_j = targets[j]["masks"].numpy()
            masks_jm1 = targets[j - 1]["masks"].numpy()

            # Get the cost matrix which will be used by the Hungarian algorithm
            mask_overlaps = compute_overlaps_masks(masks_jm1, masks_j)
            mask_overlaps = mask_overlaps * (-1)

            # Get difference between rows and columns
            orig_columns_numbers = mask_overlaps.shape[1]
            row_column_diff = mask_overlaps.shape[0] - mask_overlaps.shape[1]
            extra_columns = []
            if row_column_diff > 0:
                # There are more rows than columns, columns need to be padded
                mask_overlaps = np.pad(mask_overlaps, [(0, 0), (0, row_column_diff)], mode='constant')
                extra_columns = [k for k in range(orig_columns_numbers, orig_columns_numbers + row_column_diff)]
            elif row_column_diff < 0:
                # There are more columns than rows, rows need to be added
                mask_overlaps = np.pad(mask_overlaps, [(0, abs(row_column_diff)), (0, 0)], mode='constant')

            # this method works directly on the inputs
            # it returns a list of tuples of dim 2
            pairs = hungarian_algorithm.compute(mask_overlaps.copy())

            mask_overlaps = mask_overlaps * (-1)
            non_matching_pairs = []
            # eliminate pairs which have an IoU smaller than MATCHING_THRESHOLD
            for i in range(len(pairs) - 1, -1, -1):
                try:
                    class_jm1 = targets[j]["labels"][pairs[i][0]].item()
                except IndexError:
                    class_jm1 = -1
                try:
                    class_jm = targets[j]["labels"][pairs[i][1]].item()
                except IndexError:
                    class_jm = -1

                # eliminate matches with overlapping less than a threshold or where the classes don't correspond
                threshold = MATCHING_THRESHOLD_CAR if class_jm == 1 else MATCHING_THRESHOLD_PEDESTRIAN
                if (mask_overlaps[pairs[i][0]][pairs[i][1]] < threshold) or \
                        (class_jm != class_jm1 and class_jm != -1 and class_jm1 != -1):
                    non_matching_pairs.append(pairs[i])
                    del pairs[i]

            next_association_dict = dict()
            obj_ids = [0] * mask_overlaps.shape[1]
            colors = [0] * mask_overlaps.shape[1]

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
            adnotate(targets[j]["image_path"], targets[j]["masks"], obj_ids, colors)

        else:
            # Save last value for the next batch
            last_pair_from_batch = (images[-1], targets[-1])


if __name__ == '__main__':
    main()
