"""
For now this script is only going to loop through one KITTI sequence
and link the ground truths in time, without any training happening.
"""
import torch
import numpy as np
import munkres
from sklearn.metrics import jaccard_score

from references.pytorch_detection.utils import collate_fn
from trackrcnn_kitty.adnotate import adnotate
from trackrcnn_kitty.datasets.kitti_seg_track_dataset import KITTISegTrackDataset
from trackrcnn_kitty.datasets.transforms import get_transforms


def compute_overlaps_masks(masks1, masks2):
    """
    Use sklearn's jaccard_score which is the same as IoU
    Taken from: https://github.com/michhar/pytorch-mask-rcnn-samples/blob/master/utils.py#L366
    """
    # flatten masks
    scores = np.zeros((masks1.shape[0], masks2.shape[0]))
    for idx1, mask1 in enumerate(masks1):
        for idx2, mask2 in enumerate(masks2):
            mask1_flattened = mask1.flatten()
            mask2_flattened = mask2.flatten()

            score = jaccard_score(mask1_flattened, mask2_flattened)
            scores[idx1][idx2] = score

    return scores


def main():
    root_path = "/Users/robert.vincze/Downloads/KITTITrackSegValDataset/"
    transforms = get_transforms([], False)
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
            object_ids = []
            for i in range(len(targets[0]["masks"])):
                random_color = np.random.choice(range(256), size=3)
                colors.append(random_color)
                object_ids.append(obj_id_count)
                association_dict[obj_id_count] = (random_color, obj_id_count)
                obj_id_count = obj_id_count + 1
            adnotate(targets[0]["image_path"], targets[0]["masks"], object_ids, colors)
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
            orig_row_numbers = mask_overlaps.shape[0]
            orig_columns_numbers = mask_overlaps.shape[1]
            row_column_diff = mask_overlaps.shape[0] - mask_overlaps.shape[1]
            extra_columns = []
            extra_rows = []
            if row_column_diff > 0:
                # There are more rows than columns, columns need to be padded
                mask_overlaps = np.pad(mask_overlaps, [(0, 0), (0, row_column_diff)], mode='constant')
                extra_rows = [k for k in range(orig_columns_numbers, orig_columns_numbers + row_column_diff)]
            elif row_column_diff < 0:
                # There are more columns than rows, rows need to be added
                mask_overlaps = np.pad(mask_overlaps, [(0, abs(row_column_diff)), (0, 0)], mode='constant')
                extra_columns = [k for k in range(orig_row_numbers, orig_row_numbers + abs(row_column_diff))]

            # this method works directly on the inputs
            # it returns a list of tuples of dim 2
            indexes = hungarian_algorithm.compute(mask_overlaps)

            # extra columns need to be eliminated from indexes because they will be added as new objects
            # extra rows also need to be eliminated from indexes because they are objects from the
            # previous frame which don't appear in the current one
            indexes = list(filter(lambda x: x[1] not in extra_rows and x[0] not in extra_columns, indexes))
            next_association_dict = dict()

            obj_ids = [0] * mask_overlaps.shape[1]
            colors = [0] * mask_overlaps.shape[1]

            # Add anything in extra columns as a new detection
            for ex in extra_columns:
                obj_ids[ex] = obj_id_count + 1
                random_color = np.random.choice(range(256), size=3)
                colors[ex] = random_color
                obj_id_count = obj_id_count + 1
                next_association_dict[ex] = (random_color, obj_id_count)

            for index in indexes:
                obj_id = association_dict[index[0]][1]
                color = association_dict[index[0]][0]
                obj_ids[index[1]] = obj_id
                colors[index[1]] = color
                next_association_dict[index[1]] = (color, obj_id)

            # delete detections from previous frame that are not present here
            if len(extra_rows) > 0:
                obj_ids = obj_ids[:-len(extra_rows)]
                colors = colors[:-len(extra_rows)]

            del association_dict
            association_dict = next_association_dict
            adnotate(targets[j]["image_path"], targets[j]["masks"], obj_ids, colors)

        else:
            # Save last value for the next batch
            last_pair_from_batch = (images[-1], targets[-1])


if __name__ == '__main__':
    main()
