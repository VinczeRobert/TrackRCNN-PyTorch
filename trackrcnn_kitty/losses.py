import torch
import numpy as np

from trackrcnn_kitty.utils import as_numpy


def compute_association_loss_for_detection(curr_det_id, dst_matrix, dets_axis_0, dets_axis_1):
    """
    :param curr_det_id: identificator of the current pytorch_detection for which we want to calculate the loss
    :param dst_matrix: a square matrix containing distances between all pairs of detections
    :param dets_axis_0: list of ids of all detections for this batch
    :param dets_axis_1: = dets_axis_0
    :return: loss, normalization
    """

    # Get a boolean mask from dets_axis_0 and dets_axis_1 where a True value
    # at position i means that dets_axis_0[i] = curr_det_id, False otherwise
    mask_axis_0 = torch.eq(dets_axis_0, curr_det_id)
    mask_axis_1 = torch.eq(dets_axis_1, curr_det_id)

    # Reduce the distance matrix to keep only the distances between objects
    # for which obj_id = curr_det_id
    sliced_dst_matrix = dst_matrix[mask_axis_0].T

    # Get a list of the classes of each pytorch_detection
    detection_classes = torch.div(dets_axis_1, 1000, rounding_mode='floor')
    # Get the class of the current pytorch_detection
    curr_det_class = torch.div(curr_det_id, 1000, rounding_mode='floor')
    # Get boolean mask from detection_classes where a True value
    # at position i means that detection_classes[i] = curr_det_class, False otherwise
    classes_mask = torch.eq(detection_classes, curr_det_class)

    # Reduce the distance matrix further to keep only the distances between
    # the current object and other objects which have the same class as the current object
    sliced_dst_matrix = sliced_dst_matrix[classes_mask]
    # Keep the mask only the detections which have the same class as the current pytorch_detection
    mask_axis_1 = mask_axis_1[classes_mask]

    # Get distances between objects with the same id
    # and distances between objects that have different ids
    same_dst_matrix = sliced_dst_matrix[mask_axis_1]
    different_dst_matrix = sliced_dst_matrix[torch.logical_not(mask_axis_1)]

    # Compute batch triplet loss
    if len(same_dst_matrix) > 0 and len(different_dst_matrix) > 0:
        margin = 0.2
        # Compute hard positives and hard negatives
        hard_pos = torch.max(same_dst_matrix, dim=0).values
        hard_neg = torch.min(different_dst_matrix, dim=0).values

        triplet_loss = torch.maximum(margin + hard_pos - hard_neg, torch.tensor(0))

        # return total loss for current pytorch_detection and normalization
        return np.asscalar(as_numpy(torch.sum(triplet_loss))), len(triplet_loss)
    else:
        return 0, 1


def compute_association_loss(associations, detection_ids):
    # Create a tensor of dim (D), D being the number of detections
    all_detection_ids = torch.cat(detection_ids, dim=0).cpu()

    # associations is a tensor of dim (D, 128), D being the number of detections
    # compute euclidean distance between every pair of detections from this batch
    detection_distances = torch.cdist(associations, associations)

    unique_detection_ids = torch.unique(all_detection_ids)

    loss = 0
    normalization = 0
    for detection_id in unique_detection_ids:
        loss_per_id, normalization_per_id = compute_association_loss_for_detection(detection_id,
                                                                                   detection_distances.cpu(),
                                                                                   all_detection_ids.cpu(),
                                                                                   all_detection_ids.cpu())
        loss += loss_per_id
        normalization += normalization_per_id

    loss = (loss / normalization)

    return torch.tensor(loss)
