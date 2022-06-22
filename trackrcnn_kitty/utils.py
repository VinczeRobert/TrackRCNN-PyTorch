from itertools import groupby

import torch
import numpy as np


def as_numpy(tensor):
    return tensor.cpu().detach().numpy()


def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def check_for_degenerate_boxes(targets):
    # For a valid box the height and width have to be positive numbers
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb = boxes[bb_idx].tolist()
                raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                )


def validate_and_build_stacked_boxes(targets, is_training=True):
    # the stacked boxes are needed for the association head
    stacked_boxes = []
    if is_training:
        assert targets is not None
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError(
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
            else:
                raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

            stacked_boxes.extend(boxes)

        stacked_boxes = torch.stack(stacked_boxes)
        return stacked_boxes

    return None


def binary_mask_to_rle(binary_mask):
    """
    RLE is a simple yet efficient format for storing binary masks.
    RLE first divides a vector (or vectorized image) into a series of
    piecewise constant regions and then for each piece simply stores
    the length of that piece.
    For example, given M=[0 0 1 1 1 0 1] the RLE counts would be
    [2 3 1 1], or for M=[1 1 1 1 1 1 0] the counts would be [0 6 1]
    (note that the odd counts are always the numbers of zeros)."""

    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='F'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def write_detection_to_file(pred, file_name):
    """
    This function takes a prediction and writes the resulted
    detections in a file in the following format:
    <class_name> <confidence> <left> <top> <right> <bottom>
    e.g. 1 0.471781 0 13 174 244
    """

    with open(file_name, "w") as f:
        for i in range(len(pred["boxes"])):
            label = pred["labels"][i]
            confidence = pred["scores"][i]
            box = pred["boxes"][i]

            new_line = f"{label} {confidence} {box[0]} {box[1]} {box[2]} {box[3]}\n"
            f.write(new_line)


def write_gt_to_file(pred, file_name):
    """
    Same as above, but without the scores.
    """

    with open(file_name, "w") as f:
        for i in range(len(pred["boxes"])):
            label = pred["labels"][i]
            box = pred["boxes"][i]

            new_line = f"{label} {box[0]} {box[1]} {box[2]} {box[3]}\n"
            f.write(new_line)


def write_segmentation_mask_to_file(masks, file_name):
    """
    This function takes a list of masks and writes them
    in a file using RLE encoding.
    """
    with open(file_name, "w") as f:
        for mask in masks:
            mask = as_numpy(mask[0])
            binary_mask = mask > 0.5
            rle_mask = binary_mask_to_rle(binary_mask)
            f.write(str(rle_mask)+"\n")


def compute_iou(box, boxes, box_area, boxes_area):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [y1, x1, y2, x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.
    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    y1 = np.maximum(box[0], boxes[:, 0])
    y2 = np.minimum(box[2], boxes[:, 2])
    x1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    union = box_area + boxes_area[:] - intersection[:]
    iou = intersection / union
    return iou


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1, area2[i], area1)
    return overlaps
