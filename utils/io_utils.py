from itertools import groupby

import numpy as np
import os
import pycocotools.mask as cocomask
import torch

from utils.detection_utils import filter_masks, annotate_image


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
            mask = mask[0].cpu().detach().numpy()
            binary_mask = mask > 0.5
            rle_mask = binary_mask_to_rle(binary_mask)
            f.write(str(rle_mask)+"\n")


def save_tracking_prediction_for_batch(outputs, results_path, track):
    with torch.no_grad():
        for output in outputs:
            boxes = output["boxes"].cpu().numpy()
            if len(boxes) == 0:
                continue
            scores = output["scores"].cpu().numpy()
            labels = output["labels"].cpu().numpy()
            association_vectors = output["association_vectors"].cpu().numpy()
            masks = output["masks"].cpu().numpy()
            masks = masks.reshape((masks.shape[0], masks.shape[2], masks.shape[3]))
            masks = masks > 0.5
            masks = masks.astype("uint8")
            masks = [cocomask.encode(np.asfortranarray(m.squeeze(axis=0), dtype=np.uint8))
                     for m in np.vsplit(masks, len(boxes))]

            with open(results_path, "a") as f:
                for box, score, association_vector, class_, mask in zip(boxes, scores, association_vectors,
                                                                        labels, masks):
                    print(track, *box, score, class_, *mask["size"], mask["counts"].decode(encoding="UTF-8"),
                          *association_vector, file=f)
                track = track + 1

    return track


def save_detections_for_batch(outputs, targets, threshold_car, threshold_pedestrian, out_folder, obj_id):
    for idx, output in enumerate(outputs):
        masks_to_draw, _, _ = filter_masks(output, threshold_car, threshold_pedestrian)

        if len(masks_to_draw) == 0:
            continue

        objects_ids = [i for i in range(obj_id, obj_id + len(masks_to_draw))]
        colours = [np.random.choice(range(256), size=3) for _ in range(len(objects_ids))]

        out_filename = os.path.join(out_folder, "%06d.jpg" % idx)
        annotate_image(targets[idx]["image_path"], masks_to_draw, objects_ids, colours, out_filename)
        obj_id = obj_id + len(objects_ids)

    return obj_id


def load_tracking_predictions(detections_import_path):
    with open(detections_import_path) as f:
        content = f.readlines()

    boxes = []
    scores = []
    association_vectors = []
    classes = []
    masks = []
    for line in content:
        entries = line.split(' ')
        t = int(entries[0])
        while t + 1 > len(boxes):
            boxes.append([])
            scores.append([])
            association_vectors.append([])
            classes.append([])
            masks.append([])
        boxes[t].append([float(entries[1]), float(entries[2]), float(entries[3]), float(entries[4])])
        scores[t].append(float(entries[5]))
        classes[t].append(int(entries[6]))

        masks[t].append({"size": [int(entries[7]), int(entries[8])],
                         "counts": entries[9].strip().encode(encoding='UTF-8')})
        association_vectors[t].append([float(e) for e in entries[10:]])

    return boxes, scores, association_vectors, classes, masks
