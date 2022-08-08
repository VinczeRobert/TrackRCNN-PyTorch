import cv2 as cv
import numpy as np
import torch


def filter_masks(output, threshold_car, threshold_pedestrian):
    masks_to_draw = []
    association_vectors_to_use = []

    # Keep only masks that have a chance of being selected
    scores = output["scores"]
    scores = scores[scores >= threshold_car]
    masks = output["masks"]
    masks = masks[:len(scores)]
    association_vectors = output["association_vectors"]
    association_vectors = association_vectors[:len(association_vectors)]

    # Because pedestrians have a higher threshold, they need an extra check
    labels = output["labels"]
    labels = labels[:len(scores)]

    for i, label in enumerate(labels):
        if label == 1 or scores[i] >= threshold_pedestrian:
            masks_to_draw.append(masks[i])
            association_vectors_to_use.append(association_vectors[i])

    if len(masks_to_draw) == 0:
        return masks_to_draw, association_vectors_to_use, labels

    masks_to_draw = [mask.detach().cpu().reshape((mask.shape[1], mask.shape[2])) for mask in masks_to_draw]
    masks_to_draw = torch.stack(masks_to_draw, dim=0)
    masks_to_draw = masks_to_draw > 0.5

    association_vectors_to_use = torch.stack(association_vectors_to_use, dim=0)

    return masks_to_draw, association_vectors_to_use, labels


def annotate_image(image, masks, obj_ids, colors, save_path):
    for idx, mask in enumerate(masks):
        # equal color where mask, else image
        # this would paint your object silhouette entirely with `color`
        # random_color = np.random.choice(range(256), size=3)
        masked_image = np.where(mask[..., None], colors[idx], image)

        # use `addWeighted` to blend the two images
        # the object will be tinted toward `color`
        # image = cv.addWeighted(image, 0.8, masked_image, 0.2, 0)
        image = masked_image

    # Next we calculate bounding_boxes to draw track ids
    for i in range(len(obj_ids)):
        pos = np.where(masks[i])
        xmin = np.min(pos[1])
        xmax = np.max(pos[1])
        ymin = np.min(pos[0])
        ymax = np.max(pos[0])

        x_middle = int(xmin + (xmax - xmin) / 2)
        y_middle = int(ymin + (ymax - ymin) / 2)

        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(image, 'ID: ' + str(obj_ids[i]+1), (x_middle, y_middle), font, 0.5, (0, 0, 0), 2, cv.LINE_AA)

        cv.imwrite(save_path, image)
