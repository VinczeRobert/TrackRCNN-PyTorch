import os.path

import cv2 as cv
import numpy as np

SAVE_TRACKING_RESULTS_PATH = "D:\\Robert\\EXPERIMENTAL_RESULTS"


def adnotate(image, masks, obj_ids, colors, save_path):
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


if __name__ == '__main__':
    image_path = "/Users/robert.vincze/Downloads/KITTITrackSegValDataset/images/validation/0002/000099.png"
    mask_path = "/Users/robert.vincze/Downloads/KITTITrackSegValDataset/annotations/val-instances/0002/000099.png"

    image = cv.imread(image_path, -1)
    image = np.asarray(image, dtype=np.float64)
    mask = cv.imread(mask_path, -1)

    obj_ids = np.unique(mask)
    obj_ids = obj_ids[1:]
    obj_ids = np.asarray(list(filter(lambda x: x != 10000, obj_ids)))

    # split the color-encoded mask into a set
    # of binary masks
    masks = mask == obj_ids[:, None, None]
    colors = []

    for i in range(len(masks)):
        random_color = np.random.choice(range(256), size=3)
        colors.append(colors)

    adnotate(image_path, masks, obj_ids, colors)
