import os
import json
import numpy as np

from PIL import Image
from torchvision.models.detection.transform import GeneralizedRCNNTransform

if __name__ == '__main__':
    ROOT_PATH = "D:\\Robert\\Mapillary"
    SAVE_PATH = "D:\\Robert\\Mapillary-32x32-resize"
    MASK_THRESHOLD = 32 * 32
    SPECIAL_BACKGROUND_VALUE = 65
    TRAIN = True

    with open(os.path.join(ROOT_PATH, "config_v1.2.json")) as config_file:
        config = json.load(config_file)
    labels = config["labels"]
    labels_with_instances = [idx for idx, d in enumerate(labels) if d["instances"]]
    classes_dict = {c: idx for idx, c, in enumerate(labels_with_instances)}

    min_size = 800
    max_size = 1333
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)

    if TRAIN:
        image_path = "training/images"
        mask_path = "training/v1.2/instances"
    else:
        image_path = "validation/images"
        mask_path = "validation/v1.2/instances"

    images_root_path = os.path.join(ROOT_PATH, image_path)
    masks_root_path = os.path.join(ROOT_PATH, mask_path)
    images = list(sorted(os.listdir(images_root_path)))
    masks = list(sorted(os.listdir(masks_root_path)))

    for idx in range(len(images)):
        print(idx)
        full_image_path = os.path.join(ROOT_PATH, image_path, images[idx])
        full_mask_path = os.path.join(ROOT_PATH, mask_path, masks[idx])

        image = Image.open(full_image_path)
        mask = Image.open(full_mask_path)

        mask_array = np.array(mask)
        obj_ids, counts = np.unique(mask_array, return_counts=True)

        filtered_obj_ids = []

        for idx, obj_id in enumerate(obj_ids):
            if obj_id // 256 in labels_with_instances and counts[idx] > MASK_THRESHOLD:
                filtered_obj_ids.append(obj_id)

        if len(filtered_obj_ids) == 0:
            continue

        boolean_array = np.zeros((mask_array.shape[0], mask_array.shape[1]), dtype=np.bool)
        for id in filtered_obj_ids:
            int_bool_array = mask_array == id
            boolean_array = np.logical_or(int_bool_array, boolean_array)

        mask_array[np.logical_not(boolean_array)] = SPECIAL_BACKGROUND_VALUE
        new_mask_image = Image.fromarray(mask_array)

        image_save_path = os.path.join(SAVE_PATH, "training/images", os.path.basename(full_image_path))
        mask_save_path = os.path.join(SAVE_PATH, "training/v1.2/instances", os.path.basename(full_mask_path))
        image.save(image_save_path)
        new_mask_image.save(mask_save_path)
