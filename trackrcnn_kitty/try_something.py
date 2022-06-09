import PIL.Image as Image
import numpy as np

if __name__ == '__main__':
    img = np.array(Image.open("D://Robert//KITTITrackSegDataset//annotations//instances//0001//000000.png"))
    obj_ids = np.unique(img)
    # to correctly interpret the id of a single object
    obj_id = obj_ids[0]
    class_id = obj_id // 1000
    obj_instance_id = obj_id % 1000