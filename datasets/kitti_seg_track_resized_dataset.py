import os

from datasets.resized_dataset import ResizedDataset


class KITTISegTrackResizedDataset(ResizedDataset):
    def __init__(self, root_path, train=True):

        self.root_path = root_path
        self.train = train
        self.num_classes = 3

        if self.train:
            images_root_path = os.path.join(self.root_path, "images/training")
            masks_root_path = os.path.join(self.root_path, "annotations/instances")
        else:
            images_root_path = os.path.join(self.root_path, "images/validation")
            masks_root_path = os.path.join(self.root_path, "annotations/val-instances")

        sequence_numbers = list(sorted(os.listdir(images_root_path)))
        # only for Mac
        if sequence_numbers[0] == ".DS_Store":
            sequence_numbers = sequence_numbers[1:]

        images = []
        masks = []
        for seq in sequence_numbers:
            image_seq_root_path = os.path.join(images_root_path, seq)
            masks_seq_root_path = os.path.join(masks_root_path, seq)

            images = list(sorted(os.listdir(image_seq_root_path)))
            masks = list(sorted(os.listdir(masks_seq_root_path)))
            images = [os.path.join(seq, image) for image in images]
            masks = [os.path.join(seq, mask) for mask in masks]
            images.extend(images)
            masks.extend(masks)

        super().__init__(images, masks)
