import os

from datasets.resized_dataset import ResizedDataset


class KITTISegTrackResizedDataset(ResizedDataset):
    def __init__(self, root_path, train=True):

        self.root_path = root_path
        self.train = train
        self.num_classes = 3

        if self.train:
            data_path = os.path.join(self.root_path, "training")
        else:
            data_path = os.path.join(self.root_path, "validation")

        batch_folders = list(os.listdir(data_path))

        images = []
        targets = []
        for folder_name in batch_folders:
            images_root_path = os.path.join(data_path, folder_name, "images")
            targets_root_path = os.path.join(data_path, folder_name, "targets")

            images_paths = list(sorted(os.listdir(images_root_path)))
            targets_paths = list(sorted(os.listdir(targets_root_path)))

            for image_path, target_path in zip(images_paths, targets_paths):
                images.append(os.path.join(images_root_path, image_path))
                targets.append(os.path.join(targets_root_path, target_path))

        super().__init__(images, targets)
