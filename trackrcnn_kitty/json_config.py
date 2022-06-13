import json
import sys


class JSONConfig:

    def __init__(self, path):
        f = open(path)
        data = json.load(f)

        self.dataset = data["dataset"]
        self.dataset_path = data["dataset_path"]
        self.image_size = data["image_size"]
        self.transforms_list = data["image_transforms"]

        self.train_batch_size = data["train_batch_size"]
        self.test_batch_size = data["test_batch_size"]
        self.learning_rate = data["learning_rate"]
        self.num_epochs = data["num_epochs"]
        self.momentum = data["momentum"]
        self.weight_decay = data["weight_decay"]
        self.add_lr_scheduler = data["add_lr_scheduler"]

        self.add_associations = data["add_associations"]
        self.weights_path = data["weights_path"]

        self.use_resnet_101 = data["use_resnet_101"]
        self.train_last_layer = data["train_last_layer"]

        # task can be 1. train, 2.val, 3.train+val 4.save_preds
        self.task = data["task"]

    @staticmethod
    def get_instance(path):
        try:
            json_config = JSONConfig(path)
            return json_config
        except KeyError as e:
            print(e)
            sys.exit(-1)
