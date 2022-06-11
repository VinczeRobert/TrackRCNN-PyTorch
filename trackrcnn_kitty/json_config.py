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
        self.shuffle = data["shuffle"]

        self.batch_size = data["batch_size"]
        self.learning_rate = data["learning_rate"]
        self.num_epochs = data["num_epochs"]
        self.momentum = data["momentum"]
        self.weight_decay = data["weight_decay"]
        self.add_lr_scheduler = data["add_lr_scheduler"]

        self.add_associations = data["add_associations"]
        self.weights_path = data["weights_path"]

    @staticmethod
    def get_instance(path):
        try:
            json_config = JSONConfig(path)
            return json_config
        except KeyError as e:
            print(e)
            sys.exit(-1)
