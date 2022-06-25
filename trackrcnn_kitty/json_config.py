import json
import sys


class JSONConfig:

    def __init__(self, path):
        f = open(path)
        data = json.load(f)

        # Load dataset parameters
        self.dataset = data.get("dataset")
        self.dataset_path = data.get("dataset_path")
        self.train_image_size = data.get("train_image_size", [309, 1024])
        self.test_image_size = data.get("test_image_size", [375, 1242])
        self.fixed_image_size = data.get("fixed_image_size", False)
        self.transforms_list = data.get("transforms_list", [])

        self.train_batch_size = data["train_batch_size"]
        self.test_batch_size = data["test_batch_size"]
        self.learning_rate = data["learning_rate"]
        self.num_epochs = data["num_epochs"]

        self.add_associations = data.get("add_associations", False)
        self.weights_path = data.get("weights_path", None)
        self.preprocess_weights = data.get("preprocess_weights", False)
        self.pytorch_pretrained_model = data.get("pytorch_pretrained_model", True)
        self.epochs_to_validate = data.get("epochs_to_validate", 5)

        self.use_resnet101 = data.get("use_resnet101", True)
        self.trainable_backbone_layers = data.get("trainable_backbone_layers", 3)
        self.freeze_batchnorm = data.get("freeze_batchnorm", True)
        self.add_last_layer = data.get("add_last_layer", True)
        self.fpn_out_channels = data.get("fpn_out_channels", 256)
        self.pretrained_backbone = data.get("pretrained_backbone", False)

        # task can be 1. train, 2.val, 3.train+val 4.save_preds
        self.task = data.get("task")
        self.maskrcnn_params = data.get("maskrcnn_params", {})

    @staticmethod
    def get_instance(path):
        try:
            json_config = JSONConfig(path)
            return json_config
        except KeyError as e:
            print("Mandatory parameter is missing from config file!")
            print(e)
            sys.exit(-1)
