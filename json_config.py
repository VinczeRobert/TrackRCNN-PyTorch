import json
import sys


class JSONConfig:

    def __init__(self, path):
        f = open(path)
        data = json.load(f)

        # Load dataset parameters
        self.dataset = data.get("dataset")
        self.dataset_path = data.get("dataset_path")
        self.all_classes = data.get("all_classes", True)
        self.num_workers = data.get("num_workers", 4)
        self.sequence_number = data.get("sequence_number", None)

        # task can be 1. train, 2. val, 3. train+val, 4. save_preds, 5. annotate, 6. metrics
        self.task = data.get("task")

        self.train_image_size = data.get("train_image_size", [309, 1024])
        self.test_image_size = data.get("test_image_size", [375, 1242])
        self.fixed_image_size = data.get("fixed_image_size", False)

        self.transforms_list = data.get("transforms_list", [])

        self.train_batch_size = data["train_batch_size"]
        self.test_batch_size = data["test_batch_size"]
        self.learning_rate = data["learning_rate"]
        self.num_epochs = data["num_epochs"]
        self.epochs_to_validate = data.get("epochs_to_validate", 5)
        self.optimizer_parameters = data.get("optimizer_parameters", None)
        self.lr_scheduler_step_size = data.get("lr_scheduler_step_size", None)
        self.lr_scheduler_gamma = data.get("lr_scheduler_gamma", None)

        self.load_weights = data.get("load_weights", False)
        self.weights_path = data.get("weights_path", None)
        self.pretrain_only_backbone = data.get("pretrain_only_backbone", False)
        self.num_pretrained_classes = data.get("num_pretrained_classes", 91)
        self.epochs_to_save_weights = data.get("epochs_to_save_weights", 1)
        self.saved_weights_name = data.get("saved_weights_name", None)

        self.use_resnet101 = data.get("use_resnet101", True)
        self.trainable_backbone_layers = data.get("trainable_backbone_layers", 3)
        self.freeze_batchnorm = data.get("freeze_batchnorm", True)
        self.add_last_layer = data.get("add_last_layer", True)
        self.fpn_out_channels = data.get("fpn_out_channels", 256)

        self.maskrcnn_params = data.get("maskrcnn_params", {})

        self.add_associations = data.get("add_associations", False)
        if self.add_associations:
            self.confidence_threshold_car = data.get("confidence_threshold_car")
            self.reid_weight_car = data.get("reid_weight_car")
            self.association_threshold_car = data.get("association_threshold_car")
            self.keep_alive_car = data.get("keep_alive_car")
            self.reid_euclidean_offset_car = data.get("reid_euclidean_offset_car")
            self.reid_euclidean_scale_car = data.get("reid_euclidean_scale_car")
            self.confidence_threshold_pedestrian = data.get("confidence_threshold_pedestrian")
            self.reid_weight_pedestrian = data.get("reid_weight_pedestrian")
            self.association_threshold_pedestrian = data.get("association_threshold_pedestrian")
            self.keep_alive_pedestrian = data.get("keep_alive_pedestrian")
            self.reid_euclidean_offset_pedestrian = data.get("reid_euclidean_offset_pedestrian")
            self.reid_euclidean_scale_pedestrian = data.get("reid_euclidean_scale_pedestrian")

    @staticmethod
    def get_instance(path):
        try:
            json_config = JSONConfig(path)
            return json_config
        except KeyError as e:
            print("Mandatory parameter is missing from config file!")
            print(e)
            sys.exit(-1)
