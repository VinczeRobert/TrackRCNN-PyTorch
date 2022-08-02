import sys
import os

from trackrcnn_kitty.json_config import JSONConfig
from train_engine import TrainEngine

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Config file path is missing"
    assert os.path.exists(sys.argv[1]), "Config file is invalid"

    config = JSONConfig.get_instance(sys.argv[1])
    train_engine = TrainEngine(config)

    if config.task == "train":
        train_engine.training()
    elif config.task == "val":
        train_engine.evaluate()
    elif config.task == "train+val":
        train_engine.training_and_evaluating()
    elif config.task == "save_preds":
        train_engine.save_bounding_box_results(os.path.join("predictions", os.path.basename(config.dataset_path)))
    elif config.task == "annotate":
        train_engine.annotate_results_with_tracking()
    elif config.task == "metrics":
        train_engine.calculate_metrics()
    else:
        print("Invalid task in configuration file! Stopping program.")
        sys.exit(-1)
