import sys
import os

from trackrcnn_kitty.json_config import JSONConfig
from trackrcnn_kitty.train_engine import TrainEngine

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Config file path is missing"
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
        train_engine.evaluate_and_save_results(os.path.join("predictions", os.path.basename(config.dataset_path)))
    else:
        print("Invalid task in configuration file! Stopping program.")
        sys.exit(-1)



