import sys
import os

from json_config import JSONConfig
from trackrcnn_pytorch_engine import TrackRCNNPyTorchEngine

if __name__ == '__main__':
    assert len(sys.argv) >= 2, "Config file path is missing"
    assert os.path.exists(sys.argv[1]), "Config file is invalid"

    config = JSONConfig.get_instance(sys.argv[1])
    train_engine = TrackRCNNPyTorchEngine(config)

    if config.task == "train":
        train_engine.train()
    elif config.task == "val":
        train_engine.evaluate()
    elif config.task == "train+val":
        train_engine.train_and_evaluate()
    elif config.task == "save_preds":
        train_engine.save_bounding_box_results(config.dataset_path)
    elif config.task == "save_preds_coco":
        train_engine.forward_predictions_for_tracking(config.sequence_number)
    elif config.task == "annotate_seq":
        if config.add_associations:
            train_engine.annotate_results_with_tracking(config.sequence_number)
        else:
            train_engine.annotate_results_without_tracking()
    elif config.task == "metrics":
        train_engine.calculate_metrics()
    else:
        print("Invalid task in configuration file! Stopping program.")
        sys.exit(-1)
