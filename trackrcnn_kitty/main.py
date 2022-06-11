import sys
import os

from trackrcnn_kitty.train_engine import TrainEngine

if __name__ == '__main__':
    assert len(sys.argv) == 2, "Config file path is missing"
    assert os.path.exists(sys.argv[1]), "Config file is invalid"

    train_engine = TrainEngine(config_path=sys.argv[1])
    train_engine.run_training()
