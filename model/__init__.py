import torch

# global constants
NO_WORD = -1
NO_LINK_TYPE = -1
MODEL_ID = "facebook/bart-large-cnn"
DATASET_ID = "abisee/cnn_dailymail"
DATASET_VERSION = "3.0.0"
TEST_SPLIT = "test"
MAX_INPUT_LENGTH = 512
MAX_DISTANCE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

__all__ = [
        "NO_WORD", 
        "NO_LINK_TYPE",
        "MODEL_ID",
        "DATASET_ID",
        "DATASET_VERSION",
        "MAX_INPUT_LENGTH",
        "MAX_DISTANCE",
        "DEVICE",
        ]

