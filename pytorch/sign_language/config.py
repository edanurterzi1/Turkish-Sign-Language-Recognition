"""Paths and training hyperparameters."""

import os

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DATA_ROOT = os.environ.get(
    "SIGN_LANGUAGE_DATA_ROOT",
    "../data",
)

MODELS_DIR = "pytorch/models"
os.makedirs(MODELS_DIR, exist_ok=True)
_DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "sign_language_model.pth")
CHECKPOINT_PATH = os.environ.get("SIGN_LANGUAGE_CHECKPOINT", _DEFAULT_CHECKPOINT)

BATCH_SIZE = 32
TRAIN_VAL_SPLIT = 0.8
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = (64, 64)
