"""Loading the saved model and performing inference."""

from pathlib import Path

import torch

from . import config
from .model import SignLanguageCNN
import sys

# A bridge to prevent the 'AttributeError' encountered when loading models 
# externally that were trained and saved in a Jupyter Notebook ('__main__'):
setattr(sys.modules['__main__'], 'SignLanguageCNN', SignLanguageCNN)


def load_trained_model(checkpoint_path=None, device=None):
    """
    Loads the entire model saved via torch.save(model, ...).
    weights_only=False is required for PyTorch 2.6+.
    """
    path = Path(checkpoint_path or config.CHECKPOINT_PATH)
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Place the file under the package root or "
            "set the SIGN_LANGUAGE_CHECKPOINT environment variable."
        )

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)

    model = model.to(device)
    model.eval()
    return model
