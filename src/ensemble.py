"""Loading and running the saved ensemble."""
import glob

import numpy as np
import torch

from src.dataset import PROJECT_ROOT, normalize_features
from src.models import build_model

CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"


def load_checkpoints(pattern="final_seed*.pt", device="cpu"):
    paths = sorted(glob.glob(str(CHECKPOINT_DIR / pattern)))
    if not paths:
        raise FileNotFoundError(
            f"No checkpoints matching {pattern} in {CHECKPOINT_DIR}. "
            "Run `python -m src.train_final` first."
        )

    models = []
    for path in paths:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        model = build_model(checkpoint["model_config"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        stats = None
        if checkpoint["normalization"] == "global":
            stats = (checkpoint["norm_mean"], checkpoint["norm_std"])

        models.append((path, model, checkpoint["normalization"], stats))

    return models


@torch.no_grad()
def predict_features(models, features, device="cpu"):
    """Average softmax over the ensemble for one (time, features) matrix."""
    probabilities = []
    for _, model, normalization, stats in models:
        if normalization == "speaker":
            # Needs several clips of the same speaker; a single file has no
            # speaker statistics, so approximate with its own statistics.
            normalization = "utterance"
        x = normalize_features(features, normalization, stats)
        x = torch.from_numpy(x).unsqueeze(0).to(device)
        lengths = torch.tensor([x.size(1)])
        logits = model(x, lengths)
        probabilities.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    return np.mean(probabilities, axis=0)
