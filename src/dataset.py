from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from src.features import FEATURE_DIM

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDEX_PATH = PROJECT_ROOT / "data" / "processed" / "features_v2_index.csv"

EMOTION_TO_INDEX = {
    "neutral": 0,
    "happy": 1,
    "sad": 2,
    "angry": 3,
    "fearful": 4,
    "disgust": 5,
    "surprised": 6,
}
INDEX_TO_EMOTION = {index: emotion for emotion, index in EMOTION_TO_INDEX.items()}


# ----------------------------------------------------------------------
# Which recordings go where
# ----------------------------------------------------------------------

def load_index():
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"{INDEX_PATH} not found. Run `python -m src.split_dataset` and "
            "`python -m src.extract_features` first."
        )
    return pd.read_csv(INDEX_PATH)


def fold_frames(index, fold, use_aug=True):
    """Train on every development fold except `fold` (originals + augmented
    copies); validate on the original clips of `fold` only."""
    development = index[index["fold"] >= 0]

    train = development[development["fold"] != fold]
    if not use_aug:
        train = train[train["is_aug"] == 0]

    validation = development[
        (development["fold"] == fold) & (development["is_aug"] == 0)
    ]
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def final_train_frame(index, use_aug=True):
    """All 20 development actors, for the final model."""
    development = index[index["fold"] >= 0]
    if not use_aug:
        development = development[development["is_aug"] == 0]
    return development.reset_index(drop=True)


def test_frame(index):
    return index[(index["split"] == "test") & (index["is_aug"] == 0)].reset_index(
        drop=True
    )


# ----------------------------------------------------------------------
# Normalisation
# ----------------------------------------------------------------------

def compute_global_stats(arrays):
    """Per-feature mean/std over all frames of the given training arrays."""
    total = sum(a.shape[0] for a in arrays)
    feature_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    squared_sum = np.zeros(FEATURE_DIM, dtype=np.float64)

    for a in arrays:
        a = a.astype(np.float64)
        feature_sum += a.sum(axis=0)
        squared_sum += np.square(a).sum(axis=0)

    mean = feature_sum / total
    std = np.sqrt(np.maximum(squared_sum / total - np.square(mean), 0.0))
    std[std < 1e-8] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def normalize_features(features, mode, stats=None):
    """
    mode = "utterance": standardise each clip with its OWN mean/std per
        feature. Removes speaker/recording offsets and needs no saved stats,
        so it also works for a single new recording.
    mode = "global": standardise with training-set mean/std.
    """
    features = features.astype(np.float32)

    if mode == "utterance":
        mean = features.mean(axis=0, keepdims=True)
        std = features.std(axis=0, keepdims=True)
        std[std < 1e-6] = 1.0
        return (features - mean) / std

    if mode == "global":
        if stats is None:
            raise ValueError("Global normalisation needs (mean, std).")
        mean, std = stats
        return (features - mean) / std

    raise ValueError(f"Unknown normalisation mode: {mode}")


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

class SpeechEmotionDataset(Dataset):
    def __init__(
        self,
        frame,
        normalization="utterance",
        stats=None,
        train=False,
        specaugment=False,
        seed=0,
        noise_std=0.0,
        time_crop=False,
    ):
        if frame.empty:
            raise ValueError("No recordings in this dataset.")

        self.frame = frame.reset_index(drop=True)
        self.normalization = normalization
        self.stats = stats
        self.train = train
        self.specaugment = specaugment and train
        self.noise_std = noise_std if train else 0.0
        self.time_crop = time_crop and train
        self.rng = np.random.default_rng(seed)

        # Features are small; keep them all in RAM for speed.
        self.raw = [
            np.load(PROJECT_ROOT / path) for path in self.frame["feature_path"]
        ]
        self.labels = [
            EMOTION_TO_INDEX[e] for e in self.frame["emotion"]
        ]

        # "speaker" mode: per-actor mean/std from that actor's own ORIGINAL
        # clips (no labels used), so it is also valid for unseen actors.
        self.actor_stats = {}
        if normalization == "speaker":
            self.actor_ids = self.frame["actor_id"].astype(int).tolist()
            originals = (
                self.frame["is_aug"].to_numpy() == 0
                if "is_aug" in self.frame.columns
                else np.ones(len(self.frame), dtype=bool)
            )
            by_actor = {}
            for raw, actor, is_orig in zip(self.raw, self.actor_ids, originals):
                if is_orig:
                    by_actor.setdefault(actor, []).append(raw)
            for actor, arrays in by_actor.items():
                self.actor_stats[actor] = compute_global_stats(arrays)

    def __len__(self):
        return len(self.frame)

    def _spec_augment(self, features):
        """Randomly zero (= mean after normalisation) a few time spans and
        feature bands so the model cannot lean on any single detail."""
        n_frames, n_features = features.shape
        features = features.copy()

        for _ in range(3):  # time masks (up to ~20% of the clip each)
            width = int(self.rng.integers(0, max(2, min(20, n_frames // 5))))
            if width > 0:
                start = int(self.rng.integers(0, max(1, n_frames - width)))
                features[start:start + width, :] = 0.0

        for _ in range(3):  # feature masks
            width = int(self.rng.integers(0, 13))
            if width > 0:
                start = int(self.rng.integers(0, n_features - width))
                features[:, start:start + width] = 0.0

        return features

    def _random_crop(self, features):
        """Keep a random contiguous 70-100% window of the clip."""
        n_frames = features.shape[0]
        keep = int(n_frames * self.rng.uniform(0.7, 1.0))
        keep = max(keep, min(n_frames, 8))
        start = int(self.rng.integers(0, n_frames - keep + 1))
        return features[start:start + keep]

    def __getitem__(self, index):
        if self.normalization == "speaker":
            features = normalize_features(
                self.raw[index], "global",
                self.actor_stats[self.actor_ids[index]],
            )
        else:
            features = normalize_features(
                self.raw[index], self.normalization, self.stats
            )

        if self.time_crop:
            features = self._random_crop(features)

        if self.specaugment:
            features = self._spec_augment(features)

        if self.noise_std > 0:
            features = features + self.rng.normal(
                0.0, self.noise_std, size=features.shape
            ).astype(np.float32)

        return (
            torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32)),
            torch.tensor(self.labels[index], dtype=torch.long),
        )


def collate_fn(batch):
    """Pad variable-length sequences so they fit into a batch."""
    features, labels = zip(*batch)

    lengths = torch.tensor([s.shape[0] for s in features], dtype=torch.long)
    padded = pad_sequence(features, batch_first=True, padding_value=0.0)
    labels = torch.stack(labels)

    return padded, labels, lengths
