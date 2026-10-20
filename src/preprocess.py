import zlib
from pathlib import Path

import librosa
import numpy as np

SAMPLE_RATE = 22050

# Waveform augmentations applied (offline) to training actors only.
# Copy k of a clip uses AUGMENTATIONS[k % len(AUGMENTATIONS)].
AUGMENTATIONS = ["pitch", "stretch", "noise", "pitch_noise"]


def load_audio(file_path, sample_rate=SAMPLE_RATE):
    """
    Load and preprocess a speech recording.

    Returns
    -------
    audio : numpy.ndarray
        One-dimensional, silence-trimmed audio signal.
    sample_rate : int
        Sample rate of the returned signal.
    """
    file_path = Path(file_path)

    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)

    if audio.size == 0:
        raise ValueError(f"Loaded audio is empty: {file_path}")

    if not np.isfinite(audio).all():
        raise ValueError(f"Loaded audio contains non-finite values: {file_path}")

    trimmed_audio, _ = librosa.effects.trim(audio, top_db=25)
    if trimmed_audio.size > 0:
        audio = trimmed_audio

    return audio, sr


# ----------------------------------------------------------------------
# Augmentation
# ----------------------------------------------------------------------

def make_rng(key):
    """Deterministic random generator derived from a string key."""
    return np.random.default_rng(zlib.crc32(key.encode("utf-8")))


def add_noise(audio, rng, snr_db_range=(15.0, 30.0)):
    """Add Gaussian noise at a random signal-to-noise ratio."""
    snr_db = rng.uniform(*snr_db_range)
    signal_power = float(np.mean(audio ** 2)) + 1e-10
    noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape)
    return (audio + noise).astype(np.float32)


def shift_pitch(audio, sample_rate, rng, min_steps=0.5, max_steps=2.0):
    """Shift pitch up or down by 0.5-2 semitones (kept mild on purpose)."""
    steps = rng.choice([-1.0, 1.0]) * rng.uniform(min_steps, max_steps)
    return librosa.effects.pitch_shift(
        audio, sr=sample_rate, n_steps=float(steps)
    ).astype(np.float32)


def stretch_time(audio, rng, low=0.9, high=1.1):
    """Speed the clip up or slow it down by up to 10%."""
    rate = rng.uniform(low, high)
    return librosa.effects.time_stretch(audio, rate=float(rate)).astype(np.float32)


def augment_audio(audio, sample_rate, kind, rng):
    """Apply one named augmentation to a waveform."""
    audio = np.asarray(audio, dtype=np.float32)

    if kind == "pitch":
        return shift_pitch(audio, sample_rate, rng)
    if kind == "stretch":
        return stretch_time(audio, rng)
    if kind == "noise":
        return add_noise(audio, rng)
    if kind == "pitch_noise":
        return add_noise(shift_pitch(audio, sample_rate, rng), rng)

    raise ValueError(f"Unknown augmentation: {kind}")
