import librosa
import numpy as np

N_MFCC = 40
N_MELS = 64
N_FFT = 1024
HOP_LENGTH = 256          # ~11.6 ms at 22.05 kHz (was 512)

# Feature layout per frame:
#   [ 0: 40]  MFCC
#   [40: 80]  delta MFCC
#   [80:120]  delta-delta MFCC
#   [120]     log RMS energy
#   [121]     zero-crossing rate
#   [122]     log F0 (YIN pitch estimate)
FEATURE_DIM = N_MFCC * 3 + 3

F0_MIN = 65.0
F0_MAX = 500.0
YIN_FRAME_LENGTH = 2048
MIN_SAMPLES = 4096        # YIN and delta need a minimum amount of audio


def extract_features(audio, sample_rate):
    """
    Convert a waveform into a (time_steps, FEATURE_DIM) feature matrix:
    MFCC + deltas + energy + zero-crossing rate + pitch.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Audio must be a non-empty, one-dimensional array.")
    if sample_rate <= 0:
        raise ValueError("Sample rate must be positive.")

    if audio.size < MIN_SAMPLES:
        audio = np.pad(audio, (0, MIN_SAMPLES - audio.size))

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sample_rate, n_mfcc=N_MFCC,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    delta1 = librosa.feature.delta(mfcc, order=1, mode="nearest")
    delta2 = librosa.feature.delta(mfcc, order=2, mode="nearest")

    rms = librosa.feature.rms(
        y=audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )
    log_rms = np.log(rms + 1e-6)

    zcr = librosa.feature.zero_crossing_rate(
        audio, frame_length=N_FFT, hop_length=HOP_LENGTH
    )

    f0 = librosa.yin(
        audio, fmin=F0_MIN, fmax=F0_MAX, sr=sample_rate,
        frame_length=YIN_FRAME_LENGTH, hop_length=HOP_LENGTH,
    )
    log_f0 = np.log(f0)[np.newaxis, :]

    n_frames = min(
        mfcc.shape[1], log_rms.shape[1], zcr.shape[1], log_f0.shape[1]
    )

    stacked = np.concatenate(
        [
            mfcc[:, :n_frames],
            delta1[:, :n_frames],
            delta2[:, :n_frames],
            log_rms[:, :n_frames],
            zcr[:, :n_frames],
            log_f0[:, :n_frames],
        ],
        axis=0,
    )

    return stacked.T.astype(np.float32)
