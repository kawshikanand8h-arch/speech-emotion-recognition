"""Predict the emotion of one WAV file.

    python -m src.predict path/to/recording.wav
    python -m src.predict path/to/recording.wav --mode bilstm
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import torch
import librosa

from src.dataset import INDEX_TO_EMOTION, PROJECT_ROOT
from src.ensemble import load_checkpoints, predict_features
from src.features import extract_features
from src.preprocess import load_audio

CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"


EMOTION_CENTROIDS = {
    "angry": np.array([0.56, 0.58, 0.19]),
    "happy": np.array([0.42, 0.44, 0.56]),
    "sad": np.array([0.16, 0.23, 0.25]),
    "surprised": np.array([0.34, 0.42, 0.54]),
    "neutral": np.array([0.28, 0.35, 0.40]),
    "fearful": np.array([0.55, 0.22, 0.20]),
    "disgust": np.array([0.35, 0.38, 0.20]),
}

_ROBUST_CACHE = {}


def get_robust_model():
    """Load and cache the verified MSP-Podcast in-the-wild model with aligned head weights."""
    if "model" in _ROBUST_CACHE:
        return _ROBUST_CACHE["model"], _ROBUST_CACHE["extractor"], _ROBUST_CACHE["device"]

    import torch
    import torch.nn as nn
    from transformers import AutoFeatureExtractor, Wav2Vec2Model
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    weights = load_file(hf_hub_download(model_name, "model.safetensors"))

    class MSPRobustModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
            self.dense = nn.Linear(1024, 1024)
            self.dense.weight.data.copy_(weights["classifier.dense.weight"])
            self.dense.bias.data.copy_(weights["classifier.dense.bias"])
            self.out_proj = nn.Linear(1024, 3)
            self.out_proj.weight.data.copy_(weights["classifier.out_proj.weight"])
            self.out_proj.bias.data.copy_(weights["classifier.out_proj.bias"])

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values).last_hidden_state
            pooled = torch.mean(hidden, dim=1)
            x = torch.tanh(self.dense(pooled))
            return self.out_proj(x)

    model = MSPRobustModel().to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    _ROBUST_CACHE["model"] = model
    _ROBUST_CACHE["extractor"] = extractor
    _ROBUST_CACHE["device"] = device
    return model, extractor, device


def predict_robust(wav_path):
    """Universal in-the-wild prediction for any voice, microphone, or accent."""
    model, extractor, device = get_robust_model()

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio / peak * 0.9

    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) >= int(0.5 * 16000):
        audio = trimmed

    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        out = model(inputs["input_values"].to(device))[0].cpu().numpy()

    arousal, dominance, valence = float(out[0]), float(out[1]), float(out[2])
    dim_vec = np.array([arousal, dominance, valence])

    # Soft-min distance to centroids in (Arousal, Dominance, Valence) space
    dists = {emo: float(np.linalg.norm(dim_vec - c)) for emo, c in EMOTION_CENTROIDS.items()}
    unnorm = {emo: float(np.exp(-d * 14.0)) for emo, d in dists.items()}
    total = sum(unnorm.values())
    probabilities = {emo: unnorm[emo] / total for emo in unnorm}

    dim_metrics = {
        "Arousal (Energy/Intensity)": f"{arousal:.3f}",
        "Dominance (Assertiveness)": f"{dominance:.3f}",
        "Valence (Positivity)": f"{valence:.3f}",
    }
    return probabilities, dim_metrics


def predict_wavlm(wav_path, payload):
    from transformers import AutoFeatureExtractor, AutoModel
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = payload["model_name"]
    layers = payload["layers"]
    scaler = payload["scaler"]
    clf = payload["classifier"]

    extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    audio, _ = librosa.load(wav_path, sr=16000, mono=True)
    audio = audio - np.mean(audio)
    peak = np.max(np.abs(audio))
    if peak > 1e-4:
        audio = audio / peak * 0.9

    trimmed, _ = librosa.effects.trim(audio, top_db=25)
    if len(trimmed) >= int(0.5 * 16000):
        audio = trimmed

    inputs = extractor(audio, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        hidden = model(
            inputs["input_values"].to(device), output_hidden_states=True
        ).hidden_states

    embedding = torch.stack([h[0].mean(dim=0) for h in hidden]).cpu().numpy()
    feat = embedding[layers, :].mean(axis=0, keepdims=True)
    feat_scaled = scaler.transform(feat)
    
    probs = clf.predict_proba(feat_scaled)[0]
    return {INDEX_TO_EMOTION[i]: float(probs[i]) for i in range(len(probs))}, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wav_path", help="path to audio file (.wav, .mp3, etc.)")
    parser.add_argument("--mode", choices=["robust", "wavlm", "bilstm"], default="robust",
                        help="prediction engine: robust (default, in-the-wild), wavlm, or bilstm")
    args = parser.parse_args()

    dim_info = None
    if args.mode == "robust":
        print("Using Universal In-The-Wild Foundation Model (Wav2Vec2 MSP-Podcast)...")
        probabilities, dim_info = predict_robust(args.wav_path)
    elif args.mode == "wavlm":
        ssl_path = CHECKPOINT_DIR / "ssl_classifier.joblib"
        if not ssl_path.exists():
            raise FileNotFoundError(f"{ssl_path} not found. Run `python -m src.train_ssl` first.")
        print(f"Using Studio WavLM model from {ssl_path}...")
        payload = joblib.load(ssl_path)
        probabilities, _ = predict_wavlm(args.wav_path, payload)
    else:
        print("Using BiLSTM acoustic ensemble...")
        models = load_checkpoints()
        audio, sample_rate = load_audio(args.wav_path)
        raw_probs = predict_features(models, extract_features(audio, sample_rate))
        probabilities = {INDEX_TO_EMOTION[i]: float(raw_probs[i]) for i in range(len(raw_probs))}

    ranked = sorted(probabilities.items(), key=lambda item: -item[1])
    print(f"\n==========================================")
    print(f"  PREDICTED EMOTION: {ranked[0][0].upper()} ({ranked[0][1]:.1%})")
    print(f"==========================================\n")
    print("Class Probabilities:")
    for emotion, probability in ranked:
        bar = "█" * int(probability * 30)
        print(f"  {emotion:<10} {probability:6.1%}  {bar}")

    if dim_info:
        print("\nAffect Dimensions:")
        for k, v in dim_info.items():
            print(f"  {k:<30} {v}")


if __name__ == "__main__":
    main()
