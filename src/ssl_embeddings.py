"""OPTIONAL (step 6): extract frozen self-supervised speech embeddings.

Needs `pip install transformers` and internet access the first time (the
model is downloaded from Hugging Face). A GPU makes it much faster.

    python -m src.ssl_embeddings                       # WavLM base+
    python -m src.ssl_embeddings --model facebook/wav2vec2-base

For every recording it saves an array of shape (num_layers + 1, hidden):
the mean over time of every transformer layer's output.
"""
import argparse
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata.csv"
SSL_DIR = PROJECT_ROOT / "data" / "processed" / "ssl"
SSL_SAMPLE_RATE = 16000


def main():
    from transformers import AutoFeatureExtractor, AutoModel

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = args.model.replace("/", "_")
    out_root = SSL_DIR / name

    extractor = AutoFeatureExtractor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    metadata = pd.read_csv(METADATA_PATH)
    rows = []

    for done, (_, row) in enumerate(metadata.iterrows(), start=1):
        audio, _ = librosa.load(PROJECT_ROOT / row["file_path"],
                                sr=SSL_SAMPLE_RATE, mono=True)
        audio, _ = librosa.effects.trim(audio, top_db=25)

        inputs = extractor(audio, sampling_rate=SSL_SAMPLE_RATE,
                           return_tensors="pt")
        with torch.no_grad():
            hidden = model(
                inputs["input_values"].to(device), output_hidden_states=True
            ).hidden_states  # tuple of (1, time, hidden)

        embedding = torch.stack([h[0].mean(dim=0) for h in hidden]).cpu().numpy()

        actor_dir = out_root / f"Actor_{int(row['actor_id']):02d}"
        actor_dir.mkdir(parents=True, exist_ok=True)
        path = actor_dir / (Path(row["file_path"]).stem + ".npy")
        np.save(path, embedding.astype(np.float32))

        rows.append({**row.to_dict(),
                     "ssl_path": str(path.relative_to(PROJECT_ROOT))})

        if done % 50 == 0 or done == len(metadata):
            print(f"Processed {done}/{len(metadata)} recordings")

    pd.DataFrame(rows).to_csv(out_root / "index.csv", index=False)
    print(f"Saved embeddings and index under {out_root}")


if __name__ == "__main__":
    main()
