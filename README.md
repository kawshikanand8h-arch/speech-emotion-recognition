# Speech Emotion Recognition (RAVDESS, 7 classes)

neutral, happy, sad, angry, fearful, disgust, surprised.
Speaker-independent: actors never appear in both training and evaluation data.

## Quickstart: Universal Emotion Prediction (Real Voices & Web UI)

### 1. Launch the Interactive Web App (Live Microphone & Audio Upload)
```bash
python app.py
```
Open `http://localhost:7860` in your browser. Record directly from your microphone or upload any audio file.

### 2. Predict Any Audio File via CLI
```bash
# Uses the spontaneous in-the-wild MSP-Podcast model (robust across any speaker, mic, or pitch)
python -m src.predict path/to/recording.wav

# Optional: Predict using the WavLM multi-corpus model or BiLSTM
python -m src.predict path/to/recording.wav --mode wavlm
python -m src.predict path/to/recording.wav --mode bilstm
```

---

## Model Training & Research Pipeline (run from the project root)

```bash
python -m src.prepare_dataset       # data/raw -> data/processed/metadata.csv (multi-corpus: RAVDESS + CREMA-D)
python -m src.split_dataset         # adds 5 actor-wise CV folds, keeps held-out test actors
python -m src.extract_features      # features + augmented copies -> features_v2/ + index csv
python -m src.cross_validate        # regularized BiLSTM cross-validation benchmark
python -m src.train_final           # final BiLSTM ensemble
python -m src.evaluate              # evaluate BiLSTM on held-out test actors
```

### Foundation Model Embeddings (WavLM Base+)
```bash
python -m src.ssl_embeddings --model microsoft/wavlm-base-plus
python -m src.train_ssl --model microsoft/wavlm-base-plus
```

See `CHANGES.md` and walkthrough documentation for detailed architecture and evaluation logs.