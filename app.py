"""Speech Emotion Recognition test bench.

    python app.py

Tabs: single clip analysis, engine comparison, batch testing with metrics,
and a session history.
"""
import glob
import inspect
import tempfile
from datetime import datetime
from pathlib import Path

import gradio as gr
import librosa
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_ROOT / "data" / "processed" / "checkpoints"
WAVLM_PATH = CHECKPOINT_DIR / "ssl_classifier.joblib"

EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
LOOK = {
    "neutral": ("😐", "#94a3b8"),
    "happy": ("😄", "#facc15"),
    "sad": ("😢", "#60a5fa"),
    "angry": ("😠", "#f87171"),
    "fearful": ("😨", "#a78bfa"),
    "disgust": ("🤢", "#4ade80"),
    "surprised": ("😲", "#fb923c"),
}

ENGINES = {
    "robust": "Wav2Vec2 in-the-wild (arousal / dominance / valence)",
    "wavlm": "WavLM embeddings + trained classifier",
    "bilstm": "BiLSTM ensemble (MFCC features)",
}

RAVDESS = {"01": "neutral", "03": "happy", "04": "sad", "05": "angry",
           "06": "fearful", "07": "disgust", "08": "surprised"}
CREMA = {"NEU": "neutral", "HAP": "happy", "SAD": "sad", "ANG": "angry",
         "FEA": "fearful", "DIS": "disgust"}
SUFFIX = {"neu": "neutral", "lgh": "happy", "hap": "happy", "sd": "sad",
          "ang": "angry", "fear": "fearful", "dis": "disgust", "surp": "surprised"}

_cache = {}


def engine_available(name):
    if name == "wavlm":
        return WAVLM_PATH.exists()
    if name == "bilstm":
        return bool(glob.glob(str(CHECKPOINT_DIR / "final_seed*.pt")))
    return True


def available_engines():
    return [(label, key) for key, label in ENGINES.items() if engine_available(key)]


def run_engine(engine, path):
    """Returns (probabilities over EMOTIONS, dimension dict or None)."""
    if engine == "robust":
        from src.predict import predict_robust
        return predict_robust(path)

    if engine == "wavlm":
        import joblib
        from src.predict import predict_wavlm
        if "wavlm" not in _cache:
            _cache["wavlm"] = joblib.load(WAVLM_PATH)
        probs, _ = predict_wavlm(path, _cache["wavlm"])
        return probs, None

    from src.dataset import INDEX_TO_EMOTION
    from src.ensemble import load_checkpoints, predict_features
    from src.features import extract_features
    from src.preprocess import load_audio
    if "models" not in _cache:
        _cache["models"] = load_checkpoints()
    audio, sr = load_audio(path)
    raw = predict_features(_cache["models"], extract_features(audio, sr))
    return {INDEX_TO_EMOTION[i]: float(raw[i]) for i in range(len(raw))}, None


def label_from_filename(path):
    stem = Path(path).stem
    parts = stem.split("-")
    if len(parts) == 7 and parts[2] in RAVDESS:
        return RAVDESS[parts[2]]
    parts = stem.split("_")
    if len(parts) == 4 and parts[0].isdigit() and parts[2].upper() in CREMA:
        return CREMA[parts[2].upper()]
    return SUFFIX.get(stem.lower().split("_")[-1])


def ranked(probs):
    return sorted(probs.items(), key=lambda kv: -kv[1])


CSS = """
.card {border:1px solid var(--border-color-primary); border-radius:16px; padding:20px;
       background:var(--background-fill-secondary);}
.hero {display:flex; align-items:center; gap:18px;}
.hero .emoji {font-size:64px; line-height:1;}
.hero .label {font-size:30px; font-weight:700; text-transform:capitalize;}
.hero .sub {opacity:.7; font-size:14px; margin-top:2px;}
.bar-row {display:flex; align-items:center; gap:10px; margin:7px 0; font-size:14px;}
.bar-name {width:104px; text-transform:capitalize;}
.bar-track {flex:1; height:12px; border-radius:99px; background:var(--border-color-primary); overflow:hidden;}
.bar-fill {height:100%; border-radius:99px;}
.bar-val {width:52px; text-align:right; font-variant-numeric:tabular-nums; opacity:.8;}
.dim-row {margin:10px 0;}
.dim-head {display:flex; justify-content:space-between; font-size:14px; margin-bottom:4px;}
.dim-track {position:relative; height:8px; border-radius:99px; background:var(--border-color-primary);}
.dim-dot {position:absolute; top:-4px; width:16px; height:16px; border-radius:50%;
          background:#6366f1; transform:translateX(-50%); border:2px solid var(--background-fill-primary);}
.stat-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:12px;}
.stat {text-align:center; padding:14px; border-radius:12px; border:1px solid var(--border-color-primary);}
.stat b {display:block; font-size:26px;}
.stat span {opacity:.7; font-size:13px;}
.note {opacity:.65; font-size:13px; margin-top:12px;}
"""


def hero_html(probs, engine):
    (top, conf), (second, second_conf) = ranked(probs)[:2]
    emoji, color = LOOK[top]
    return (
        f'<div class="card hero" style="border-left:6px solid {color}">'
        f'<div class="emoji">{emoji}</div><div>'
        f'<div class="label">{top}</div>'
        f'<div class="sub">{conf:.1%} confidence · runner-up {second} ({second_conf:.1%})</div>'
        f'<div class="sub">{ENGINES[engine]}</div></div></div>'
    )


def bars_html(probs):
    rows = ""
    for emotion, p in ranked(probs):
        emoji, color = LOOK[emotion]
        rows += (
            f'<div class="bar-row"><div class="bar-name">{emoji} {emotion}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p*100:.1f}%;background:{color}"></div></div>'
            f'<div class="bar-val">{p:.1%}</div></div>'
        )
    return f'<div class="card">{rows}</div>'


def dims_html(dims):
    if not dims:
        return ""
    rows = ""
    for name, value in dims.items():
        v = float(value)
        pos = min(max(v, 0.0), 1.0) * 100
        rows += (
            f'<div class="dim-row"><div class="dim-head"><span>{name}</span><span>{v:.3f}</span></div>'
            f'<div class="dim-track"><div class="dim-dot" style="left:{pos:.1f}%"></div></div></div>'
        )
    note = ("Emotion scores for this engine are distances to emotion centres in "
            "arousal / dominance / valence space, not a trained 7-class classifier.")
    return f'<div class="card">{rows}<div class="note">{note}</div></div>'


def signal_figure(path):
    audio, sr = librosa.load(path, sr=16000, mono=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.6), sharex=True)
    ax1.plot(np.arange(len(audio)) / sr, audio, color="#6366f1", linewidth=0.6)
    ax1.set_ylabel("amplitude")
    mel = librosa.power_to_db(librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64), ref=np.max)
    ax2.imshow(mel, aspect="auto", origin="lower", cmap="magma",
               extent=[0, len(audio) / sr, 0, 64])
    ax2.set_ylabel("mel band")
    ax2.set_xlabel("seconds")
    for ax in (ax1, ax2):
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.tight_layout()
    return fig, len(audio) / sr


def analyze(path, engine, history):
    empty = ("", "", "", None, history, pd.DataFrame(history))
    if not path:
        gr.Warning("Record or upload a clip first.")
        return empty
    try:
        probs, dims = run_engine(engine, path)
        fig, duration = signal_figure(path)
    except Exception as exc:
        gr.Warning(f"Could not process this clip: {exc}")
        return empty

    top, conf = ranked(probs)[0]
    history = history + [{
        "time": datetime.now().strftime("%H:%M:%S"),
        "file": Path(path).name,
        "engine": engine,
        "prediction": top,
        "confidence": round(conf, 3),
        "seconds": round(duration, 1),
    }]
    return hero_html(probs, engine), bars_html(probs), dims_html(dims), fig, history, pd.DataFrame(history)


def compare(path):
    if not path:
        gr.Warning("Record or upload a clip first.")
        return pd.DataFrame()
    rows = []
    for _, key in available_engines():
        try:
            probs, _ = run_engine(key, path)
        except Exception as exc:
            rows.append({"engine": key, "prediction": f"error: {exc}"})
            continue
        top, conf = ranked(probs)[0]
        row = {"engine": key, "prediction": top, "confidence": round(conf, 3)}
        row.update({e: round(probs.get(e, 0.0), 3) for e in EMOTIONS})
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=EMOTIONS)
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    ax.imshow(cm, cmap="Purples")
    ax.set_xticks(range(len(EMOTIONS)), EMOTIONS, rotation=45, ha="right")
    ax.set_yticks(range(len(EMOTIONS)), EMOTIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    for i in range(len(EMOTIONS)):
        for j in range(len(EMOTIONS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    return fig


def batch_test(files, engine, label_source, progress=gr.Progress()):
    if not files:
        gr.Warning("Upload one or more clips first.")
        return "", None, pd.DataFrame(), None
    paths = [f if isinstance(f, str) else f.name for f in files]
    rows = []
    for path in progress.tqdm(paths, desc="Testing clips"):
        actual = label_from_filename(path) if label_source == "from filename" else label_source
        try:
            probs, _ = run_engine(engine, path)
            top, conf = ranked(probs)[0]
        except Exception as exc:
            top, conf = f"error: {exc}", 0.0
        rows.append({"file": Path(path).name, "actual": actual, "predicted": top,
                     "confidence": round(conf, 3),
                     "correct": None if actual is None else actual == top})
    df = pd.DataFrame(rows)

    csv_path = Path(tempfile.gettempdir()) / "ser_batch_results.csv"
    df.to_csv(csv_path, index=False)

    scored = df[df["actual"].notna() & df["predicted"].isin(EMOTIONS)]
    if scored.empty:
        msg = ('<div class="card">No ground-truth labels found. Name files like RAVDESS / CREMA-D '
               '(or end them with _ang, _sd, _lgh, _surp), or pick a fixed label above.</div>')
        return msg, None, df, str(csv_path)

    acc = accuracy_score(scored["actual"], scored["predicted"])
    f1 = f1_score(scored["actual"], scored["predicted"], labels=EMOTIONS,
                  average="macro", zero_division=0)
    stats = (
        f'<div class="stat-grid">'
        f'<div class="stat"><b>{acc:.1%}</b><span>accuracy</span></div>'
        f'<div class="stat"><b>{f1:.3f}</b><span>macro-F1</span></div>'
        f'<div class="stat"><b>{len(scored)}</b><span>scored clips</span></div>'
        f'<div class="stat"><b>{len(df) - len(scored)}</b><span>skipped</span></div></div>'
    )
    return stats, confusion_figure(scored["actual"], scored["predicted"]), df, str(csv_path)


def build_ui(**blocks_kwargs):
    engines = available_engines()
    default_engine = engines[0][1]
    missing = [ENGINES[k] for k in ENGINES if not engine_available(k)]

    with gr.Blocks(title="Speech Emotion Recognition", **blocks_kwargs) as demo:
        history = gr.State([])

        gr.Markdown("# 🎙️ Speech Emotion Recognition\nTest your model on live recordings, "
                    "compare engines, and score whole folders of clips.")
        if missing:
            gr.Markdown("Not loaded (checkpoint missing): " + ", ".join(missing))

        with gr.Tabs():
            with gr.Tab("Analyze"):
                with gr.Row():
                    with gr.Column(scale=5):
                        audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                         label="Record or upload")
                        engine = gr.Radio(engines, value=default_engine, label="Engine")
                        run = gr.Button("Analyze", variant="primary")
                        signal = gr.Plot(label="Waveform and mel spectrogram")
                    with gr.Column(scale=5):
                        hero = gr.HTML()
                        bars = gr.HTML()
                        dims = gr.HTML()

            with gr.Tab("Compare engines"):
                c_audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                                   label="Record or upload")
                c_run = gr.Button("Run every available engine", variant="primary")
                c_table = gr.Dataframe(label="Predictions", interactive=False)

            with gr.Tab("Batch test"):
                gr.Markdown("Upload labelled clips to get accuracy, macro-F1 and a confusion matrix. "
                            "Labels come from RAVDESS / CREMA-D filenames, a `_ang` / `_sd` / `_lgh` / "
                            "`_surp` ending, or a fixed label you choose.")
                with gr.Row():
                    b_files = gr.File(file_count="multiple", file_types=["audio"], label="Clips")
                    with gr.Column():
                        b_engine = gr.Radio(engines, value=default_engine, label="Engine")
                        b_label = gr.Dropdown(["from filename"] + EMOTIONS, value="from filename",
                                              label="True label")
                        b_run = gr.Button("Run batch", variant="primary")
                b_stats = gr.HTML()
                with gr.Row():
                    b_cm = gr.Plot(label="Confusion matrix")
                    b_table = gr.Dataframe(label="Per-clip results", interactive=False)
                b_csv = gr.File(label="Download results (CSV)")

            with gr.Tab("History"):
                h_table = gr.Dataframe(label="This session", interactive=False)

        outputs = [hero, bars, dims, signal, history, h_table]
        run.click(analyze, [audio, engine, history], outputs)
        audio.stop_recording(analyze, [audio, engine, history], outputs)
        audio.upload(analyze, [audio, engine, history], outputs)
        c_run.click(compare, c_audio, c_table)
        b_run.click(batch_test, [b_files, b_engine, b_label], [b_stats, b_cm, b_table, b_csv])

    return demo


if __name__ == "__main__":
    theme = gr.themes.Soft(primary_hue="indigo")
    if "css" in inspect.signature(gr.Blocks.__init__).parameters:
        demo = build_ui(css=CSS, theme=theme)
        demo.launch(server_name="0.0.0.0", server_port=7860)
    else:
        demo = build_ui()
        demo.launch(server_name="0.0.0.0", server_port=7860, css=CSS, theme=theme)
