"""OPTIONAL (step 6): cross-validated classifier on frozen SSL embeddings.

Uses the same actor-wise folds as cross_validate.py, so the numbers are
directly comparable. Sweeps which transformer layer(s) to use.

    python -m src.train_ssl --model microsoft/wavlm-base-plus
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

from src.dataset import EMOTION_TO_INDEX, INDEX_TO_EMOTION, PROJECT_ROOT

SSL_DIR = PROJECT_ROOT / "data" / "processed" / "ssl"


def load_embeddings(model_name):
    index = pd.read_csv(SSL_DIR / model_name.replace("/", "_") / "index.csv")
    embeddings = np.stack([np.load(PROJECT_ROOT / p) for p in index["ssl_path"]])
    labels = index["emotion"].map(EMOTION_TO_INDEX).to_numpy()
    return index, embeddings, labels


def select(embeddings, layers):
    """(N, layers, hidden) -> (N, features) averaging the chosen layers."""
    return embeddings[:, layers, :].mean(axis=1)


def cross_validate_layers(index, embeddings, labels, layers, c_value,
                          speaker_norm):
    folds = index["fold"].to_numpy()
    actors = index["actor_id"].to_numpy()
    features = select(embeddings, layers)

    if speaker_norm:
        # Subtract each actor's mean embedding (uses only that actor's own
        # recordings, no labels), which removes much of the voice identity.
        features = features.copy()
        for actor in np.unique(actors):
            mask = actors == actor
            features[mask] -= features[mask].mean(axis=0, keepdims=True)

    all_true, all_pred, fold_f1 = [], [], []
    for fold in range(5):
        train = folds != fold
        train &= folds >= 0
        val = folds == fold

        scaler = StandardScaler().fit(features[train])
        classifier = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced"
        ).fit(scaler.transform(features[train]), labels[train])

        predictions = classifier.predict(scaler.transform(features[val]))
        fold_f1.append(f1_score(labels[val], predictions, average="macro"))
        all_true.extend(labels[val].tolist())
        all_pred.extend(predictions.tolist())

    accuracy = float(np.mean(np.array(all_true) == np.array(all_pred)))
    return accuracy, float(np.mean(fold_f1)), float(np.std(fold_f1)), \
        all_true, all_pred


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/wavlm-base-plus")
    args = parser.parse_args()

    index, embeddings, labels = load_embeddings(args.model)
    n_layers = embeddings.shape[1]
    print(f"{len(labels)} recordings, {n_layers} layers, "
          f"{embeddings.shape[2]} dims\n")

    candidates = [[i] for i in range(0, n_layers, 2)] + [
        list(range(n_layers // 4, 3 * n_layers // 4 + 1)),
        list(range(n_layers)),
    ]

    best = None
    print(f"{'layers':<22}{'speaker-norm':<14}{'C':<8}{'acc':<8}{'macroF1':<10}")
    for layers in candidates:
        for speaker_norm in (False, True):
            for c_value in (0.01, 0.1, 1.0):
                acc, f1_mean, f1_std, y_true, y_pred = cross_validate_layers(
                    index, embeddings, labels, layers, c_value, speaker_norm
                )
                label = (str(layers) if len(layers) < 4
                         else f"{layers[0]}..{layers[-1]} avg")
                print(f"{label:<22}{str(speaker_norm):<14}{c_value:<8}"
                      f"{acc:<8.3f}{f1_mean:<6.3f}+/-{f1_std:.3f}")
                if best is None or f1_mean > best[0]:
                    best = (f1_mean, label, speaker_norm, c_value, y_true,
                            y_pred)

    f1_mean, label, speaker_norm, c_value, y_true, y_pred = best
    names = [INDEX_TO_EMOTION[i] for i in range(len(EMOTION_TO_INDEX))]
    print(f"\nBest: layers {label}, speaker-norm {speaker_norm}, C {c_value}, "
          f"macro-F1 {f1_mean:.3f}")
    print("(Picking the best of many settings on the same folds is slightly "
          "optimistic; confirming on the test actors below.)")
    print(names)
    print(confusion_matrix(y_true, y_pred, labels=list(range(len(names)))))

    # ------------------------------------------------------------------
    # Final Evaluation on Held-Out Test Actors
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("EVALUATING ON HELD-OUT TEST ACTORS (Unseen during tuning)")
    print("=" * 60)

    best_layers = list(range(n_layers)) if "0.." in label else (
        list(range(n_layers // 4, 3 * n_layers // 4 + 1)) if ".." in label else eval(label)
    )

    dev_mask = (index["fold"] >= 0).to_numpy()
    test_mask = (index["fold"] < 0).to_numpy()

    test_actors = sorted(index.loc[test_mask, "actor_id"].unique().tolist())
    print(f"Test actors: {test_actors} ({test_mask.sum()} clips)")

    features = select(embeddings, best_layers)
    actors = index["actor_id"].to_numpy()

    for spk_norm in (False, True):
        norm_label = "WITH speaker-norm" if spk_norm else "WITHOUT speaker-norm (pure zero-shot)"
        feat_proc = features.copy()
        if spk_norm:
            for actor in np.unique(actors):
                mask = actors == actor
                feat_proc[mask] -= feat_proc[mask].mean(axis=0, keepdims=True)

        scaler = StandardScaler().fit(feat_proc[dev_mask])
        clf = LogisticRegression(
            C=c_value, max_iter=2000, class_weight="balanced"
        ).fit(scaler.transform(feat_proc[dev_mask]), labels[dev_mask])

        y_test = labels[test_mask]
        y_pred_test = clf.predict(scaler.transform(feat_proc[test_mask]))
        acc = float(np.mean(y_test == y_pred_test))
        f1 = float(f1_score(y_test, y_pred_test, average="macro"))

        print(f"\nTest Results {norm_label}:")
        print(f"  Accuracy: {acc:.4f} ({acc*100:.1f}%)")
        print(f"  Macro-F1: {f1:.4f}")
        print(confusion_matrix(y_test, y_pred_test, labels=list(range(len(names)))))

    # Save final model for predict.py
    import joblib
    checkpoints_dir = PROJECT_ROOT / "data" / "processed" / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # We train the deployable classifier on dev set without speaker norm so single-file predict works out-of-the-box
    scaler_final = StandardScaler().fit(features[dev_mask])
    clf_final = LogisticRegression(
        C=c_value, max_iter=2000, class_weight="balanced"
    ).fit(scaler_final.transform(features[dev_mask]), labels[dev_mask])

    save_payload = {
        "model_name": args.model,
        "layers": best_layers,
        "scaler": scaler_final,
        "classifier": clf_final,
        "c_value": c_value,
        "emotion_to_index": EMOTION_TO_INDEX,
        "index_to_emotion": INDEX_TO_EMOTION,
    }
    model_save_path = checkpoints_dir / "ssl_classifier.joblib"
    joblib.dump(save_payload, model_save_path)
    print(f"\nSaved deployable model for `predict.py` to: {model_save_path}")


if __name__ == "__main__":
    main()
