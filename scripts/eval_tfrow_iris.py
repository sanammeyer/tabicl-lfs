"""Evaluate TF-row embedding separability on the Iris dataset.

This script:
- Loads Iris, fits the TabICLClassifier preprocessing + loads pretrained model
- Extracts pre-ICL TF-row embeddings (anchors=train, queries=test)
- Computes high-dimensional separability metrics (no dimensionality reduction):
  * Silhouette score (cosine and euclidean)
  * Calinski–Harabasz index, Davies–Bouldin index
  * 1-NN / 5-NN accuracy (cosine on L2-normalized embeddings)
  * Logistic regression probe accuracy (linear separability)
  * Nearest-centroid accuracy (cosine)
- Saves 2D PCA and t-SNE plots colored by class to `runs/`
- Saves embeddings and metadata to `runs/iris_tfrow_embeddings.npz`

Run from repo root (after installing editable):
    pip install -e .
    python scripts/eval_tfrow_iris.py --n_estimators 4 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tabicl import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.pdlc.head import l2_normalize


def project_2d(X: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    method = method.lower()
    if X.shape[0] <= 2:
        return np.pad(X.astype(np.float32), ((0, 0), (0, max(0, 2 - X.shape[1]))))[:, :2]
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, X.shape[0] // 10))
        return TSNE(
            n_components=2, random_state=seed, init="pca", perplexity=perplexity
        ).fit_transform(X)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")


def nearest_centroid_cosine(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """Cosine nearest-centroid classifier on L2-normalized embeddings.

    Returns predicted labels for X_test.
    """
    # Expect normalized inputs. If not, normalize here defensively.
    X_train_n = l2_normalize(X_train)
    X_test_n = l2_normalize(X_test)
    classes = np.unique(y_train)
    centroids = np.stack([X_train_n[y_train == c].mean(axis=0) for c in classes], axis=0)
    centroids = l2_normalize(centroids)
    # Cosine similarity = dot product on unit vectors
    sims = X_test_n @ centroids.T  # (n_test, n_classes)
    idx = np.argmax(sims, axis=1)
    return classes[idx]


def eval_separability(
    emb_tr: np.ndarray,
    y_tr: np.ndarray,
    emb_te: np.ndarray,
    y_te: np.ndarray,
) -> dict:
    """Compute a suite of high-dimensional separability metrics."""
    # Use L2-normalized embeddings for cosine-based metrics
    tr_n = l2_normalize(emb_tr)
    te_n = l2_normalize(emb_te)
    all_n = np.vstack([tr_n, te_n])
    y_all = np.concatenate([y_tr, y_te])

    out = {}

    # Unsupervised cluster indices
    try:
        out["silhouette_cosine"] = float(silhouette_score(all_n, y_all, metric="cosine"))
    except Exception as e:
        out["silhouette_cosine"] = f"error: {e}"
    try:
        out["silhouette_euclidean"] = float(silhouette_score(np.vstack([emb_tr, emb_te]), y_all, metric="euclidean"))
    except Exception as e:
        out["silhouette_euclidean"] = f"error: {e}"
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(np.vstack([emb_tr, emb_te]), y_all))
    except Exception as e:
        out["calinski_harabasz"] = f"error: {e}"
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(np.vstack([emb_tr, emb_te]), y_all))
    except Exception as e:
        out["davies_bouldin"] = f"error: {e}"

    # Supervised probes
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn1.fit(tr_n, y_tr)
    knn5.fit(tr_n, y_tr)
    out["knn1_acc"] = float(accuracy_score(y_te, knn1.predict(te_n)))
    out["knn5_acc"] = float(accuracy_score(y_te, knn5.predict(te_n)))

    # Logistic regression linear probe (on normalized embeddings)
    lg = LogisticRegression(max_iter=2000, multi_class="auto")
    lg.fit(tr_n, y_tr)
    out["logreg_acc"] = float(accuracy_score(y_te, lg.predict(te_n)))

    # Nearest-centroid (cosine)
    y_nc = nearest_centroid_cosine(emb_tr, y_tr, emb_te)
    out["nc_acc"] = float(accuracy_score(y_te, y_nc))

    # LDA classifier (optional sanity)
    try:
        lda = LinearDiscriminantAnalysis()
        lda.fit(emb_tr, y_tr)
        out["lda_acc"] = float(accuracy_score(y_te, lda.predict(emb_te)))
    except Exception as e:
        out["lda_acc"] = f"error: {e}"

    return out


def format_metrics(m: dict) -> str:
    keys = [
        "silhouette_cosine",
        "silhouette_euclidean",
        "calinski_harabasz",
        "davies_bouldin",
        "knn1_acc",
        "knn5_acc",
        "logreg_acc",
        "nc_acc",
        "lda_acc",
    ]
    lines = []
    for k in keys:
        v = m.get(k, None)
        if isinstance(v, float):
            if k.endswith("acc"):
                lines.append(f"{k:>20s}: {v:.3f}")
            else:
                lines.append(f"{k:>20s}: {v:.3f}")
        else:
            lines.append(f"{k:>20s}: {v}")
    return "\n".join(lines)


def save_plots(
    out_dir: Path,
    emb_tr: np.ndarray,
    y_tr: np.ndarray,
    emb_te: np.ndarray,
    y_te: np.ndarray,
    seed: int,
) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Fit projection on combined for consistency
    all_emb = np.vstack([emb_tr, emb_te])
    P_pca = project_2d(all_emb, method="pca", seed=seed)
    P_tsne = project_2d(all_emb, method="tsne", seed=seed)
    n_tr = emb_tr.shape[0]
    for name, P in [("pca", P_pca), ("tsne", P_tsne)]:
        P_tr, P_te = P[:n_tr], P[n_tr:]
        plt.figure(figsize=(6.8, 5.5))
        # Use consistent colormap
        classes = np.unique(np.concatenate([y_tr, y_te]))
        # anchors
        plt.scatter(P_tr[:, 0], P_tr[:, 1], c=y_tr, cmap="tab10", s=28, alpha=0.9, marker="o", label="train")
        # queries
        plt.scatter(P_te[:, 0], P_te[:, 1], c=y_te, cmap="tab10", s=22, alpha=0.8, marker="x", label="test")
        plt.title(f"Iris TF-row ({name.upper()})")
        plt.legend(loc="best")
        plt.tight_layout()
        out_path = out_dir / f"iris_tfrow_{name}.png"
        plt.savefig(out_path, dpi=160)
        plt.close()
    return out_dir / "iris_tfrow_pca.png", out_dir / "iris_tfrow_tsne.png"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_estimators", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y
    )

    clf = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path="~/.cache/huggingface/hub/models--jingang--TabICL-clf/snapshots/eaf789a9b25ee8486d6f48997ba076f850bbc30b/tabicl-classifier-v1.1-0506.ckpt",
    )
    clf.fit(X_train, y_train)

    res = extract_tf_row_embeddings(clf, X_test)

    emb_tr = res["embeddings_train"]
    emb_te = res["embeddings_test"]
    y_tr = np.asarray(res["train_labels"])  # decoded labels (strings or ints)
    y_te = np.asarray(y_test)

    # Evaluate high-dimensional separability
    metrics = eval_separability(emb_tr, y_tr, emb_te, y_te)

    # Save embeddings and metadata
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "iris_tfrow_embeddings.npz",
        embeddings_train=emb_tr,
        embeddings_test=emb_te,
        train_labels=y_tr,
        test_labels=y_te,
        norm_method=res["norm_method"],
        feature_permutation=np.array(res["feature_permutation"], dtype=int),
        class_shift_offset=res["class_shift_offset"],
        embedding_dim=res["embedding_dim"],
        model_checkpoint=str(res["model_checkpoint"]),
        metrics=np.array(list(metrics.items()), dtype=object),
    )

    # Plots
    pca_path, tsne_path = save_plots(out_dir, l2_normalize(emb_tr), y_tr, l2_normalize(emb_te), y_te, args.seed)

    # Report
    print("Saved embeddings to:", (out_dir / "iris_tfrow_embeddings.npz").resolve())
    print("Saved plots:", pca_path.resolve(), tsne_path.resolve())
    print("\nHigh-dimensional separability (Iris, TF-row, pre-ICL):\n" + format_metrics(metrics))


if __name__ == "__main__":  # pragma: no cover
    main()

