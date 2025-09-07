"""Extract row-level (tf_row) embeddings from the TabICL model on the Iris dataset.

This script:
1. Loads the Iris dataset and splits into train/test
2. Fits the scikit-learn `TabICLClassifier` (only prepares transforms & loads checkpoint)
3. Reconstructs one ensemble variant (optionally the one with identity feature order)
4. Runs the column embedder + row interaction to obtain row embeddings BEFORE in-context learning
5. Saves embeddings to `iris_tfrow_embeddings.npz` and prints a short summary

Notes:
- Row embeddings depend on (a) normalization method, (b) feature shuffle pattern, (c) class shift.
- We pick a single variant. If an identity feature permutation exists, we choose that; otherwise we take the first.
- Training labels inside a variant may be cyclically shifted. We reverse the shift to recover original labels.

Run from repo root (after installing in editable mode):
    python scripts/extract_tfrow_iris.py
"""

from __future__ import annotations

import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from tabicl import TabICLClassifier


def extract_tf_row_embeddings(clf: TabICLClassifier, X_test_df):
    """Return row interaction (tf_row) embeddings for a single ensemble variant.

    Parameters
    ----------
    clf : Fitted TabICLClassifier
    X_test_df : pandas.DataFrame
        Test portion (NOT transformed). Training portion is inside the classifier state.

    Returns
    -------
    dict with keys:
        norm_method : str used normalization method
        variant_index : int index of chosen ensemble variant
        feature_permutation : list[int] feature index order applied
        class_shift_offset : int applied class shift (already reversed in returned train labels)
        train_labels : ndarray original (unshifted) train labels (string/int)
        test_size : int number of test rows
        train_size : int number of train rows
        embeddings_all : ndarray shape (T, D) row embeddings (train first, then test)
        embeddings_train : ndarray shape (train_size, D)
        embeddings_test : ndarray shape (test_size, D)
        embedding_dim : int D
        model_checkpoint : Path to checkpoint used
    """

    assert hasattr(clf, "ensemble_generator_"), "Classifier must be fitted first."

    # 1. Transform test set via the fitted feature encoder (categorical -> numeric)
    X_test_num = clf.X_encoder_.transform(X_test_df)

    # 2. Build ensemble data variants (already applies unique feature filtering + preprocessing)
    data = clf.ensemble_generator_.transform(X_test_num)

    # Choose a normalization method (first one present)
    norm_method = next(iter(data.keys()))
    Xs, ys_shifted = data[norm_method]  # Xs: (n_variants, T, H_filtered)  ys_shifted: (n_variants, train_size)

    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    shift_offsets = clf.ensemble_generator_.class_shift_offsets_[norm_method]

    # Try to find identity permutation to get canonical feature order
    variant_index = None
    for i, pattern in enumerate(shuffle_patterns):
        if list(pattern) == sorted(pattern):  # identity (after unique feature filtering)
            variant_index = i
            break
    if variant_index is None:
        variant_index = 0

    X_variant = Xs[variant_index]  # (T, H_filtered)
    y_variant_shifted = ys_shifted[variant_index]  # (train_size,)
    shift_offset = shift_offsets[variant_index]

    # Reverse class shift
    y_variant = (y_variant_shifted - shift_offset) % clf.n_classes_
    y_variant = clf.y_encoder_.inverse_transform(y_variant.astype(int))

    train_size = y_variant_shifted.shape[0]
    T = X_variant.shape[0]
    test_size = T - train_size

    # 3. Convert to tensors and pass through col_embedder + row_interactor (inference mode)
    model = clf.model_
    model.eval()

    device = clf.device_
    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(device)  # (1, T, H)

    # Use inference configuration from classifier
    inference_config = clf.inference_config_

    with torch.no_grad():
        col_out = model.col_embedder(
            X_tensor,
            train_size=train_size,
            feature_shuffles=None,  # already applied
            mgr_config=inference_config.COL_CONFIG,
        )  # (1, T, H+C, E)
        row_reps = model.row_interactor(col_out, mgr_config=inference_config.ROW_CONFIG)  # (1, T, C*E)

    embeddings = row_reps.squeeze(0).cpu().numpy()  # (T, D)

    return {
        "norm_method": norm_method,
        "variant_index": variant_index,
        "feature_permutation": list(shuffle_patterns[variant_index]),
        "class_shift_offset": int(shift_offset),
        "train_labels": y_variant,
        "train_size": train_size,
        "test_size": test_size,
        "embeddings_all": embeddings,
        "embeddings_train": embeddings[:train_size],
        "embeddings_test": embeddings[train_size:],
        "embedding_dim": embeddings.shape[1],
        "model_checkpoint": getattr(clf, "model_path_", None),
    }


def main():
    iris = load_iris(as_frame=True)
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y
    )

    clf = TabICLClassifier(n_estimators=4, batch_size=4, use_amp=True, verbose=False)
    clf.fit(X_train, y_train)

    results = extract_tf_row_embeddings(clf, X_test)

    out_path = Path("iris_tfrow_embeddings.npz")
    np.savez(
        out_path,
        embeddings_all=results["embeddings_all"],
        embeddings_train=results["embeddings_train"],
        embeddings_test=results["embeddings_test"],
        train_labels=results["train_labels"],
        test_labels=np.asarray(y_test),
        norm_method=results["norm_method"],
        feature_permutation=np.array(results["feature_permutation"], dtype=int),
        class_shift_offset=results["class_shift_offset"],
        model_checkpoint=str(results["model_checkpoint"]),
    )

    print(f"Saved embeddings to {out_path.resolve()}")
    print(f"Variant index: {results['variant_index']}  norm_method: {results['norm_method']}")
    print(f"Feature permutation (first 10): {results['feature_permutation'][:10]}")
    print(f"Embedding dim: {results['embedding_dim']}")
    print(f"Train size: {results['train_size']}  Test size: {results['test_size']}")
    print("First 3 train embeddings (L2 norm):", np.linalg.norm(results["embeddings_train"][:3], axis=1))
    print("First 3 test embeddings (L2 norm):", np.linalg.norm(results["embeddings_test"][:3], axis=1))

    # Quick 2D visualization via PCA (train vs test, colored by label)
    try:
        figures_dir = Path("figures")
        figures_dir.mkdir(parents=True, exist_ok=True)

        emb_train = results["embeddings_train"]
        emb_test = results["embeddings_test"]
        y_train = np.asarray(results["train_labels"])  # already inverse-transformed
        y_test_np = np.asarray(y_test)

        # Fit PCA on all rows to get a shared projection
        pca = PCA(n_components=2, random_state=0)
        Z_all = pca.fit_transform(np.vstack([emb_train, emb_test]))
        Z_train = Z_all[: emb_train.shape[0]]
        Z_test = Z_all[emb_train.shape[0] :]

        # Colors by class label
        unique_labels = np.unique(np.concatenate([y_train, y_test_np]))
        cmap = plt.get_cmap("tab10")
        color_map = {lab: cmap(i % 10) for i, lab in enumerate(unique_labels)}

        plt.figure(figsize=(6, 5))
        for lab in unique_labels:
            mask_tr = y_train == lab
            mask_te = y_test_np == lab
            if mask_tr.any():
                plt.scatter(
                    Z_train[mask_tr, 0],
                    Z_train[mask_tr, 1],
                    c=[color_map[lab]],
                    label=f"train:{lab}",
                    s=25,
                    alpha=0.8,
                    marker="o",
                    edgecolors="none",
                )
            if mask_te.any():
                plt.scatter(
                    Z_test[mask_te, 0],
                    Z_test[mask_te, 1],
                    c=[color_map[lab]],
                    label=f"test:{lab}",
                    s=40,
                    alpha=0.9,
                    marker="x",
                )

        ev = pca.explained_variance_ratio_
        plt.title(f"TF-row PCA (EVR: {ev[0]:.2f}, {ev[1]:.2f})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(loc="best", fontsize=8, ncol=2)
        plt.tight_layout()
        out_fig = figures_dir / "iris_tfrow_pca.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"Saved PCA plot to {out_fig.resolve()}")
    except Exception as e:
        print("Plotting failed:", repr(e))


if __name__ == "__main__":  # pragma: no cover
    main()
