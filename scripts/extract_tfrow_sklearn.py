"""Extract TF-row embeddings from TabICL on common sklearn datasets.

Usage examples:
  - Wine:          python scripts/extract_tfrow_sklearn.py --dataset wine
  - Breast cancer: python scripts/extract_tfrow_sklearn.py --dataset breast_cancer
  - Digits(10c):   python scripts/extract_tfrow_sklearn.py --dataset digits

Outputs:
  - <name>_tfrow_embeddings.npz (embeddings + labels + metadata)
  - figures/<name>_tfrow_pca.png (2D PCA of train/test embeddings)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine, load_breast_cancer, load_digits
import pandas as pd

from tabicl import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings


def load_dataset(name: str):
    name = name.lower()
    if name == "wine":
        b = load_wine(as_frame=True)
        X, y = b.data, b.target
    elif name in ("breast_cancer", "cancer"):
        b = load_breast_cancer(as_frame=True)
        X, y = b.data, b.target
    elif name == "digits":
        b = load_digits()
        X = pd.DataFrame(b.data, columns=[f"px{i}" for i in range(b.data.shape[1])])
        y = pd.Series(b.target)
    else:
        raise ValueError(f"Unsupported dataset '{name}'. Use: wine, breast_cancer, digits")
    return X, y


def pca_plot(emb_train: np.ndarray, emb_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray, out_path: Path, title: str):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(np.vstack([emb_train, emb_test]))
    Z_tr = Z[: emb_train.shape[0]]
    Z_te = Z[emb_train.shape[0] :]

    labs = np.unique(np.concatenate([y_train, y_test]))
    cmap = plt.get_cmap("tab10")
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(labs)}

    plt.figure(figsize=(6, 5))
    for lab in labs:
        mtr = y_train == lab
        mte = y_test == lab
        if mtr.any():
            plt.scatter(Z_tr[mtr, 0], Z_tr[mtr, 1], c=[color_map[lab]], s=20, alpha=0.8, marker="o", label=f"train:{lab}")
        if mte.any():
            plt.scatter(Z_te[mte, 0], Z_te[mte, 1], c=[color_map[lab]], s=35, alpha=0.9, marker="x", label=f"test:{lab}")

    ev = pca.explained_variance_ratio_
    plt.title(f"{title}  PCA (EVR: {ev[0]:.2f}, {ev[1]:.2f})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default="wine", help="wine | breast_cancer | digits")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_path", type=str, default=None, help="optional local checkpoint path to avoid download")
    ap.add_argument("--no_plot", action="store_true")
    ap.add_argument("--randomize_variant", action="store_true")
    args = ap.parse_args()

    X, y = load_dataset(args.dataset)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=args.test_size, random_state=args.seed, stratify=y)

    clf = TabICLClassifier(
        n_estimators=4,
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path=args.model_path,
        allow_auto_download=True if args.model_path is None else False,
    )
    clf.fit(X_tr, y_tr)

    # Extract TF-row embeddings (pre-ICL) for train+test episode
    res = extract_tf_row_embeddings(
        clf,
        X_te,
        choose_random_variant=args.randomize_variant,
    )

    out_prefix = f"{args.dataset}_tfrow"
    npz_path = Path(f"{out_prefix}_embeddings.npz")
    np.savez(
        npz_path,
        embeddings_all=res["embeddings_all"],
        embeddings_train=res["embeddings_train"],
        embeddings_test=res["embeddings_test"],
        train_labels=res["train_labels"],
        test_labels=np.asarray(y_te),
        norm_method=res["norm_method"],
        feature_permutation=np.array(res["feature_permutation"], dtype=int),
        class_shift_offset=res["class_shift_offset"],
        model_checkpoint=str(res["model_checkpoint"]),
    )

    print(f"Saved embeddings to {npz_path.resolve()}")
    print(f"Variant idx={res['variant_index']}  norm={res['norm_method']}  emb_dim={res['embedding_dim']}")
    print(f"Train size={res['train_size']}  Test size={res['test_size']}")

    if not args.no_plot:
        figs_dir = Path("figures")
        fig_path = figs_dir / f"{args.dataset}_tfrow_pca.png"
        pca_plot(res["embeddings_train"], res["embeddings_test"], np.asarray(res["train_labels"]), np.asarray(y_te), fig_path, f"{args.dataset}")
        print(f"Saved PCA plot to {fig_path.resolve()}")


if __name__ == "__main__":
    main()

