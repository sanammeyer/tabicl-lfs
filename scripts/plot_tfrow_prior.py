"""Plot TF-row embeddings on a random prior episode to sanity-check separability.

Saves a 2D projection (t-SNE or PCA) of anchor and query embeddings, colored by class.

Example:
    python scripts/plot_tfrow_prior.py --prior mix_scm --n_estimators 4 --proj tsne \
        --max_points 2000 --out runs/tfrow_sanity.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from tabicl import TabICLClassifier, InferenceConfig
from tabicl.prior.dataset import PriorDataset
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.pdlc.head import l2_normalize


def project_2d(X: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    if X.shape[0] <= 2:
        return np.pad(X.astype(np.float32), ((0, 0), (0, max(0, 2 - X.shape[1]))))[:, :2]
    method = method.lower()
    if method == "pca":
        try:
            from sklearn.decomposition import PCA
            return PCA(n_components=2, random_state=seed).fit_transform(X)
        except Exception:
            pass
    # default to t-SNE
    try:
        from sklearn.manifold import TSNE
        # for stability, use perplexity suited to sample size
        perplexity = max(5, min(30, X.shape[0] // 10))
        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(X)
    except Exception:
        # final fallback: random projection
        rng = np.random.default_rng(seed)
        W = rng.normal(size=(X.shape[1], 2)).astype(np.float32)
        return X @ W


def main():
    p = argparse.ArgumentParser()
    # Prior sampling params (mirror pdlc_prior_train minimal set)
    p.add_argument("--prior", type=str, default="mix_scm", choices=["mlp_scm", "tree_scm", "mix_scm", "dummy"])
    p.add_argument("--min_features", type=int, default=4)
    p.add_argument("--max_features", type=int, default=32)
    p.add_argument("--max_classes", type=int, default=6)
    p.add_argument("--min_seq_len", type=int, default=120)
    p.add_argument("--max_seq_len", type=int, default=400)
    p.add_argument("--log_seq_len", action="store_true")

    # Model / extraction
    p.add_argument("--n_estimators", type=int, default=4)
    p.add_argument("--model_path", type=str, default=None)
    p.add_argument("--device", type=str, default=None, help="override device (cuda/cpu/mps)")
    p.add_argument("--randomize_variant", action="store_true")

    # Projection / plotting
    p.add_argument("--proj", type=str, default="tsne", choices=["tsne", "pca"]) \
        
    p.add_argument("--max_points", type=int, default=2000, help="max points to plot for readability")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/tfrow_sanity.png")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    else:
        device = torch.device(args.device)

    # Prior generator
    prior_gen = PriorDataset(
        batch_size=1,
        prior_type=args.prior,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=args.log_seq_len,
        device="cpu",
    )

    # Sample one dataset
    X, y, d, seq_lens, train_sizes = prior_gen.get_batch(batch_size=1)
    T = int(seq_lens[0].item())
    H = int(d[0].item())
    split = int(train_sizes[0].item())
    X = X[0, :T, :H].cpu().numpy()
    y = y[0, :T].cpu().numpy()
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    # Prepare classifier wrapper and inference config
    clf = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path=args.model_path,
        allow_auto_download=True if args.model_path is None else False,
    )
    # Device
    if clf.device is None:
        clf.device_ = device
    elif isinstance(clf.device, str):
        clf.device_ = torch.device(clf.device)
    else:
        clf.device_ = clf.device
    clf._load_model()
    clf.model_.to(clf.device_)
    init_cfg = {
        "COL_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
        "ROW_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
        "ICL_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
    }
    if clf.inference_config is None:
        clf.inference_config_ = InferenceConfig()
        clf.inference_config_.update_from_dict(init_cfg)
    elif isinstance(clf.inference_config, dict):
        cfg = InferenceConfig()
        for k, v in clf.inference_config.items():
            if k in init_cfg:
                init_cfg[k].update(v)
        cfg.update_from_dict(init_cfg)
        clf.inference_config_ = cfg
    else:
        clf.inference_config_ = clf.inference_config

    # Preprocess using anchors only (same as prior_train)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_tr_enc = le.fit_transform(y_tr)
    clf.y_encoder_ = le
    clf.classes_ = le.classes_
    clf.n_classes_ = len(le.classes_)

    from tabicl.sklearn.preprocessing import TransformToNumerical, EnsembleGenerator
    X_encoder = TransformToNumerical(verbose=clf.verbose)
    X_tr_num = X_encoder.fit_transform(X_tr)
    clf.X_encoder_ = X_encoder
    eg = EnsembleGenerator(
        n_estimators=clf.n_estimators,
        norm_methods=clf.norm_methods or ["none", "power"],
        feat_shuffle_method=clf.feat_shuffle_method,
        class_shift=clf.class_shift,
        outlier_threshold=clf.outlier_threshold,
        random_state=clf.random_state,
    )
    eg.fit(X_tr_num, y_tr_enc)
    clf.ensemble_generator_ = eg
    clf.n_features_in_ = X_tr_num.shape[1]

    res = extract_tf_row_embeddings(
        clf,
        X_te,
        choose_random_variant=args.randomize_variant,
        rng=rng,
    )
    assert int(res["train_size"]) == len(y_tr)
    assert np.array_equal(np.asarray(res["train_labels"]), np.asarray(y_tr))
    emb_tr = l2_normalize(res["embeddings_train"])  # anchors
    emb_te = l2_normalize(res["embeddings_test"])   # queries
    y_anchor = np.asarray(res["train_labels"])      # decoded labels
    y_query = np.asarray(y_te)

    # Optionally subsample for plotting
    def subsample(X, y, max_n):
        if X.shape[0] <= max_n:
            return X, y, np.arange(X.shape[0])
        idx = np.random.default_rng(args.seed).choice(X.shape[0], size=max_n, replace=False)
        return X[idx], y[idx], idx

    emb_tr_plot, y_tr_plot, _ = subsample(emb_tr, y_anchor, args.max_points // 2)
    emb_te_plot, y_te_plot, _ = subsample(emb_te, y_query, args.max_points // 2)

    # Compute 2D projection on combined space for consistency
    all_emb = np.vstack([emb_tr_plot, emb_te_plot])
    proj = project_2d(all_emb, method=args.proj, seed=args.seed)
    P_tr = proj[: emb_tr_plot.shape[0]]
    P_te = proj[emb_tr_plot.shape[0] :]

    # Plot (use consistent color scaling across anchors/queries)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7.5, 6))
    all_labs = np.concatenate([y_tr_plot, y_te_plot])
    vmin, vmax = int(all_labs.min()), int(all_labs.max())
    # anchors
    plt.scatter(
        P_tr[:, 0], P_tr[:, 1], c=y_tr_plot, cmap="tab10", vmin=vmin, vmax=vmax,
        s=18, alpha=0.9, marker="o", label="anchors",
    )
    # queries
    plt.scatter(
        P_te[:, 0], P_te[:, 1], c=y_te_plot, cmap="tab10", vmin=vmin, vmax=vmax,
        s=12, alpha=0.7, marker="x", label="queries",
    )
    plt.title(f"TF-row embeddings ({args.proj.upper()})  C={len(np.unique(y))}, H={H}, T={T}, split={split}")
    plt.legend(loc="best")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved TF-row embedding plot to {out_path.resolve()}")


if __name__ == "__main__":
    main()
