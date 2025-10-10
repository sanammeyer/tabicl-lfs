"""Evaluate TF-row embedding separability on a synthetic prior episode.

Mirrors scripts/plot_tfrow_prior.py for data extraction, but adds high-D metrics:
- Silhouette (cosine/euclidean), Calinski–Harabasz, Davies–Bouldin
- 1-NN / 5-NN accuracy (cosine, on L2-normalized embeddings)
- Logistic regression probe, nearest-centroid (cosine), LDA

Run:
  python scripts/eval_tfrow_prior.py --prior mix_scm --n_estimators 4 --proj tsne \
      --seed 42 --out runs/tfrow_prior_eval
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    accuracy_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tabicl import TabICLClassifier, InferenceConfig
from tabicl.prior.dataset import PriorDataset
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.pdlc.head import l2_normalize


def project_2d(X: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    if X.shape[0] <= 2:
        return np.pad(X.astype(np.float32), ((0, 0), (0, max(0, 2 - X.shape[1]))))[:, :2]
    method = method.lower()
    if method == "pca":
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=seed).fit_transform(X)
    else:
        from sklearn.manifold import TSNE

        perplexity = max(5, min(30, X.shape[0] // 10))
        return TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity).fit_transform(X)


def nearest_centroid_cosine(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    X_train_n = l2_normalize(X_train)
    X_test_n = l2_normalize(X_test)
    classes = np.unique(y_train)
    centroids = np.stack([X_train_n[y_train == c].mean(axis=0) for c in classes], axis=0)
    centroids = l2_normalize(centroids)
    sims = X_test_n @ centroids.T
    return classes[np.argmax(sims, axis=1)]


def _class_stats(y_tr: np.ndarray, y_te: np.ndarray) -> dict:
    classes_tr, counts_tr = np.unique(y_tr, return_counts=True)
    classes_te, counts_te = np.unique(y_te, return_counts=True)
    cls_all = sorted(set(classes_tr.tolist()) | set(classes_te.tolist()))
    stats = {
        "classes": cls_all,
        "n_classes_union": len(cls_all),
        "train_counts": {int(c): int(cnt) for c, cnt in zip(classes_tr, counts_tr)},
        "test_counts": {int(c): int(cnt) for c, cnt in zip(classes_te, counts_te)},
    }
    return stats


def _drop_missing_classes(X_tr, y_tr, X_te, y_te):
    """Keep only classes that appear in both splits.

    Returns filtered (X_tr, y_tr, X_te, y_te, kept_classes)
    """
    keep = sorted(set(np.unique(y_tr)).intersection(set(np.unique(y_te))))
    mask_tr = np.isin(y_tr, keep)
    mask_te = np.isin(y_te, keep)
    return X_tr[mask_tr], y_tr[mask_tr], X_te[mask_te], y_te[mask_te], keep


def eval_metrics(emb_tr, y_tr, emb_te, y_te, *, pca_k: int = 32):
    tr_n = l2_normalize(emb_tr)
    te_n = l2_normalize(emb_te)
    all_euc = np.vstack([emb_tr, emb_te])
    all_cos = np.vstack([tr_n, te_n])
    y_all = np.concatenate([y_tr, y_te])

    out = {}
    try:
        out["silhouette_cosine"] = float(silhouette_score(all_cos, y_all, metric="cosine"))
    except Exception as e:
        out["silhouette_cosine"] = f"error: {e}"
    try:
        out["silhouette_euclidean"] = float(silhouette_score(all_euc, y_all, metric="euclidean"))
    except Exception as e:
        out["silhouette_euclidean"] = f"error: {e}"
    try:
        out["calinski_harabasz"] = float(calinski_harabasz_score(all_euc, y_all))
    except Exception as e:
        out["calinski_harabasz"] = f"error: {e}"
    try:
        out["davies_bouldin"] = float(davies_bouldin_score(all_euc, y_all))
    except Exception as e:
        out["davies_bouldin"] = f"error: {e}"
    # PCA-whitened stability slice
    try:
        from sklearn.decomposition import PCA
        k = max(2, min(pca_k, all_euc.shape[1], all_euc.shape[0]-1))
        Z = PCA(n_components=k, whiten=True, random_state=0).fit_transform(all_euc)
        out["calinski_harabasz_pcaK"] = float(calinski_harabasz_score(Z, y_all))
        out["davies_bouldin_pcaK"] = float(davies_bouldin_score(Z, y_all))
    except Exception as e:
        out["calinski_harabasz_pcaK"] = f"error: {e}"
        out["davies_bouldin_pcaK"] = f"error: {e}"

    # Use Euclidean on L2-normalized embeddings (equivalent to cosine)
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(tr_n, y_tr)
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(tr_n, y_tr)
    out["knn1_acc"] = float(accuracy_score(y_te, knn1.predict(te_n)))
    out["knn5_acc"] = float(accuracy_score(y_te, knn5.predict(te_n)))

    lg = LogisticRegression(max_iter=2000, multi_class="auto", solver="lbfgs", C=1.0).fit(tr_n, y_tr)
    out["logreg_acc"] = float(accuracy_score(y_te, lg.predict(te_n)))
    # Optional AUC (OvR macro) when >2 classes
    try:
        if len(np.unique(y_tr)) > 2:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(np.concatenate([y_tr, y_te]))
            proba = lg.predict_proba(te_n)
            y_true_bin = label_binarize(y_te, classes=classes)
            # Align proba columns to classes order
            # LogisticRegression uses classes_ attribute
            cols = [np.where(lg.classes_ == c)[0][0] for c in classes]
            proba_aligned = proba[:, cols]
            out["logreg_auc_macro"] = float(roc_auc_score(y_true_bin, proba_aligned, multi_class="ovr", average="macro"))
    except Exception as e:
        out["logreg_auc_macro"] = f"error: {e}"

    out["nc_acc"] = float(accuracy_score(y_te, nearest_centroid_cosine(emb_tr, y_tr, emb_te)))

    try:
        out["lda_acc"] = float(accuracy_score(y_te, LinearDiscriminantAnalysis().fit(emb_tr, y_tr).predict(emb_te)))
    except Exception as e:
        out["lda_acc"] = f"error: {e}"

    return out


def eval_raw_feature_metrics(X_tr, y_tr, X_te, y_te, *, pca_k: int = 32):
    """Run the same suite on preprocessed raw input features.

    Inputs are assumed already standardized/outlier-handled consistently between splits.
    """
    from tabicl.pdlc.head import l2_normalize as l2n

    tr = X_tr.astype(np.float32)
    te = X_te.astype(np.float32)
    all_euc = np.vstack([tr, te])
    all_cos = np.vstack([l2n(tr), l2n(te)])
    y_all = np.concatenate([y_tr, y_te])

    out = {}
    try:
        out["raw_silhouette_cosine"] = float(silhouette_score(all_cos, y_all, metric="cosine"))
    except Exception as e:
        out["raw_silhouette_cosine"] = f"error: {e}"
    try:
        out["raw_silhouette_euclidean"] = float(silhouette_score(all_euc, y_all, metric="euclidean"))
    except Exception as e:
        out["raw_silhouette_euclidean"] = f"error: {e}"
    try:
        out["raw_calinski_harabasz"] = float(calinski_harabasz_score(all_euc, y_all))
    except Exception as e:
        out["raw_calinski_harabasz"] = f"error: {e}"
    try:
        out["raw_davies_bouldin"] = float(davies_bouldin_score(all_euc, y_all))
    except Exception as e:
        out["raw_davies_bouldin"] = f"error: {e}"
    # PCA-whitened
    try:
        from sklearn.decomposition import PCA
        k = max(2, min(pca_k, all_euc.shape[1], all_euc.shape[0]-1))
        Z = PCA(n_components=k, whiten=True, random_state=0).fit_transform(all_euc)
        out["raw_calinski_harabasz_pcaK"] = float(calinski_harabasz_score(Z, y_all))
        out["raw_davies_bouldin_pcaK"] = float(davies_bouldin_score(Z, y_all))
    except Exception as e:
        out["raw_calinski_harabasz_pcaK"] = f"error: {e}"
        out["raw_davies_bouldin_pcaK"] = f"error: {e}"

    # Probes
    knn1 = KNeighborsClassifier(n_neighbors=1, metric="euclidean").fit(l2n(tr), y_tr)
    knn5 = KNeighborsClassifier(n_neighbors=5, metric="euclidean").fit(l2n(tr), y_tr)
    out["raw_knn1_acc"] = float(accuracy_score(y_te, knn1.predict(l2n(te))))
    out["raw_knn5_acc"] = float(accuracy_score(y_te, knn5.predict(l2n(te))))

    lg = LogisticRegression(max_iter=2000, multi_class="auto", solver="lbfgs", C=1.0).fit(tr, y_tr)
    out["raw_logreg_acc"] = float(accuracy_score(y_te, lg.predict(te)))
    try:
        if len(np.unique(y_tr)) > 2:
            from sklearn.preprocessing import label_binarize
            classes = np.unique(np.concatenate([y_tr, y_te]))
            proba = lg.predict_proba(te)
            y_true_bin = label_binarize(y_te, classes=classes)
            cols = [np.where(lg.classes_ == c)[0][0] for c in classes]
            proba_aligned = proba[:, cols]
            out["raw_logreg_auc_macro"] = float(roc_auc_score(y_true_bin, proba_aligned, multi_class="ovr", average="macro"))
    except Exception as e:
        out["raw_logreg_auc_macro"] = f"error: {e}"

    try:
        out["raw_lda_acc"] = float(accuracy_score(y_te, LinearDiscriminantAnalysis().fit(tr, y_tr).predict(te)))
    except Exception as e:
        out["raw_lda_acc"] = f"error: {e}"

    # Nearest-centroid (cosine)
    out["raw_nc_acc"] = float(accuracy_score(y_te, nearest_centroid_cosine(tr, y_tr, te)))

    return out


def fmt(out: dict) -> str:
    order = [
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
    return "\n".join(f"{k:>20s}: {out.get(k)}" for k in order)


def main():
    p = argparse.ArgumentParser()
    # Prior / data
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
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--randomize_variant", action="store_true")
    p.add_argument("--diagnostic_identity", action="store_true", help="Use identity ensemble (no aug) for diagnostics")
    p.add_argument("--try_perm_formats", action="store_true", help="Try both list and flat feature_shuffles (n_estimators=1)")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--resample_if_missing", action="store_true", help="Resample episode if class missing in either split")
    p.add_argument("--max_resample_tries", type=int, default=50)
    p.add_argument("--pca_k", type=int, default=32, help="k PCs for whitened CH/DB (stability)")
    p.add_argument("--min_per_class", type=int, default=0, help="Require at least this many anchors and queries per class (resample if needed)")
    # Prior tweaks
    p.add_argument("--balanced_binary", action="store_true", help="Force balanced classes for binary episodes")
    p.add_argument("--ordered_prob", type=float, default=None, help="Override multiclass_ordered_prob (e.g., 0.8)")
    p.add_argument("--no_permute_labels", action="store_true", help="Disable label permutation in Reg2Cls")

    # Output
    p.add_argument("--proj", type=str, default="tsne", choices=["tsne", "pca"]) \
        
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/tfrow_prior_eval")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)

    # Prior generator
    # Fixed-HP overrides for easier/separable episodes (optional)
    fixed_hp_overrides = {}
    if args.balanced_binary:
        fixed_hp_overrides["balanced"] = True
    if args.ordered_prob is not None:
        fixed_hp_overrides["multiclass_ordered_prob"] = float(args.ordered_prob)
    if args.no_permute_labels:
        fixed_hp_overrides["permute_labels"] = False

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
    # Apply overrides if any
    if fixed_hp_overrides:
        prior_gen.prior.fixed_hp.update(fixed_hp_overrides)
    def sample_episode():
        X, y, d, seq_lens, train_sizes = prior_gen.get_batch(batch_size=1)
        T = int(seq_lens[0].item())
        H = int(d[0].item())
        split = int(train_sizes[0].item())
        X = X[0, :T, :H].cpu().numpy()
        y = y[0, :T].cpu().numpy()
        return X[:split], y[:split], X[split:], y[split:], H, T, split

    # Optional diagnostic identity ensemble
    diag = args.diagnostic_identity
    # Classifier wrapper + inference config
    clf = TabICLClassifier(
        n_estimators=(1 if diag else args.n_estimators),
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path=args.model_path,
        allow_auto_download=True if args.model_path is None else False,
    )
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

    # Output dir
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Accumulators
    import csv
    csv_path = out_dir / "tfrow_prior_metrics.csv"
    wrote_header = False
    rows = []

    for ep in range(args.episodes):
        # sample episode with coverage handling
        tries = 0
        while True:
            X_tr, y_tr, X_te, y_te, H, T, split = sample_episode()
            set_tr, set_te = set(np.unique(y_tr)), set(np.unique(y_te))
            coverage_ok = set_tr == set_te and len(set_tr) >= 2
            min_ok = True
            if args.min_per_class > 0:
                from collections import Counter
                c_tr = Counter(y_tr.tolist())
                c_te = Counter(y_te.tolist())
                min_ok = all(c_tr[c] >= args.min_per_class and c_te[c] >= args.min_per_class for c in set_tr)
            if coverage_ok or not args.resample_if_missing or tries >= args.max_resample_tries:
                if coverage_ok and min_ok:
                    break
                if not args.resample_if_missing or tries >= args.max_resample_tries:
                    break
            tries += 1

        if not coverage_ok and not args.resample_if_missing:
            X_tr, y_tr, X_te, y_te, kept = _drop_missing_classes(X_tr, y_tr, X_te, y_te)
            coverage_ok = True  # by construction

        # baselines / stats
        stats = _class_stats(y_tr, y_te)
        C = len(sorted(set(np.unique(y_tr)).intersection(set(np.unique(y_te)))))
        maj_class = max(stats["test_counts"], key=stats["test_counts"].get)
        maj_acc = stats["test_counts"][maj_class] / sum(stats["test_counts"].values())
        uniform_acc = 1.0 / max(1, C)

        # Fit encoders on anchors
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
            norm_methods=["none"] if diag else (clf.norm_methods or ["none", "power"]),
            feat_shuffle_method="none" if diag else clf.feat_shuffle_method,
            class_shift=False if diag else clf.class_shift,
            outlier_threshold=clf.outlier_threshold,
            random_state=clf.random_state,
        )
        eg.fit(X_tr_num, y_tr_enc)
        clf.ensemble_generator_ = eg
        clf.n_features_in_ = X_tr_num.shape[1]

        # Extract embeddings (optionally try both perm formats when single-estimator)
        def extract_with_format(fmt: str):
            return extract_tf_row_embeddings(
                clf,
                X_te,
                choose_random_variant=False,
                rng=rng,
                feature_perm_format=fmt,
            )

        if args.try_perm_formats and clf.n_estimators == 1:
            try:
                res = extract_with_format("list")
                res2 = extract_with_format("flat")
                # prefer one with higher logreg later; here we just keep both in file
                variants = [("list", res), ("flat", res2)]
            except Exception:
                res = extract_with_format("list")
                variants = [("list", res)]
        else:
            res = extract_with_format("list")
            variants = [("list", res)]

        for fmt_name, res in variants:
            emb_tr = res["embeddings_train"]
            emb_te = res["embeddings_test"]
            y_anchor = np.asarray(res["train_labels"])  # decoded labels
            y_query = np.asarray(y_te)

            # Metrics
            m = eval_metrics(emb_tr, y_anchor, emb_te, y_query, pca_k=args.pca_k)

            # Raw-feature metrics using the same preprocessor used for identity ensemble
            # Find which norm method was used
            method_used = "none" if diag else next(iter(eg.preprocessors_.keys()))
            preproc = eg.preprocessors_[method_used]
            X_tr_pre = preproc.X_transformed_
            X_te_pre = preproc.transform(X_te)
            m.update(eval_raw_feature_metrics(X_tr_pre, y_tr, X_te_pre, y_te, pca_k=args.pca_k))
            m.update({
                "episode": ep,
                "perm_format": fmt_name,
                "coverage_ok": coverage_ok,
                "maj_acc": float(maj_acc),
                "uniform_acc": float(uniform_acc),
                "C": int(C),
                "n_anchor": int(len(y_anchor)),
                "n_query": int(len(y_query)),
            })

            # Append per-class counts (compact)
            m["counts_anchor"] = stats["train_counts"]
            m["counts_query"] = stats["test_counts"]

            rows.append(m)

        # Optionally plot only the first episode (last tried variant)
        if ep == 0:
            emb_tr = res["embeddings_train"]
            emb_te = res["embeddings_test"]
            all_emb = np.vstack([l2_normalize(emb_tr), l2_normalize(emb_te)])
            proj = project_2d(all_emb, method=args.proj, seed=args.seed)
            n_tr = emb_tr.shape[0]
            P_tr, P_te = proj[:n_tr], proj[n_tr:]
            fig = plt.figure(figsize=(7.5, 6))
            vmin, vmax = int(min(y_anchor.min(), y_query.min())), int(max(y_anchor.max(), y_query.max()))
            plt.scatter(P_tr[:, 0], P_tr[:, 1], c=y_anchor, cmap="tab10", vmin=vmin, vmax=vmax, s=18, alpha=0.9, marker="o", label="anchors")
            plt.scatter(P_te[:, 0], P_te[:, 1], c=y_query, cmap="tab10", vmin=vmin, vmax=vmax, s=12, alpha=0.7, marker="x", label="queries")
            plt.title("TF-row embeddings (prior) {}".format(args.proj.upper()))
            plt.legend(loc="best")
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            out_img = out_dir / f"tfrow_prior_{args.proj}.png"
            plt.savefig(out_img, dpi=150)
            plt.close(fig)

    # Write CSV
    fieldnames = sorted({k for r in rows for k in r.keys() if not isinstance(r[k], dict)})
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            # flatten dict-like counts? skip; metrics file keeps primary numbers
            rr = {k: (v if not isinstance(v, dict) else str(v)) for k, v in r.items() if k in fieldnames}
            w.writerow(rr)

    # Summary
    import statistics as st
    def mean_std(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), (int, float))]
        if not vals:
            return None
        return float(np.mean(vals)), float(np.std(vals))

    for k in ["knn1_acc", "knn5_acc", "logreg_acc", "nc_acc", "silhouette_cosine", "silhouette_euclidean"]:
        ms = mean_std(k)
        if ms:
            print(f"{k:>16s}: {ms[0]:.3f} ± {ms[1]:.3f}")

    print("Saved metrics CSV:", csv_path.resolve())


if __name__ == "__main__":
    main()
