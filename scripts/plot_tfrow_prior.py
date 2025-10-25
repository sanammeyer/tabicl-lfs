"""Plot TF-row embeddings on a random prior episode to sanity-check separability.

Saves a 2D projection (t-SNE or PCA) of anchor and query embeddings, colored by class.
Optionally fits a classifier (logistic regression or XGBoost) on anchor embeddings and evaluates on queries.

Example:
    # Plot + t-SNE, fit logistic regression, and save embeddings
    python scripts/plot_tfrow_prior.py --prior mix_scm --n_estimators 4 --proj tsne \
        --max_points 2000 --fit_logreg --save_npz runs/tfrow_embeddings.npz \
        --out runs/tfrow_sanity.png
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
    p.add_argument("--proj", type=str, default="tsne", choices=["tsne", "pca"]) 
    p.add_argument("--max_points", type=int, default=2000, help="max points to plot for readability")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/tfrow_sanity.png")

    # Classifier on embeddings
    p.add_argument("--fit_logreg", action="store_true", help="[deprecated] kept for backward-compat; prefer --clf")
    p.add_argument("--clf", type=str, default="none", choices=["none", "logreg", "xgb"], help="classifier to train on embeddings")
    # LR hyperparams
    p.add_argument("--logreg_C", type=float, default=1.0)
    p.add_argument("--logreg_max_iter", type=int, default=1000)
    # XGBoost hyperparams
    p.add_argument("--xgb_estimators", type=int, default=300, help="number of boosting rounds (n_estimators)")
    p.add_argument("--xgb_max_depth", type=int, default=6)
    p.add_argument("--xgb_lr", type=float, default=0.1, help="learning rate (eta)")
    p.add_argument("--xgb_subsample", type=float, default=0.8)
    p.add_argument("--xgb_colsample", type=float, default=0.8, help="colsample_bytree")
    p.add_argument("--xgb_reg_lambda", type=float, default=1.0)
    p.add_argument("--xgb_min_child_weight", type=float, default=1.0)
    p.add_argument("--xgb_tree_method", type=str, default="hist", help="tree_method: hist, approx, gpu_hist (if available)")
    # Save artifacts
    p.add_argument("--save_npz", type=str, default="", help="optional .npz path to save embeddings+labels+metadata")
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

    # Optionally fit a classifier on embeddings (anchors -> train, queries -> test)
    clf_info = None
    chosen_clf = args.clf
    if args.fit_logreg:
        chosen_clf = "logreg"  # backward-compat
    if chosen_clf in {"logreg", "xgb"}:
        try:
            from sklearn.metrics import accuracy_score, f1_score, log_loss
            if chosen_clf == "logreg":
                from sklearn.pipeline import make_pipeline
                from sklearn.preprocessing import StandardScaler
                from sklearn.linear_model import LogisticRegression
                model = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(C=args.logreg_C, max_iter=args.logreg_max_iter, multi_class="auto"),
                )
                model.fit(emb_tr, y_anchor)
                y_pred_tr = model.predict(emb_tr)
                y_pred_te = model.predict(emb_te)
                proba_te = model.predict_proba(emb_te)
                est_classes = model[-1].classes_
            else:
                # XGBoost classifier
                try:
                    import xgboost as xgb  # type: ignore
                except Exception as e:
                    raise RuntimeError("xgboost is required for --clf xgb. Install via 'pip install xgboost' or 'conda install -c conda-forge xgboost'.") from e

                from sklearn.preprocessing import LabelEncoder
                le_clf = LabelEncoder().fit(y_anchor)
                y_tr_enc = le_clf.transform(y_anchor)
                num_classes = len(le_clf.classes_)
                objective = "binary:logistic" if num_classes == 2 else "multi:softprob"
                model = xgb.XGBClassifier(
                    n_estimators=args.xgb_estimators,
                    max_depth=args.xgb_max_depth,
                    learning_rate=args.xgb_lr,
                    subsample=args.xgb_subsample,
                    colsample_bytree=args.xgb_colsample,
                    reg_lambda=args.xgb_reg_lambda,
                    min_child_weight=args.xgb_min_child_weight,
                    objective=objective,
                    eval_metric="logloss",
                    tree_method=args.xgb_tree_method,
                    random_state=args.seed,
                    n_jobs=0,
                )
                model.fit(emb_tr, y_tr_enc)
                # predictions
                y_pred_int = model.predict(emb_te).astype(int)
                from numpy import asarray as _np_asarray  # avoid shadowing
                y_pred_te = le_clf.inverse_transform(_np_asarray(y_pred_int))
                # proba: ensure 2D
                proba_te = model.predict_proba(emb_te)
                if proba_te.ndim == 1:
                    p1 = proba_te.reshape(-1, 1)
                    proba_te = np.concatenate([1.0 - p1, p1], axis=1)
                est_classes = le_clf.classes_
                # Train predictions for train-acc
                y_pred_tr_int = model.predict(emb_tr).astype(int)
                y_pred_tr = le_clf.inverse_transform(y_pred_tr_int)

            acc_tr = float(accuracy_score(y_anchor, y_pred_tr))
            acc_te = float(accuracy_score(y_query, y_pred_te))
            f1_te = float(f1_score(y_query, y_pred_te, average="macro"))
            # Log-loss on seen test classes only (skip unseen)
            seen_mask = np.isin(y_query, est_classes)
            if np.any(seen_mask):
                nll_te = float(log_loss(y_query[seen_mask], proba_te[seen_mask], labels=est_classes))
            else:
                nll_te = float("nan")
            clf_info = {
                "name": chosen_clf,
                "acc_tr": acc_tr,
                "acc_te": acc_te,
                "f1_te": f1_te,
                "nll_te": nll_te,
                "classes_": est_classes,
            }
            print(
                f"{chosen_clf.upper()} on embeddings: train-acc={acc_tr:.3f}  test-acc={acc_te:.3f}  test-mF1={f1_te:.3f}  test-NLL(seen)={nll_te:.4f}"
            )
        except Exception as e:
            print(f"[warn] {chosen_clf} classifier failed: {e}")

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
    title = f"TF-row embeddings ({args.proj.upper()})  C={len(np.unique(y))}, H={H}, T={T}, split={split}"
    if clf_info is not None:
        title += f"\n{clf_info['name'].upper()} test-acc={clf_info['acc_te']:.3f}  test-mF1={clf_info['f1_te']:.3f}"
    plt.title(title)
    plt.legend(loc="best")
    plt.xticks([]); plt.yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    print(f"Saved TF-row embedding plot to {out_path.resolve()}")

    # Optionally save embeddings, labels, and metadata
    if args.save_npz:
        npz_path = Path(args.save_npz)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "emb_tr": emb_tr,
            "emb_te": emb_te,
            "y_tr": np.asarray(y_anchor),
            "y_te": np.asarray(y_query),
            "proj_tr": P_tr,
            "proj_te": P_te,
            "proj_method": args.proj,
            "seed": int(args.seed),
            "H": int(H),
            "T": int(T),
            "split": int(split),
            "C": int(len(np.unique(y))),
        }
        if clf_info is not None:
            save_dict.update({
                "clf": clf_info["name"],
                "clf_acc_tr": clf_info["acc_tr"],
                "clf_acc_te": clf_info["acc_te"],
                "clf_f1_te": clf_info["f1_te"],
                "clf_nll_te": clf_info["nll_te"],
            })
        np.savez_compressed(npz_path, **save_dict)
        print(f"Saved embeddings + labels to {npz_path.resolve()}")


if __name__ == "__main__":
    main()
