"""Train PDLC head on synthetic datasets sampled from TabICL prior.

Each episode:
- Sample one synthetic dataset via SCM prior (diverse features/classes/lengths)
- Split anchors/queries using the prior's train_size (already sanity-checked)
- Fit TabICLClassifier on anchors to load transforms/backbone (frozen)
- Extract pre-ICL TF-row embeddings for anchors+queries using the same pipeline
- Train PDLC head on balanced anchor pairs; validate with PDLC update on queries

Example:
    python scripts/pdlc_prior_train.py --episodes 200 --pairs 8192 --topk 256 --prior mix_scm
"""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import os

import numpy as np
import torch

from tabicl import TabICLClassifier
from tabicl.pdlc.embed import extract_tf_row_embeddings
from tabicl.pdlc.head import (
    PDLCHead,
    TrainConfig,
    build_balanced_pairs,
    l2_normalize,
    pdlc_posteriors,
    nll_accuracy,
    pair_auc_head,
)
from tabicl.prior.dataset import PriorDataset


def main():
    p = argparse.ArgumentParser()
    # Training config
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--pairs", type=int, default=4096, help="unique pairs per episode (duplicated by symmetry)")
    p.add_argument("--topk", type=int, default=256, help="top-K anchors per query for PDLC (0 disables)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--symmetry_weight", type=float, default=0.1)
    p.add_argument("--epochs_per_episode", type=int, default=3)
    p.add_argument("--head_hidden", type=int, default=256)
    p.add_argument("--head_dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", type=str, default="models/pdlc_head_prior.pt")
    p.add_argument("--model_path", type=str, default=None, help="Optional TabICL checkpoint path to force local load")
    p.add_argument("--log_csv", type=str, default="", help="Optional path to append per-episode metrics as CSV")
    # Episode quality controls
    p.add_argument("--min_anchors_per_class", type=int, default=2)
    p.add_argument("--min_queries_per_class", type=int, default=1)
    p.add_argument("--hard_neg_frac", type=float, default=0.5)
    p.add_argument("--randomize_variant", action="store_true", help="randomize ensemble norm/shuffle variant per episode")
    p.add_argument("--report_pair_auc", action="store_true", help="report head/cosine pair AUC on sampled eval pairs")
    p.add_argument("--warmup_episodes", type=int, default=0, help="episodes to warm-start by mimicking cosine similarity")
    p.add_argument("--mimic_weight", type=float, default=0.0, help="weight of cosine-mimic loss during warmup")
    p.add_argument("--max_anchor_class_ratio", type=float, default=0.7, help="skip episodes with anchors too imbalanced (max class ratio)")
    # Prior config (kept minimal; for advanced control tune src/tabicl/prior/prior_config.py)
    p.add_argument("--prior", type=str, default="mix_scm", choices=["mlp_scm", "tree_scm", "mix_scm", "dummy"], help="which prior to sample episodes from")
    p.add_argument("--min_features", type=int, default=4)
    p.add_argument("--max_features", type=int, default=32)
    p.add_argument("--max_classes", type=int, default=6)
    p.add_argument("--min_seq_len", type=int, default=120)
    p.add_argument("--max_seq_len", type=int, default=400)
    p.add_argument("--log_seq_len", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Set up prior episode generator
    prior_gen = PriorDataset(
        batch_size=1,
        prior_type=args.prior,
        min_features=args.min_features,
        max_features=args.max_features,
        max_classes=args.max_classes,
        min_seq_len=args.min_seq_len,
        max_seq_len=args.max_seq_len,
        log_seq_len=args.log_seq_len,
        device="cpu",  # generation on CPU
    )

    head = None
    best_nll = float("inf")
    best_state = None
    opt = None

    # Initialize a single TabICLClassifier and load backbone once
    clf = TabICLClassifier(
        n_estimators=4,
        batch_size=4,
        use_amp=True,
        verbose=False,
        model_path=args.model_path,
        allow_auto_download=True if args.model_path is None else False,
    )
    # Manually load model and prepare inference config once
    # Select device
    if clf.device is None:
        clf.device_ = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    elif isinstance(clf.device, str):
        clf.device_ = torch.device(clf.device)
    else:
        clf.device_ = clf.device
    clf._load_model()
    clf.model_.to(clf.device_)
    # Inference configuration mirrors TabICLClassifier.fit logic
    init_cfg = {
        "COL_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
        "ROW_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
        "ICL_CONFIG": {"device": clf.device_, "use_amp": clf.use_amp, "verbose": clf.verbose},
    }
    from tabicl import InferenceConfig
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

    # CSV logging setup
    log_path = Path(args.log_csv) if args.log_csv else None
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not log_path.exists() or os.stat(log_path).st_size == 0
        if write_header:
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "episode",
                    "pair_loss",
                    "pdlc_nll",
                    "pdlc_acc",
                    "prior_nll",
                    "auc_head",
                    "auc_cos",
                    "C",
                    "H",
                    "T",
                    "split",
                    "topk",
                    "epochs_per_episode",
                    "pairs_per_epoch",
                    "hard_neg_frac",
                    "randomize_variant",
                    "warmup",
                    "mimic_weight_used",
                    "min_anchors_per_class_used",
                    "min_queries_per_class",
                    "anchor_max_ratio",
                ])
    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    processed = 0
    max_tries = args.episodes * 10
    tries = 0
    while processed < args.episodes and tries < max_tries:
        tries += 1
        # 1) Sample one synthetic dataset
        X, y, d, seq_lens, train_sizes = prior_gen.get_batch(batch_size=1)
        T = int(seq_lens[0].item())
        H = int(d[0].item())
        split = int(train_sizes[0].item())
        X = X[0, :T, :H].cpu().numpy()
        y = y[0, :T].cpu().numpy()

        # 2) Split anchors/queries
        X_tr, y_tr = X[:split], y[:split]
        X_te, y_te = X[split:], y[split:]

        # Enforce per-class minima; resample if violated
        ok = True
        u_tr, c_tr = np.unique(y_tr, return_counts=True)
        u_te, c_te = np.unique(y_te, return_counts=True)
        classes = np.unique(y)
        # Require coverage in both splits
        if not set(classes).issubset(set(u_tr)) or not set(classes).issubset(set(u_te)):
            ok = False
        # Require min anchors and queries per class
        count_tr = {cls: int(c_tr[u_tr.tolist().index(cls)]) if cls in u_tr else 0 for cls in classes}
        count_te = {cls: int(c_te[u_te.tolist().index(cls)]) if cls in u_te else 0 for cls in classes}
        min_anchor_req = args.min_anchors_per_class
        if len(classes) >= 4:
            min_anchor_req = max(min_anchor_req, args.min_anchors_per_class + 1)
        if any(v < min_anchor_req for v in count_tr.values()):
            ok = False
        if any(v < args.min_queries_per_class for v in count_te.values()):
            ok = False
        # Skip overly imbalanced anchors (dominant class > threshold)
        total_tr = sum(count_tr.values())
        if total_tr > 0 and max(count_tr.values()) / total_tr > args.max_anchor_class_ratio:
            ok = False
        if not ok:
            continue  # resample a new dataset

        # 3) Prepare episode-specific preprocessing using the same frozen backbone
        # Mirror minimal parts of TabICLClassifier.fit without re-loading the model
        # Encode labels
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_tr)
        clf.y_encoder_ = le
        clf.classes_ = le.classes_
        clf.n_classes_ = len(le.classes_)

        # Transform features to numerical
        from tabicl.sklearn.preprocessing import TransformToNumerical, EnsembleGenerator
        X_encoder = TransformToNumerical(verbose=clf.verbose)
        X_tr_num = X_encoder.fit_transform(X_tr)
        clf.X_encoder_ = X_encoder

        # Fit ensemble generator on anchors
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
        # sklearn compatibility: set n_features_in_
        clf.n_features_in_ = X_tr_num.shape[1]

        # 4) Extract TF-row embeddings for anchors+queries
        # During warmup, avoid randomizing variant to reduce distribution shift
        do_rand_variant = args.randomize_variant and processed >= args.warmup_episodes
        res = extract_tf_row_embeddings(
            clf,
            X_te,
            choose_random_variant=do_rand_variant,
            rng=rng,
        )
        emb_tr = l2_normalize(res["embeddings_train"])  # anchors
        emb_te = l2_normalize(res["embeddings_test"])   # queries
        y_anchor = np.asarray(res["train_labels"])      # decoded labels
        y_query = np.asarray(y_te)

        # 5) Init head lazily
        if head is None:
            head = PDLCHead(emb_dim=emb_tr.shape[1], hidden=args.head_hidden, dropout=args.head_dropout).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # 6) Train on balanced pairs from anchors
        # Build a fixed eval pair set to measure pre/post training AUC
        from tabicl.pdlc.head import pair_auc_on_pairs as _auc_pairs
        eval_ab, eval_ba, eval_t = build_balanced_pairs(
            emb_tr,
            y_anchor,
            num_pairs=min(4096, max(1024, args.pairs // max(1, args.epochs_per_episode))),
            rng=rng,
            class_balance=True,
            hard_neg_frac=args.hard_neg_frac,
        )
        auc_head_pre, auc_cos_pre = _auc_pairs(head, eval_ab, eval_ba, eval_t, device)

        # Train multiple mini-epochs per episode, resampling pairs each epoch
        losses = []
        cfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size, symmetry_weight=args.symmetry_weight)
        from tabicl.pdlc.head import train_head_on_pairs as _train
        pairs_per_epoch = max(1, args.pairs // max(1, args.epochs_per_episode))
        mimic_w = args.mimic_weight if processed < args.warmup_episodes else 0.0
        for _epk in range(max(1, args.epochs_per_episode)):
            ab, ba, t = build_balanced_pairs(
                emb_tr,
                y_anchor,
                num_pairs=pairs_per_epoch,
                rng=rng,
                class_balance=True,
                hard_neg_frac=args.hard_neg_frac,
            )
            loss = _train(head, ab, ba, t, cfg, device, optimizer=opt, epochs=1, mimic_weight=mimic_w)
            losses.append(loss)
        loss = float(np.mean(losses))

        # Post-training AUC on the same eval pairs
        auc_head_post, auc_cos_post = _auc_pairs(head, eval_ab, eval_ba, eval_t, device)

        # 7) PDLC validation
        topk = None if args.topk <= 0 else min(args.topk, emb_tr.shape[0])
        posts, classes_arr = pdlc_posteriors(head, emb_tr, y_anchor, emb_te, topk=topk, device=device)
        nll, acc = nll_accuracy(posts, y_query, classes_arr)

        # Prior-only baseline NLL (anchor frequency prior)
        prior_classes, prior_counts = np.unique(y_anchor, return_counts=True)
        prior = prior_counts / prior_counts.sum()
        class_to_idx = {c: i for i, c in enumerate(prior_classes.tolist())}
        idx = np.array([class_to_idx.get(t, -1) for t in y_query])
        mask = idx >= 0
        prior_nll = float(
            -np.log(prior[idx[mask]] + 1e-12).mean()
        ) if mask.any() else float('nan')

        # Optional: pair AUCs (head vs cosine)
        auc_head = auc_cos = None
        if args.report_pair_auc:
            auc_head, auc_cos = pair_auc_head(head, emb_tr, y_anchor, rng=rng, device=device, num_eval_pairs=4096)

        processed += 1
        ep = processed
        if args.report_pair_auc and auc_head is not None and auc_cos is not None:
            print(
                f"[Episode {ep:03d}] pair-loss={loss:.4f}  PDLC: NLL={nll:.4f}  acc={acc:.3f}  priorNLL={prior_nll:.4f}  "
                f"AUC(head)={auc_head:.3f}  AUC(cos)={auc_cos:.3f}  AUC(head_pre)={auc_head_pre:.3f}  AUC(head_post)={auc_head_post:.3f}  (C={len(classes_arr)}, H={H}, T={T}, split={split})"
            )
        else:
            print(
                f"[Episode {ep:03d}] pair-loss={loss:.4f}  PDLC: NLL={nll:.4f}  acc={acc:.3f}  priorNLL={prior_nll:.4f}  (C={len(classes_arr)}, H={H}, T={T}, split={split})"
            )

        # Append CSV log row if requested
        if log_path:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    ep,
                    f"{loss:.6f}",
                    f"{nll:.6f}",
                    f"{acc:.6f}",
                    f"{prior_nll:.6f}",
                    "" if auc_head is None else f"{auc_head:.6f}",
                    "" if auc_cos is None else f"{auc_cos:.6f}",
                    len(classes_arr),
                    H,
                    T,
                    split,
                    0 if topk is None else topk,
                    args.epochs_per_episode,
                    pairs_per_epoch,
                    args.hard_neg_frac,
                    bool(args.randomize_variant),
                    bool(processed <= args.warmup_episodes),
                    0.0 if processed > args.warmup_episodes else args.mimic_weight,
                    min_anchor_req,
                    args.min_queries_per_class,
                    round(max(count_tr.values()) / sum(count_tr.values()), 6) if sum(count_tr.values()) > 0 else "",
                ])

        if np.isfinite(nll) and nll < best_nll:
            best_nll = nll
            best_state = {"model": head.state_dict(), "emb_dim": head.emb_dim}
            torch.save(best_state, out_path)

    if best_state is not None:
        print(f"Saved best PDLC head to {out_path.resolve()} (best NLL={best_nll:.4f})")
    else:
        print("Done. No finite NLL recorded; nothing saved.")


if __name__ == "__main__":
    main()
