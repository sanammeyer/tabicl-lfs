from __future__ import annotations

from typing import Dict, Any, Optional

import numpy as np
import torch

from tabicl import TabICLClassifier


def extract_tf_row_embeddings(
    clf: TabICLClassifier,
    X_test_df,
    *,
    choose_random_variant: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """Return row interaction (tf_row) embeddings for a single ensemble variant.

    Mirrors the logic used in scripts/extract_tfrow_iris.py but as a reusable function.

    Returns a dict containing embeddings for train/test parts and metadata.
    """

    assert hasattr(clf, "ensemble_generator_"), "Classifier must be fitted first."

    # Transform test data through fitted encoder
    X_test_num = clf.X_encoder_.transform(X_test_df)

    # Build ensemble datasets
    data = clf.ensemble_generator_.transform(X_test_num)

    # Optionally randomize normalization method and variant index to expose invariances
    methods = list(data.keys())
    if choose_random_variant:
        rng = rng or np.random.default_rng()
        norm_method = methods[rng.integers(0, len(methods))]
    else:
        norm_method = methods[0]

    Xs, ys_shifted = data[norm_method]

    shuffle_patterns = clf.ensemble_generator_.feature_shuffle_patterns_[norm_method]
    shift_offsets = clf.ensemble_generator_.class_shift_offsets_[norm_method]

    if choose_random_variant:
        rng = rng or np.random.default_rng()
        variant_index = int(rng.integers(0, len(shuffle_patterns)))
    else:
        # Prefer identity permutation if present
        variant_index = None
        for i, pattern in enumerate(shuffle_patterns):
            if list(pattern) == sorted(pattern):
                variant_index = i
                break
        if variant_index is None:
            variant_index = 0

    X_variant = Xs[variant_index]
    y_variant_shifted = ys_shifted[variant_index]
    shift_offset = shift_offsets[variant_index]

    # Reverse class shift and recover original labels
    y_variant = (y_variant_shifted - shift_offset) % clf.n_classes_
    y_variant = clf.y_encoder_.inverse_transform(y_variant.astype(int))

    train_size = y_variant_shifted.shape[0]
    T = X_variant.shape[0]
    test_size = T - train_size

    # Send through backbone (pre-ICL): col_embedder -> row_interactor
    model = clf.model_
    model.eval()
    device = clf.device_
    X_tensor = torch.from_numpy(X_variant).float().unsqueeze(0).to(device)
    inference_config = clf.inference_config_

    with torch.no_grad():
        # Provide the feature permutation used for this variant to the embedder
        feature_perm = [list(shuffle_patterns[variant_index])]
        col_out = model.col_embedder(
            X_tensor,
            train_size=train_size,
            feature_shuffles=feature_perm,
            mgr_config=inference_config.COL_CONFIG,
        )
        row_reps = model.row_interactor(col_out, mgr_config=inference_config.ROW_CONFIG)

    embeddings = row_reps.squeeze(0).cpu().numpy()

    return {
        "norm_method": norm_method,
        "variant_index": variant_index,
        "feature_permutation": list(shuffle_patterns[variant_index]),
        "class_shift_offset": int(shift_offset),
        "train_labels": y_variant,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "embeddings_all": embeddings,
        "embeddings_train": embeddings[:train_size],
        "embeddings_test": embeddings[train_size:],
        "embedding_dim": int(embeddings.shape[1]),
        "model_checkpoint": getattr(clf, "model_path_", None),
    }
