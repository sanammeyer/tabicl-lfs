#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable, Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.model_selection import train_test_split


_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tabicl import TabICLClassifier  # noqa: E402


@dataclass(frozen=True)
class ToyDatasetSpec:
    name: str
    make: Callable[[np.random.RandomState], Tuple[np.ndarray, np.ndarray]]


def _setup_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _make_xor(rng: np.random.RandomState, n: int = 300, noise: float = 0.18) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    X = X + rng.normal(scale=noise, size=X.shape).astype(np.float32)
    return X, y


def _make_spirals(rng: np.random.RandomState, n: int = 400, noise: float = 0.12) -> Tuple[np.ndarray, np.ndarray]:
    n0 = n // 2
    n1 = n - n0
    t0 = np.linspace(0.2, 1.0, n0, dtype=np.float32)
    t1 = np.linspace(0.2, 1.0, n1, dtype=np.float32)
    ang0 = 4.0 * np.pi * t0
    ang1 = 4.0 * np.pi * t1 + np.pi
    r0 = t0
    r1 = t1

    x0 = np.stack([r0 * np.cos(ang0), r0 * np.sin(ang0)], axis=1)
    x1 = np.stack([r1 * np.cos(ang1), r1 * np.sin(ang1)], axis=1)
    X = np.concatenate([x0, x1], axis=0).astype(np.float32)
    X = X + rng.normal(scale=noise, size=X.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], axis=0)
    p = rng.permutation(n)
    return X[p], y[p]


def _make_checkerboard(
    rng: np.random.RandomState, n: int = 500, noise: float = 0.06, n_checks: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    X = rng.uniform(-1.0, 1.0, size=(n, 2)).astype(np.float32)
    X = X + rng.normal(scale=noise, size=X.shape).astype(np.float32)
    # label by parity of grid cell
    ix = np.floor((X[:, 0] + 1.0) * 0.5 * n_checks).astype(int)
    iy = np.floor((X[:, 1] + 1.0) * 0.5 * n_checks).astype(int)
    y = ((ix + iy) % 2).astype(int)
    return X, y


def _make_ring(
    rng: np.random.RandomState, n: int = 400, noise: float = 0.08
) -> Tuple[np.ndarray, np.ndarray]:
    n0 = n // 2
    n1 = n - n0
    # inner Gaussian
    x0 = rng.normal(scale=0.25 + 0.5 * noise, size=(n0, 2))
    # outer ring
    theta = rng.uniform(0, 2 * np.pi, size=(n1,))
    radius = rng.normal(loc=1.0, scale=0.08 + 0.5 * noise, size=(n1,))
    x1 = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    X = np.concatenate([x0, x1], axis=0).astype(np.float32)
    X = X + rng.normal(scale=noise, size=X.shape).astype(np.float32)
    y = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)], axis=0)
    p = rng.permutation(n)
    return X[p], y[p]


def _default_toy_specs(n_samples: int, noise: float) -> List[ToyDatasetSpec]:
    def moons(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=rng.randint(0, 1_000_000))
        return X.astype(np.float32), y.astype(int)

    def circles(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=rng.randint(0, 1_000_000)
        )
        return X.astype(np.float32), y.astype(int)

    def blobs(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[(-0.8, -0.6), (0.9, 0.8)],
            cluster_std=[0.55 + noise, 0.55 + noise],
            random_state=rng.randint(0, 1_000_000),
        )
        return X.astype(np.float32), y.astype(int)

    def aniso_blobs(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        X, y = make_blobs(
            n_samples=n_samples,
            centers=[(-0.5, 0.0), (0.8, 0.0)],
            cluster_std=[0.45 + noise, 0.45 + noise],
            random_state=rng.randint(0, 1_000_000),
        )
        A = np.array([[0.6, -0.8], [0.4, 0.9]], dtype=np.float32)
        X = (X.astype(np.float32) @ A.T).astype(np.float32)
        return X, y.astype(int)

    def xor(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        return _make_xor(rng, n=n_samples, noise=max(0.05, 0.9 * noise))

    def spirals(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        return _make_spirals(rng, n=n_samples, noise=max(0.03, 0.7 * noise))

    def checkerboard(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        return _make_checkerboard(rng, n=n_samples, noise=max(0.02, 0.6 * noise), n_checks=4)

    def ring(rng: np.random.RandomState) -> Tuple[np.ndarray, np.ndarray]:
        return _make_ring(rng, n=n_samples, noise=max(0.02, 0.7 * noise))

    return [
        ToyDatasetSpec("Two Moons", moons),
        ToyDatasetSpec("Circles", circles),
        ToyDatasetSpec("Blobs", blobs),
        ToyDatasetSpec("Anisotropic Blobs", aniso_blobs),
        ToyDatasetSpec("XOR", xor),
        ToyDatasetSpec("Ring", ring),
        ToyDatasetSpec("Spirals", spirals),
        ToyDatasetSpec("Checkerboard", checkerboard),
    ]


def _meshgrid(X: np.ndarray, h: float) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = float(X[:, 0].min() - 0.6), float(X[:, 0].max() + 0.6)
    y_min, y_max = float(X[:, 1].min() - 0.6), float(X[:, 1].max() + 0.6)
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


def _predict_proba_chunks(
    clf: TabICLClassifier, X_df: pd.DataFrame, chunk_size: int
) -> np.ndarray:
    if chunk_size <= 0:
        return clf.predict_proba(X_df)
    out: List[np.ndarray] = []
    for i in range(0, len(X_df), chunk_size):
        out.append(clf.predict_proba(X_df.iloc[i : i + chunk_size]))
    return np.concatenate(out, axis=0)


def _plot_panel(
    ax: plt.Axes,
    clf: TabICLClassifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    h: float,
    chunk_size: int,
    title: str,
) -> None:
    xx, yy = _meshgrid(np.vstack([X_train, X_test]), h=h)
    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

    cols = ["x0", "x1"]
    Xtr_df = pd.DataFrame(X_train, columns=cols)
    Xte_df = pd.DataFrame(X_test, columns=cols)
    grid_df = pd.DataFrame(grid, columns=cols)

    clf.fit(Xtr_df, y_train)
    proba = _predict_proba_chunks(clf, grid_df, chunk_size=chunk_size)[:, 1]
    Z = proba.reshape(xx.shape)

    cmap = plt.cm.RdBu
    ax.contourf(xx, yy, Z, cmap=cmap, alpha=0.85, levels=40, vmin=0.0, vmax=1.0)
    ax.contour(xx, yy, Z, levels=[0.5], colors="k", linewidths=0.9, alpha=0.9)

    # Marker style encodes train/test split; point color encodes the class.
    train_colors = cmap(y_train.astype(np.float32))
    test_colors = cmap(y_test.astype(np.float32))

    ax.scatter(
        X_train[:, 0],
        X_train[:, 1],
        facecolors=train_colors,
        edgecolors="k",
        s=18,
        linewidths=0.25,
        alpha=0.95,
        label="train (filled)",
    )
    ax.scatter(
        X_test[:, 0],
        X_test[:, 1],
        facecolors="none",
        edgecolors=test_colors,
        s=26,
        marker="o",
        linewidths=1.0,
        alpha=0.85,
        label="test (hollow)",
    )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal", adjustable="box")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Plot toy decision boundaries comparing TabICL vs TabICL+TabPDL (PDLC head)."
    )
    p.add_argument(
        "--tabicl_ckpt",
        type=str,
        default=str(_REPO_ROOT / "checkpoints/mini_tabicl_stage2_ea_icl_only/step-1000.ckpt"),
        help="Path to a TabICL checkpoint (icl_head='tabicl').",
    )
    p.add_argument(
        "--tabpdl_ckpt",
        type=str,
        default=str(_REPO_ROOT / "checkpoints/mini_tabicl_stage2_pdl_posterior_avg/step-1000.ckpt"),
        help="Path to a TabICL checkpoint trained with icl_head='tabpdl' (PDLC head).",
    )
    p.add_argument(
        "--out",
        type=str,
        default=str(_REPO_ROOT / "figures/decision_boundary_toy_tabicl_vs_tabpdl.pdf"),
        help="Output path (.pdf/.png).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=300)
    p.add_argument("--noise", type=float, default=0.20)
    p.add_argument("--test_size", type=float, default=0.35)
    p.add_argument("--grid_step", type=float, default=0.08, help="Mesh step size (smaller = finer, slower).")
    p.add_argument(
        "--chunk_size",
        type=int,
        default=1024,
        help="Number of grid points per predict_proba call (<=0 disables chunking).",
    )
    p.add_argument("--n_estimators", type=int, default=8)
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument(
        "--datasets",
        type=str,
        default="",
        help="Optional comma-separated subset of dataset names to plot (matches the left labels).",
    )
    args = p.parse_args()

    _setup_style()
    rng = np.random.RandomState(args.seed)

    specs_all = _default_toy_specs(n_samples=args.n_samples, noise=args.noise)
    if str(args.datasets).strip():
        wanted = {s.strip().lower() for s in str(args.datasets).split(",") if s.strip()}
        specs = [s for s in specs_all if s.name.lower() in wanted]
        if not specs:
            raise ValueError(f"--datasets did not match any known names. Options: {[s.name for s in specs_all]}")
    else:
        specs = specs_all

    def make_clf(model_path: str) -> TabICLClassifier:
        device = None
        if args.device == "cpu":
            device = "cpu"
        elif args.device == "cuda":
            device = "cuda"
        return TabICLClassifier(
            n_estimators=args.n_estimators,
            norm_methods=["none"],
            feat_shuffle_method="none",
            class_shift=False,
            batch_size=1,
            use_amp=False,
            model_path=model_path,
            allow_auto_download=False,
            device=device,
            verbose=False,
        )

    clfs: Dict[str, TabICLClassifier] = {
        "TabICL": make_clf(args.tabicl_ckpt),
        "TabICL + PDLC head": make_clf(args.tabpdl_ckpt),
    }

    fig, axes = plt.subplots(
        nrows=len(clfs),
        ncols=len(specs),
        figsize=(3.0 * len(specs), 2.8 * len(clfs)),
        constrained_layout=True,
    )
    if len(clfs) == 1 and len(specs) == 1:
        axes = np.array([[axes]])
    elif len(clfs) == 1:
        axes = np.array([axes])
    elif len(specs) == 1:
        axes = np.array([[a] for a in axes])

    for col, spec in enumerate(specs):
        X, y = spec.make(rng)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=rng.randint(0, 1_000_000), stratify=y
        )
        for row, (model_name, clf) in enumerate(clfs.items()):
            ax = axes[row, col]
            title = spec.name if row == 0 else ""
            _plot_panel(
                ax=ax,
                clf=clf,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                h=float(args.grid_step),
                chunk_size=int(args.chunk_size),
                title=title,
            )
            if col == 0:
                ax.set_ylabel(model_name)

    split_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="k",
            markerfacecolor="k",
            markeredgecolor="k",
            markersize=6,
            label="train (filled)",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            color="k",
            markerfacecolor="none",
            markeredgecolor="k",
            markersize=6,
            label="test (hollow)",
        ),
    ]
    fig.legend(
        split_handles,
        [h.get_label() for h in split_handles],
        loc="lower center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, -0.005),
        title="Split (point color = class; background = P(y=1))",
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
