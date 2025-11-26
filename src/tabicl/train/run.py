from __future__ import annotations

import os
import timeit
import warnings
import functools
from contextlib import nullcontext

import math
import numpy as np

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.multiprocessing import set_start_method
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from tqdm import tqdm
import wandb

from tabicl import TabICL
from tabicl.prior.dataset import PriorDataset
from tabicl.prior.genload import LoadPriorDataset
from tabicl.train.optim import get_scheduler
from tabicl.train.train_config import build_parser
import csv

warnings.filterwarnings(
    "ignore", message=".*The PyTorch API of nested tensors is in prototype stage.*", category=UserWarning
)


class Timer:
    """Context manager for timing code execution."""

    def __enter__(self):
        self.start_time = timeit.default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = timeit.default_timer() - self.start_time
        return False  # Don't suppress exceptions


def ddp_cleanup(func):
    """Decorator to clean up DDP process group after method execution.

    Ensures that destroy_process_group() is called if DDP is enabled,
    even if an exception occurs during method execution.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        finally:
            if self.ddp:
                destroy_process_group()

    return wrapper


class Trainer:
    """This class handles the complete training lifecycle for TabICL, including:

    - Environment setup and distributed training configuration
    - Model building and initialization
    - Optimizer, scheduler, and dataloader configuration
    - Checkpoint management and recovery
    - Training loop execution with gradient accumulation
    - Metrics tracking and logging using wandb

    Parameters
    ----------
    config : argparse.Namespace
        Training configuration parameters containing all settings for model,
        optimizer, distributed training, and data generation.
    """

    def __init__(self, config):
        self.config = config
        self.configure_ddp()
        self.configure_wandb()
        self.build_model()
        self.configure_prior()
        # Probe: materialize once if configured
        self._probe_data = None
        if getattr(self.config, "probe_every", 0) and self.config.probe_every > 0:
            try:
                self.materialize_probe_batch()
            except Exception as e:
                print(f"Warning: failed to materialize probe batch: {e}")
        self.configure_optimizer()
        self.configure_amp()
        self.load_checkpoint()
        # Local CSV logging (master only)
        self.metrics_csv_path = getattr(self.config, "metrics_csv", None)
        self.csv_fields = [
            "step",
            "ce",
            "accuracy",
            "lr",
            "prior_time",
            "train_time",
            # Probe metrics (may be empty when not on probe step)
            "probe/attn_kl_mean",
            "probe/attn_entropy_mean",
            "probe/delta_ce",
            "probe/ce",
            "probe/chance_entropy",
        ]

    def configure_ddp(self):
        """Set up distributed training and system configuration.

        This method:
        1. Configures distributed data parallel (DDP) if enabled
        2. Sets up device and process information
        3. Adjusts batch size for multi-GPU training
        4. Sets random seeds for reproducibility
        """
        # Setup distributed training
        self.ddp = int(os.environ.get("RANK", -1)) != -1

        if self.ddp:
            init_process_group(backend="nccl")
            self.ddp_rank = int(os.environ["RANK"])
            self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
            self.ddp_world_size = int(os.environ["WORLD_SIZE"])
            self.master_process = self.ddp_rank == 0
            self.config.device = f"cuda:{self.ddp_local_rank}"
            torch.cuda.set_device(self.config.device)

            # Adjust batch size for distributed training
            original_batch_size = self.config.batch_size
            self.config.batch_size = math.ceil(original_batch_size / self.ddp_world_size)

            if self.master_process:
                print(f"DDP training with {self.ddp_world_size} processes")
                if original_batch_size % self.ddp_world_size == 0:
                    print(f"Per-GPU batch size: {self.config.batch_size}")
                else:
                    print(
                        f"Original batch size ({original_batch_size}) cannot be divided by world size ({self.ddp_world_size}).\n"
                        f"Use ceiling division for equal per-GPU batch size: {self.config.batch_size}.\n"
                        f"Effective batch size is {self.config.batch_size * self.ddp_world_size}.\n"
                    )
        else:
            self.master_process = True
            self.ddp_rank = 0
            self.ddp_world_size = 1
            self.ddp_local_rank = 0
            print("No DDP training")

        self.curr_step = 0  # Initialize current step for training

        # Set random seeds
        seed_offset = self.ddp_rank if self.ddp else 0
        np.random.seed(self.config.np_seed + seed_offset)
        torch.manual_seed(self.config.torch_seed + seed_offset)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    def configure_wandb(self):
        """Set up Weights & Biases logging."""

        if self.config.wandb_log and self.master_process:
            id_path = os.path.join(self.config.checkpoint_dir, "wand_id.txt")
            if self.config.wandb_id is None:
                if os.path.exists(id_path):
                    with open(id_path, "r") as f:
                        self.config.wandb_id = f.read().strip()

            self.wandb_run = wandb.init(
                dir=self.config.wandb_dir,
                project=self.config.wandb_project,
                name=self.config.wandb_name,
                id=self.config.wandb_id,
                config=self.config,
                resume="allow",
                mode=self.config.wandb_mode,
            )

            with open(id_path, "w") as f:
                f.write(self.wandb_run.id)
        else:
            self.wandb_run = None

    def build_model(self):
        """Build and initialize the TabICL model."""

        self.model_config = {
            "max_classes": self.config.max_classes,
            "embed_dim": self.config.embed_dim,
            "col_num_blocks": self.config.col_num_blocks,
            "col_nhead": self.config.col_nhead,
            "col_num_inds": self.config.col_num_inds,
            # Backward compatibility: fall back to old --col_elliptical if present
            "row_elliptical": getattr(self.config, "row_elliptical", getattr(self.config, "col_elliptical", False)),
            "row_num_blocks": self.config.row_num_blocks,
            "row_nhead": self.config.row_nhead,
            "row_num_cls": self.config.row_num_cls,
            "row_rope_base": self.config.row_rope_base,
            "icl_num_blocks": self.config.icl_num_blocks,
            "icl_nhead": self.config.icl_nhead,
            "icl_elliptical": self.config.icl_elliptical,
            "elliptical_delta": getattr(self.config, "elliptical_delta", 1.0),
            "elliptical_scale_mode": getattr(self.config, "elliptical_scale_mode", "max"),
            "ff_factor": self.config.ff_factor,
            "dropout": self.config.dropout,
            "activation": self.config.activation,
            "norm_first": self.config.norm_first,
            "icl_head": getattr(self.config, "icl_head", "tabicl"),
            "pdlc_config": {
                "topk": getattr(self.config, "pdlc_topk", None),
                "agg": getattr(self.config, "pdlc_agg", "posterior_avg"),
                "embed_norm": getattr(self.config, "pdlc_embed_norm", "none"),
            },
        }

        model = TabICL(**self.model_config)
        model.to(device=self.config.device)

        if self.master_process:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Model has {num_params} parameters.")

        # Freeze model components if requested
        if self.config.freeze_col:
            model.col_embedder.eval()
            for param in model.col_embedder.parameters():
                param.requires_grad = False

        if self.config.freeze_row:
            model.row_interactor.eval()
            for param in model.row_interactor.parameters():
                param.requires_grad = False

        if self.config.freeze_icl:
            model.icl_predictor.eval()
            for param in model.icl_predictor.parameters():
                param.requires_grad = False

        # Optional: train only the TabPDL head (when enabled)
        if getattr(self.config, "train_pdlc_only", False) and getattr(self.config, "icl_head", "tabicl") == "tabpdl":
            # Freeze all modules first
            model.col_embedder.eval()
            for p in model.col_embedder.parameters():
                p.requires_grad = False
            model.row_interactor.eval()
            for p in model.row_interactor.parameters():
                p.requires_grad = False
            model.icl_predictor.eval()
            for p in model.icl_predictor.parameters():
                p.requires_grad = False

            # Then unfreeze only the TabPDL head parameters
            pdlc_head = getattr(model.icl_predictor, "pdlc_head", None)
            if pdlc_head is not None:
                pdlc_head.train()
                for p in pdlc_head.parameters():
                    p.requires_grad = True

        # Compile model if requested
        if self.config.model_compile:
            model = torch.compile(model, dynamic=True)
            if self.master_process:
                print("Model compiled successfully.")

        # Wrap model into DDP container if using distributed training
        if self.ddp:
            # Enable find_unused_parameters to support optional components
            # such as the TabPDL head, which may not contribute gradients
            # on every rank / step (e.g., due to data-dependent control flow).
            self.model = DDP(
                model,
                device_ids=[self.ddp_local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True,
            )
            self.raw_model = self.model.module
        else:
            self.model = model
            self.raw_model = model

    def configure_prior(self):
        """
        Sets up a tabular dataset generator that creates synthetic datasets
        during training with controllable properties and data distributions.
        """

        if self.config.prior_dir is None:
            # Generate prior data on the fly
            dataset = PriorDataset(
                batch_size=self.config.batch_size,
                batch_size_per_gp=self.config.batch_size_per_gp,
                min_features=self.config.min_features,
                max_features=self.config.max_features,
                max_classes=self.config.max_classes,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                log_seq_len=self.config.log_seq_len,
                seq_len_per_gp=self.config.seq_len_per_gp,
                min_train_size=self.config.min_train_size,
                max_train_size=self.config.max_train_size,
                replay_small=self.config.replay_small,
                prior_type=self.config.prior_type,
                device=self.config.prior_device,
                n_jobs=1,  # Set to 1 to avoid nested parallelism during DDP
            )
        else:
            # Load pre-generated prior data from disk
            dataset = LoadPriorDataset(
                data_dir=self.config.prior_dir,
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                start_from=self.config.load_prior_start,
                delete_after_load=self.config.delete_after_load,
                device=self.config.prior_device,
            )

        if self.master_process:
            print(dataset)

        # Create dataloader for efficient loading and prefetching
        self.dataloader = DataLoader(
            dataset,
            batch_size=None,  # No additional batching since PriorDataset handles batching internally
            shuffle=False,
            num_workers=1,
            prefetch_factor=4,
            pin_memory=True if self.config.prior_device == "cpu" else False,
            pin_memory_device=self.config.device if self.config.prior_device == "cpu" else "",
        )

    def configure_optimizer(self):
        """Configure optimizer and scheduler (single parameter group)."""
        params = [p for p in self.raw_model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(
            params,
            lr=float(self.config.lr),
            weight_decay=float(self.config.weight_decay),
        )
        self.scheduler = get_scheduler(config=self.config, optimizer=self.optimizer)

    def _set_icl_qk_requires_grad(self, requires_grad: bool):
        """Freeze/unfreeze ICL Q/K (in-proj) to encourage metric learning early on."""
        try:
            blocks = self.raw_model.icl_predictor.tf_icl.blocks
        except Exception:
            return
        for blk in blocks:
            attn = getattr(blk, "attn", None)
            if attn is None:
                continue
            for attr in ("in_proj_weight", "in_proj_bias"):
                par = getattr(attn, attr, None)
                if par is not None and hasattr(par, "requires_grad"):
                    par.requires_grad = requires_grad

    def _update_qk_warmup_freeze(self, step: int):
        """Apply warmup freeze for Q/K projections during initial steps."""
        warmup = int(getattr(self.config, "freeze_qk_warmup_steps", 0) or 0)
        if warmup <= 0:
            return
        if step < warmup:
            if not getattr(self, "_qk_frozen", False):
                self._set_icl_qk_requires_grad(False)
                self._qk_frozen = True
        else:
            if getattr(self, "_qk_frozen", False):
                self._set_icl_qk_requires_grad(True)
                self._qk_frozen = False

    def configure_amp(self):
        """Configure automatic mixed precision (AMP) for training."""

        self.amp = self.config.amp and "cuda" in self.config.device
        self.scaler = torch.GradScaler("cuda", enabled=self.amp)
        if self.amp:
            if self.master_process:
                print("Automatic Mixed Precision is enabled.")
            # Respect requested dtype for autocast
            dtype_str = str(getattr(self.config, "dtype", "bfloat16")).lower()
            if dtype_str in ("bf16", "bfloat16"):
                amp_dtype = torch.bfloat16
            elif dtype_str in ("fp16", "float16"):
                amp_dtype = torch.float16
            else:
                amp_dtype = torch.float32
            self.amp_ctx = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            self.amp_ctx = nullcontext()

    # ------------------------------
    # Probe utilities
    # ------------------------------

    @torch.no_grad()
    def materialize_probe_batch(self) -> None:
        """Draw and freeze a small, deterministic probe batch (CPU tensors).

        Supports both on-the-fly generation (PriorDataset) and disk-based loading
        (LoadPriorDataset) when --prior_dir is set. For the latter, a separate
        one-shot loader reads the shard at --probe_from_prior_idx and keeps the
        first --probe_batch_size datasets.
        """
        # Save and set seeds deterministically (only matters for on-the-fly)
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        np.random.seed(int(self.config.probe_seed))
        torch.manual_seed(int(self.config.probe_seed))

        try:
            ds = self.dataloader.dataset
            from tabicl.prior.genload import LoadPriorDataset as _LPD
            if isinstance(ds, _LPD) or getattr(self.config, "prior_dir", None):
                # Disk-based: load a single shard independently to avoid consuming training iterator
                prior_dir = self.config.prior_dir
                if not prior_dir:
                    raise RuntimeError("prior_dir not set but dataset is LoadPriorDataset")
                start_from = int(getattr(self.config, "probe_from_prior_idx", 0) or 0)
                probe_ds = _LPD(
                    data_dir=prior_dir,
                    batch_size=max(1, int(self.config.probe_batch_size)),
                    ddp_world_size=1,
                    ddp_rank=0,
                    start_from=start_from,
                    max_batches=1,
                    delete_after_load=False,
                    device="cpu",
                )
                probe_loader = DataLoader(probe_ds, batch_size=None, shuffle=False, num_workers=0)
                X, y, d, seq_len, train_size = next(iter(probe_loader))
                # Subset to the requested probe batch size if the shard is larger
                B = int(self.config.probe_batch_size)
                if hasattr(X, "is_nested") and X.is_nested:
                    # Nested tensors: slice first B items via wrapper
                    X = X[:B]
                    y = y[:B]
                else:
                    X = X[:B]
                    y = y[:B]
                    d = d[:B]
                    seq_len = seq_len[:B]
                    train_size = train_size[:B]
            else:
                # On-the-fly PriorDataset path
                X, y, d, seq_len, train_size = ds.get_batch(batch_size=int(self.config.probe_batch_size))

            # Convert nested to padded tensors if needed
            tensors = [X, y]
            tensors = [t.to_padded_tensor(padding=0.0) if hasattr(t, "is_nested") and t.is_nested else t for t in tensors]
            X, y = tensors
            self._probe_data = {
                "X": X.cpu(),
                "y": y.cpu(),
                "d": d.cpu() if isinstance(d, torch.Tensor) else torch.as_tensor(d).cpu(),
                "seq_len": seq_len.cpu() if isinstance(seq_len, torch.Tensor) else torch.as_tensor(seq_len).cpu(),
                "train_size": train_size.cpu() if isinstance(train_size, torch.Tensor) else torch.as_tensor(train_size).cpu(),
                "seed": int(self.config.probe_seed),
            }
            if self.master_process:
                try:
                    ts = int(self._probe_data["train_size"][0].item())
                except Exception:
                    ts = int(self._probe_data["train_size"][0]) if hasattr(self._probe_data["train_size"], "__getitem__") else -1
                print(
                    f"[probe] Materialized fixed batch: B={X.shape[0]}, T={X.shape[1]}, H={X.shape[2]}, train_size={ts}"
                )
        finally:
            np.random.set_state(np_state)
            torch.set_rng_state(torch_state)

    # Deprecated: learned-parameter elliptical diagnostics removed (parameter-free estimator now)

    @torch.no_grad()
    def _compute_icl_attn_metrics(self, R: torch.Tensor, y_train: torch.Tensor, train_size: int) -> dict:
        """Compute KL(A_ellip || A_dot) and attention entropy on ICL encoder blocks."""
        metrics = {}
        try:
            blocks = self.raw_model.icl_predictor.tf_icl.blocks
        except Exception:
            return metrics

        B, T, E = R.shape
        cut = int(train_size)

        def attn_kl_entropy_for_block(blk: nn.Module, x: torch.Tensor) -> tuple[float, float]:
            q_in = blk.norm1(x)
            k_in = blk.norm1(x)
            v_in = blk.norm1(x)
            nh = blk.attn.num_heads
            head_dim = E // nh
            q, k, v = F._in_projection_packed(q_in, k_in, v_in, blk.attn.in_proj_weight, blk.attn.in_proj_bias)
            q = q.view(B, T, nh, head_dim).transpose(1, 2)
            k = k.view(B, T, nh, head_dim).transpose(1, 2)

            # Parameter-free elliptical estimator is not reconstructed here; use dot-product as baseline
            ellip = None

            q_left, k_left = q[..., :cut, :], k[..., :cut, :]
            q_right = q[..., cut:, :]
            if ellip is not None:
                ql_e, kl_e = q_left * ellip, k_left * ellip
                qr_e = q_right * ellip
            else:
                ql_e, kl_e, qr_e = q_left, k_left, q_right

            ql_d, kl_d, qr_d = q_left, k_left, q_right
            scale = float(head_dim) ** -0.5

            scores_e_left = torch.matmul(ql_e, kl_e.transpose(-1, -2)) * scale
            scores_d_left = torch.matmul(ql_d, kl_d.transpose(-1, -2)) * scale
            probs_e_left = scores_e_left.softmax(dim=-1)
            probs_d_left = scores_d_left.softmax(dim=-1)

            scores_e_right = torch.matmul(qr_e, kl_e.transpose(-1, -2)) * scale
            scores_d_right = torch.matmul(qr_d, kl_d.transpose(-1, -2)) * scale
            probs_e_right = scores_e_right.softmax(dim=-1)
            probs_d_right = scores_d_right.softmax(dim=-1)

            eps = 1e-12
            log_e_left = (probs_e_left + eps).log()
            log_e_right = (probs_e_right + eps).log()
            kl_left = F.kl_div(log_e_left, probs_d_left, reduction="batchmean")
            kl_right = F.kl_div(log_e_right, probs_d_right, reduction="batchmean")

            ent_left = -(probs_e_left * log_e_left).sum(dim=-1).mean()
            ent_right = -(probs_e_right * log_e_right).sum(dim=-1).mean()

            kl = 0.5 * (kl_left + kl_right)
            ent = 0.5 * (ent_left + ent_right)
            return float(kl.item()), float(ent.item())

        kls, ents = [], []
        x = R
        for blk in blocks:
            kl, ent = attn_kl_entropy_for_block(blk, x)
            kls.append(kl)
            ents.append(ent)
            x = blk(x, attn_mask=train_size)

        if kls:
            metrics["probe/attn_kl_mean"] = float(np.mean(kls))
        if ents:
            metrics["probe/attn_entropy_mean"] = float(np.mean(ents))
        return metrics

    @torch.no_grad()
    def _compute_probe_delta_ce(self, logits: torch.Tensor, y: torch.Tensor, train_size: int) -> dict:
        y_test = y[:, train_size:]
        pred = logits[:, train_size:]
        ce = F.cross_entropy(pred.flatten(end_dim=-2), y_test.long().flatten()).item()
        B = y.shape[0]
        entropies, weights = [], []
        for i in range(B):
            yt = y_test[i]
            if yt.numel() == 0:
                continue
            vals, counts = yt.unique(return_counts=True)
            p = counts.float() / counts.sum()
            h = float(-(p * (p + 1e-12).log()).sum().item())
            entropies.append(h)
            weights.append(float(yt.numel()))
        if weights:
            w = np.array(weights, dtype=float); w = w / w.sum()
            h_avg = float((w * np.array(entropies, dtype=float)).sum())
            delta_ce = ce - h_avg
            return {"probe/delta_ce": float(delta_ce), "probe/ce": float(ce), "probe/chance_entropy": float(h_avg)}
        else:
            return {"probe/delta_ce": float("nan"), "probe/ce": float(ce), "probe/chance_entropy": float("nan")}

    @torch.no_grad()
    def run_probe_once(self) -> dict:
        if not self._probe_data:
            return {}

        prev_mode = self.model.training
        self.model.eval()
        try:
            Xc = self._probe_data["X"].to(self.config.device)
            yc = self._probe_data["y"].to(self.config.device)
            dc = self._probe_data["d"].to(self.config.device)
            train_size = int(self._probe_data["train_size"][0].item())

            y_train = yc[:, :train_size]
            # Build embeddings and row representations using training paths
            emb = self.raw_model.col_embedder._train_forward(Xc, d=dc, train_size=train_size)
            R = self.raw_model.row_interactor._train_forward(emb, d=dc)

            # Compute full logits (train+test) via internal training path to avoid inference constraints
            logits_full = self.raw_model.icl_predictor._icl_predictions(R.clone(), y_train)
            metrics = self._compute_probe_delta_ce(logits_full, yc, train_size)

            # Attention diagnostics on the ICL encoder (condition on training labels)
            R[:, :train_size] = R[:, :train_size] + self.raw_model.icl_predictor.y_encoder(y_train.float())
            metrics.update(self._compute_icl_attn_metrics(R, y_train, train_size))

            return metrics
        finally:
            if prev_mode:
                self.model.train()

    def maybe_run_probe(self, step: int) -> None:
        if not self.master_process:
            return
        N = int(getattr(self.config, "probe_every", 0) or 0)
        if N <= 0:
            return
        if (step + 1) % N != 0:
            return
        try:
            metrics = self.run_probe_once()
            if metrics and self.wandb_run is not None:
                wandb.log(metrics, step=self.curr_step)
            if metrics and getattr(self, "metrics_csv_path", None):
                self._csv_log_row({"step": self.curr_step, **metrics})
        except Exception as e:
            print(f"[probe] Warning: probe failed at step {self.curr_step}: {e}")

    def get_latest_checkpoint(self):
        """Returns the latest checkpoint from `checkpoint_dir`

        Only considers files with the .ckpt extension (PyTorch checkpoint files).
        """
        ckpt_dir = self.config.checkpoint_dir

        if not os.path.isdir(ckpt_dir):
            return None

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]

        if not checkpoints:
            return None

        # Sort the checkpoint files by step number and get the latest
        try:
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0]))[-1]
            checkpoint_path = os.path.join(ckpt_dir, latest_checkpoint)
            return checkpoint_path
        except Exception as e:
            print(f"Error parsing checkpoint filenames: {e}")
            return None

    def load_checkpoint(self):
        """Load model and training state from checkpoint.

        First checks if `checkpoint_path` is directly specified. If not, attempts to find
        the latest checkpoint in the checkpoint directory.
        """

        checkpoint_path = None
        if hasattr(self.config, "checkpoint_path") and self.config.checkpoint_path:
            checkpoint_path = self.config.checkpoint_path
        elif hasattr(self.config, "checkpoint_dir") and self.config.checkpoint_dir:
            checkpoint_path = self.get_latest_checkpoint()

        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            print("No checkpoint found, starting from scratch.")
            return

        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.config.device, weights_only=True)

        # Load model state
        if "state_dict" not in checkpoint:
            raise ValueError("Checkpoint does not contain model state")

        # Allow missing keys to support optional features (e.g., elliptical ICL params in later stages)
        load_result = self.raw_model.load_state_dict(checkpoint["state_dict"], strict=False)
        try:
            missing = getattr(load_result, "missing_keys", [])
            unexpected = getattr(load_result, "unexpected_keys", [])
        except Exception:
            missing, unexpected = [], []
        if self.master_process and (missing or unexpected):
            print(
                f"Loaded with missing_keys={len(missing)}, unexpected_keys={len(unexpected)}.\n"
                f"Missing: {missing[:5]}{'...' if len(missing)>5 else ''} | "
                f"Unexpected: {unexpected[:5]}{'...' if len(unexpected)>5 else ''}"
            )

        # Optionally load optimizer and scheduler state
        if self.config.only_load_model:
            print("Only loading model weights")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.curr_step = checkpoint["curr_step"]
            print(f"Resuming training at step {self.curr_step}")

    def save_checkpoint(self, name: str):
        """Save model and training state to checkpoint file.

        Parameters
        ----------
        name : str
            Filename for the checkpoint
        """

        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, name)
        checkpoint = {
            "config": self.model_config,
            "state_dict": self.raw_model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "curr_step": self.curr_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def manage_checkpoint(self):
        """
        Manages the number of temporary checkpoints by deleting the oldest ones
        if the count exceeds `max_checkpoints`. Permanent checkpoints are ignored.
        """
        ckpt_dir = self.config.checkpoint_dir
        limit = self.config.max_checkpoints

        # Filter for files with "ckpt" extension matching the pattern "step-*.ckpt"
        checkpoints = [f for f in os.listdir(ckpt_dir) if f.startswith("step-") and f.endswith(".ckpt")]
        temp_checkpoints = []
        for ckpt in checkpoints:
            try:
                step = int(ckpt.split("-")[1].split(".")[0])
                # Consider a checkpoint temporary if its step is not divisible by save_perm_every
                if step % self.config.save_perm_every != 0:
                    temp_checkpoints.append((step, ckpt))
            except:
                continue  # Ignore files that don't match the format

        # Sort temporary checkpoints by step number (ascending)
        temp_checkpoints.sort(key=lambda x: x[0])

        # Remove oldest temporary checkpoints if limit is exceeded
        num_to_delete = len(temp_checkpoints) - limit
        if num_to_delete > 0:
            for step, ckpt_name in temp_checkpoints[:num_to_delete]:
                ckpt_path = os.path.join(ckpt_dir, ckpt_name)
                try:
                    os.remove(ckpt_path)
                except Exception as e:
                    print(f"Error removing checkpoint {ckpt_path}: {e}")

    @ddp_cleanup
    def train(self):
        """Main training loop.

        Iterates through batches, processes them, updates model parameters,
        and handles checkpoint saving and metric logging.
        """

        if self.master_process:
            step_progress = tqdm(range(self.curr_step, self.config.max_steps), desc="Step", leave=True)
        else:
            step_progress = range(self.curr_step, self.config.max_steps)

        dataloader = iter(self.dataloader)
        for step in step_progress:
            # Optionally freeze/unfreeze Q/K during warmup
            self._update_qk_warmup_freeze(step)
            # Get the next batch
            with Timer() as prior_timer:
                batch = next(dataloader)
            prior_time = prior_timer.elapsed

            # Train the model on the batch
            with Timer() as train_timer:
                results = self.run_batch(batch)
            train_time = train_timer.elapsed

            # Clear CUDA cache to free memory
            torch.cuda.empty_cache()

            self.curr_step = step + 1
            if self.master_process:
                # Add timing information to results
                results.update({"prior_time": prior_time, "train_time": train_time})

                # Update progress bar with rounded values for cleaner display
                step_progress.set_postfix(**{k: round(v, 3) if isinstance(v, float) else v for k, v in results.items()})

                # Save checkpoints
                is_temp_save = self.curr_step % self.config.save_temp_every == 0
                is_perm_save = self.curr_step % self.config.save_perm_every == 0

                if is_temp_save or is_perm_save:
                    ckpt_name = f"step-{self.curr_step}.ckpt"
                    self.save_checkpoint(name=ckpt_name)

                    # Manage checkpoint limit only for temporary checkpoints
                    if is_temp_save and not is_perm_save and self.config.max_checkpoints > 0:
                        self.manage_checkpoint()

            # Logging to Weights & Biases
            if self.wandb_run is not None:
                # Add learning rate to results
                results["lr"] = self.scheduler.get_last_lr()[0]
                wandb.log(results, step=self.curr_step)

            # Periodic diagnostics probe
            self.maybe_run_probe(step)

            # Local CSV logging of training metrics
            if getattr(self, "metrics_csv_path", None):
                row = {"step": self.curr_step, **results}
                self._csv_log_row(row)

    def validate_micro_batch(self, micro_seq_len, micro_train_size):
        """
        Determine a common (seq_len, train_size) for a micro batch.

        If inputs are inconsistent (due to --seq_len_per_gp True), pick the minimum
        seq_len and train_size across the micro batch to enable robust trimming.

        Parameters
        ----------
        micro_seq_len : Tensor (micro_batch_size,)
            Sequence lengths for each dataset.

        micro_train_size : Tensor (micro_batch_size,)
            Training sizes (split positions) for each dataset.

        Returns
        -------
        tuple (int, int)
            The (seq_len, train_size) to use for trimming within the micro batch.
        """
        # Robust choice: pick the minimal values across the micro batch
        # This guarantees valid slicing for all items and avoids shape mismatches.
        seq_len = int(micro_seq_len.min().item())
        train_size = int(micro_train_size.min().item())
        # Guard against degenerate splits after per-micro-batch alignment:
        # ensure at least one train token and one test token remain.
        if seq_len <= 1:
            # Fallback: force a minimal workable length
            seq_len = 2
        # Clamp train_size into [1, seq_len - 1]
        if train_size < 1:
            train_size = 1
        elif train_size >= seq_len:
            train_size = seq_len - 1
        return seq_len, train_size

    def align_micro_batch(self, micro_X, micro_y, micro_d, seq_len):
        """
        Truncate micro batch tensors to required dimensions.

        Truncates sequence length and feature dimensions to the validated `seq_len`
        and the maximum active features (`micro_d.max()`) respectively. This optimizes
        memory and computation by removing unused tensor elements.

        Parameters
        ----------
        micro_X : Tensor (B, T, H)
            Input features per dataset.

        micro_y : Tensor (B, T)
            Target labels per dataset.

        micro_d : Tensor (B,)
            Number of active features per dataset.

        seq_len : int
            Validated sequence length for this micro batch.

        Returns
        -------
        tuple (Tensor, Tensor)
            Truncated (micro_X, micro_y) tensors with shapes
            (B, seq_len, micro_d.max()) and (B, seq_len).
        """
        # Truncate sequence length
        if micro_X.shape[1] > seq_len:
            micro_X = micro_X[:, :seq_len]

        if micro_y.shape[1] > seq_len:
            micro_y = micro_y[:, :seq_len]

        # Truncate feature dimension
        max_features = micro_d.max().item()
        if micro_X.shape[-1] > max_features:
            micro_X = micro_X[..., :max_features]

        return micro_X, micro_y

    def run_micro_batch(self, micro_batch, micro_batch_idx, num_micro_batches):
        """Process a micro batch for gradient accumulation.

        Parameters
        ----------
        micro_batch : tuple
            (micro_X, micro_y, micro_d, micro_seq_len, micro_train_size) tensors for the micro batch

        micro_batch_idx : int
            Index of the current micro batch

        num_micro_batches : int
            Total number of micro batches

        Returns
        -------
        dict
            Result dictionary
        """
        micro_X, micro_y, micro_d, micro_seq_len, micro_train_size = micro_batch
        seq_len, train_size = self.validate_micro_batch(micro_seq_len, micro_train_size)
        micro_X, micro_y = self.align_micro_batch(micro_X, micro_y, micro_d, seq_len)

        # Move to device
        micro_X = micro_X.to(self.config.device)
        micro_y = micro_y.to(self.config.device)
        micro_d = micro_d.to(self.config.device)

        y_train = micro_y[:, :train_size]
        y_test = micro_y[:, train_size:]

        # Set DDP gradient sync for last micro batch only
        if self.ddp:
            self.model.require_backward_grad_sync = micro_batch_idx == num_micro_batches - 1

        # Choose loss based on ICL head type
        icl_head = getattr(self.raw_model, "icl_head", "tabicl")

        if icl_head == "tabpdl":
            # Pairwise PDLC training: optimize a weighted BCE over query-support pairs.
            with self.amp_ctx:
                # Forward pass through full model to build representations and PDLC aux
                pred = self.model(micro_X, y_train, micro_d)  # aggregated logits for metrics

                # Retrieve pairwise logits S from PDLC head (shape: B, M, N)
                enc = self.raw_model.icl_predictor
                aux = getattr(enc, "last_pdlc_aux", None)
                if aux is None or "pair_logits" not in aux:
                    raise RuntimeError("PDLC head did not populate pair_logits in last_pdlc_aux.")
                pair_logits = aux["pair_logits"]  # (B, M, N)

                B, M, N = pair_logits.shape
                # Build binary targets T_{ij} = 1 iff y_q == y_s
                y_s = y_train  # (B, N)
                y_q = y_test   # (B, M)
                # Broadcast equality across query and support positions
                T = (y_q.unsqueeze(-1) == y_s.unsqueeze(1)).float()  # (B, M, N)

                # Optional support mask (currently all valid in training setup)
                support_mask = aux.get("support_mask", torch.ones_like(y_s, dtype=torch.bool))
                M_mask = support_mask.unsqueeze(1).expand_as(pair_logits)  # (B, M, N)

                # Select valid pairs
                S_flat = pair_logits[M_mask]
                T_flat = T[M_mask]

                # Dynamic positive class weight
                eps = 1e-6
                n_pos = T_flat.sum()
                n_total = T_flat.numel()
                n_neg = n_total - n_pos
                pos_weight = n_neg / (n_pos + eps)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                loss = criterion(S_flat, T_flat)

            # Scale loss for gradient accumulation and backpropagate
            scaled_loss = loss / num_micro_batches
            self.scaler.scale(scaled_loss).backward()

            # Metrics (classification CE/accuracy on aggregated logits, no grad)
            with torch.no_grad():
                micro_results = {}
                pred_flat = pred.flatten(end_dim=-2)
                true_flat = y_test.long().flatten()
                ce_val = F.cross_entropy(pred_flat, true_flat)
                micro_results["ce"] = (ce_val.item() / num_micro_batches)
                accuracy = (pred_flat.argmax(dim=1) == true_flat).sum() / len(true_flat)
                micro_results["accuracy"] = accuracy.item() / num_micro_batches

            return micro_results

        # Default: standard TabICL training with cross-entropy on query logits
        with self.amp_ctx:
            pred = self.model(micro_X, y_train, micro_d)  # (B, test_size, max_classes)
            pred = pred.flatten(end_dim=-2)
            true = y_test.long().flatten()
            loss = F.cross_entropy(pred, true)
            # No learned-parameter elliptical regularization (parameter-free estimator)

        # Scale loss for gradient accumulation and backpropagate
        scaled_loss = loss / num_micro_batches
        self.scaler.scale(scaled_loss).backward()

        with torch.no_grad():
            micro_results = {}
            micro_results["ce"] = scaled_loss.item()
            accuracy = (pred.argmax(dim=1) == true).sum() / len(true)
            micro_results["accuracy"] = accuracy.item() / num_micro_batches
            # No separate regularization metric

        return micro_results

    def run_batch(self, batch):
        """
        Trains the model on a batch of datasets. Handles gradient accumulation by
        splitting the batch into micro-batches. Supports variable-sized datasets
        by padding. Skips micro-batches on CUDA OOM errors. Updates model
        parameters and returns loss and accuracy metrics.

        Parameters
        ----------
        batch: tuple
            Contains tensors (X, y, d, seq_len, train_size) for the batch.
            X and y can be Tensors or NestedTensors (for variable sequence lengths).

        Returns
        ------
        dict
            Dictionary containing 'ce' (cross-entropy loss) and 'accuracy'.

        Raises
        ------
        RuntimeError
            If more than 10% of micro-batches fail due to OOM errors.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        # Pad nested tensors to the same size
        batch = [t.to_padded_tensor(padding=0.0) if t.is_nested else t for t in batch]

        # Split the batch into micro-batches along the first dimension
        num_micro_batches = math.ceil(self.config.batch_size / self.config.micro_batch_size)
        micro_batches = [torch.split(t, self.config.micro_batch_size, dim=0) for t in batch]
        micro_batches = list(zip(*micro_batches))

        results = {"ce": 0.0, "accuracy": 0.0}
        failed_batches = 0

        for idx, micro_batch in enumerate(micro_batches):
            try:
                micro_results = self.run_micro_batch(micro_batch, idx, num_micro_batches)
                for k, v in micro_results.items():
                    results[k] += v
            except torch.cuda.OutOfMemoryError:
                print(
                    f"Warning: OOM error in micro-batch {idx+1}/{num_micro_batches} at step {self.curr_step}. Skipping."
                )
                torch.cuda.empty_cache()
                failed_batches += 1
                continue

        failure_ratio = failed_batches / num_micro_batches
        if failure_ratio > 0.1:
            raise RuntimeError(
                f"({failure_ratio:.1%}) of micro-batches failed due to OOM at step {self.curr_step}. "
                f"Please check configuration to reduce memory consumption."
            )

        # Clip the gradient
        if self.config.gradient_clipping > 0:
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clipping)

        # Update parameters
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update the learning rate
        self.optimizer.zero_grad(set_to_none=True)
        self.scheduler.step()

        return results

    # ------------------------------
    # CSV metrics logging
    # ------------------------------
    def _csv_log_row(self, row: dict) -> None:
        if not self.master_process or not getattr(self, "metrics_csv_path", None):
            return
        path = self.metrics_csv_path
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        except Exception:
            pass
        file_exists = os.path.exists(path)
        # Normalize fields
        out = {k: row.get(k, None) for k in getattr(self, "csv_fields", [])}
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.csv_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(out)


if __name__ == "__main__":
    parser = build_parser()
    config = parser.parse_args()

    try:
        # Set the start method for subprocesses to 'spawn'
        set_start_method("spawn")
    except RuntimeError:
        pass  # Ignore the error if the context has already been set

    # Create trainer and start training
    trainer = Trainer(config)
    trainer.train()
