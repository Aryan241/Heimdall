#!/usr/bin/env python3
"""
Stage 3 — Train the ASPP decoder head on a frozen Depth Anything backbone.

Supports GAMUS (HDF5), DFC2019 and ISPRS layouts. The best checkpoint is selected by
*validation RMSE* (native-GSD centre crops), not by training loss.

Usage:
  # from scratch
  python scripts/train_decoder.py --dataset-dir data/GAMUS --epochs 40 --batch-size 4
  # continue / fine-tune an existing head (e.g. the epoch-78 checkpoint) at a lower LR
  python scripts/train_decoder.py --dataset-dir data/GAMUS --resume checkpoints/decoder/decoder_best_all.pth \
         --epochs 10 --lr 2e-5
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from tqdm import tqdm


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the Heimdall decoder head")
    ap.add_argument("--dataset-dir", "--dataset_dir", "--data-dir", "--data_dir", dest="dataset_dir", type=Path,
                    nargs="+", required=True, help="One or more dataset roots (e.g. GAMUS and AHN together).")
    ap.add_argument("--strata", type=str, default=None, help="Optional tile-id filter (e.g. a city code).")
    ap.add_argument("--tag", default=None, help="Checkpoint name suffix (default: strata or 'all').")
    ap.add_argument("--gpu", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=4)
    ap.add_argument("--accumulate", type=int, default=4, help="Gradient accumulation steps.")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--model", type=str, default="vit-b",
                    choices=["vit-s", "vit-b", "vit-l", "da3-metric-l", "da3-mono-l"])
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="Where cache_backbone.py wrote its caches (for read-only dataset mounts).")
    ap.add_argument("--head", default="multiscale", choices=["affine", "multiscale"],
                    help="multiscale: decode the DPT feature pyramid (stronger, backbone runs each step). "
                         "affine: RGB + scalar depth only (weaker, can use --backbone-cache).")
    ap.add_argument("--backbone-cache", action="store_true",
                    help="Use pre-computed backbone outputs from scripts/cache_backbone.py (much faster).")
    ap.add_argument("--output-dir", "--output_dir", dest="output_dir", type=Path, default=Path("checkpoints/decoder"))
    ap.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=512)
    ap.add_argument("--num-workers", "--num_workers", dest="num_workers", type=int, default=2)
    ap.add_argument("--resume", type=Path, default=None, help="Head checkpoint to start from.")
    ap.add_argument("--val-split", default="val", help="Validation split folder (val / test).")
    ap.add_argument("--height-weight", type=float, default=0.0,
                    help="Ablation E1: weight of a plain-metres loss term scaled by true height. "
                         "0 (default) reproduces the shipped model.")
    ap.add_argument("--height-scale", type=float, default=10.0,
                    help="Metres at which the height weighting doubles (default 10).")
    ap.add_argument("--unfreeze-blocks", type=int, default=0,
                    help="Ablation E2: fine-tune the last N backbone transformer blocks. "
                         "0 (default) keeps the backbone frozen.")
    ap.add_argument("--backbone-lr", type=float, default=None,
                    help="Learning rate for thawed backbone blocks (default: lr/10).")
    ap.add_argument("--val-max", type=int, default=300, help="Max validation tiles per epoch.")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--max-oom", type=int, default=40,
                    help="Give up after this many out-of-memory events (shared GPUs).")
    ap.add_argument("--oom-wait", type=int, default=30, help="Seconds to wait after an OOM.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    log = logging.getLogger("train")

    from heimdall.decoder.dataset import RemoteSensingHeightDataset
    from heimdall.decoder.model import DomainAdaptationWrapper
    from heimdall.device import get_device, supports_amp

    device = torch.device(f"cuda:{args.gpu}") if args.gpu is not None and torch.cuda.is_available() else get_device()
    use_amp = supports_amp(device)
    log.info("Device: %s  AMP: %s", device, use_amp)

    if args.head == "multiscale" and args.backbone_cache:
        log.warning("--backbone-cache only caches the scalar depth, which the multiscale head does not "
                    "use; ignoring it and running the frozen backbone in the loop.")
        args.backbone_cache = False
    cache = args.model if args.backbone_cache else None

    def build(split: str, augment: bool):
        parts = [RemoteSensingHeightDataset(d, split=split, strata=args.strata, patch_size=args.patch_size,
                                            augment=augment, backbone_cache=cache, cache_root=args.cache_root)
                 for d in args.dataset_dir]
        parts = [p for p in parts if len(p)]
        for d, p in zip(args.dataset_dir, parts):
            log.info("  %s/%s: %d tiles", d, split, len(p))
        return ConcatDataset(parts) if len(parts) > 1 else (parts[0] if parts else [])

    train_ds = build("train", not args.no_augment)
    if len(train_ds) == 0:
        log.error("Training set is empty — expected images/ and heights/ (with train/ val/ splits) under %s",
                  args.dataset_dir)
        return 1
    val_ds = build(args.val_split, False)
    if len(val_ds) > args.val_max:
        val_ds = Subset(val_ds, np.linspace(0, len(val_ds) - 1, args.val_max).astype(int).tolist())
    log.info("Train tiles: %d   Val tiles: %d", len(train_ds), len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) \
        if len(val_ds) else None

    # With a complete cache the frozen backbone is never needed during training.
    cache_base = lambda d: (args.cache_root / d.name) if args.cache_root else d
    have_cache = args.backbone_cache and all((cache_base(d) / "backbone" / args.model / "train").exists()
                                             for d in args.dataset_dir)
    model = DomainAdaptationWrapper(model_key=args.model, device_str=str(device),
                                    load_backbone=not have_cache, head=args.head).to(device)
    start_epoch = 1
    if args.height_weight > 0:
        from heimdall.decoder.loss import MetricDepthLoss
        model.loss_fn = MetricDepthLoss(alpha=1.0, beta=0.5, gamma=0.1,
                                        height_weight=args.height_weight,
                                        height_scale_m=args.height_scale).to(device)
        log.info("Height-weighted loss enabled: weight %.2f, doubling at %.1f m",
                 args.height_weight, args.height_scale)

    groups = [{"params": list(model.head.parameters()), "lr": args.lr}]
    thawed = model.unfreeze_last_blocks(args.unfreeze_blocks) if args.unfreeze_blocks else []
    if thawed:
        # A much smaller step for pretrained weights than for the randomly initialised head,
        # which is what keeps fine-tuning from erasing the prior it is meant to adapt.
        groups.append({"params": thawed, "lr": args.backbone_lr or args.lr / 10})
    optimizer = torch.optim.AdamW(groups, lr=args.lr, weight_decay=1e-4)
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.head.load_state_dict({k.replace("module.", ""): v for k, v in ckpt["head_state_dict"].items()})
        log.info("Resumed head from %s (epoch %s)", args.resume, ckpt.get("epoch"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    log.info("Trainable params: %.2f M (head %.2f M + backbone %.2f M)",
             sum(p.numel() for g in groups for p in g["params"]) / 1e6,
             sum(p.numel() for p in model.head.parameters()) / 1e6,
             sum(p.numel() for p in thawed) / 1e6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or args.strata or "all"
    best_rmse = float("inf")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0 = time.time()
        running = 0.0
        oom = steps = 0
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stdout)
        for bi, batch in enumerate(pbar):
            images, heights = batch["image"].to(device), batch["height"].to(device)
            depth = batch["depth"].to(device) if "depth" in batch else None
            try:
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    out = model(images, target_height=heights, precomputed_depth=depth)
                    loss = out["loss"] / args.accumulate
                (scaler.scale(loss) if scaler else loss).backward()
            except torch.OutOfMemoryError:
                # Shared GPU: another process may grab memory at any moment. Drop this batch,
                # wait for it to be released and carry on rather than losing the whole run.
                oom += 1
                optimizer.zero_grad(set_to_none=True)
                del images, heights, depth
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if oom > args.max_oom:
                    log.error("%d out-of-memory events — another process is using the GPU. "
                              "Lower --batch-size/--patch-size or wait for the GPU to free up.", oom)
                    raise
                log.warning("CUDA OOM (%d/%d) — skipping batch and retrying in %ds",
                            oom, args.max_oom, args.oom_wait)
                time.sleep(args.oom_wait)
                continue
            if (bi + 1) % args.accumulate == 0 or bi + 1 == len(train_loader):
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                if scaler:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            running += out["loss"].item()
            steps += 1
            pbar.set_postfix(loss=f"{out['loss'].item():.3f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
        scheduler.step()
        train_loss = running / max(1, steps)

        val = evaluate(model, val_loader, device) if val_loader else {}
        log.info("Epoch %d | train loss %.4f | val RMSE %.3f m  MAE %.3f m  r %.3f | %.0fs",
                 epoch, train_loss, val.get("rmse", float("nan")), val.get("mae", float("nan")),
                 val.get("pearson_r", float("nan")), time.time() - t0)

        state = {"epoch": epoch, "head_state_dict": model.head.state_dict(), "loss": train_loss,
                 "val": val, "model_key": args.model, "head": args.head}
        if thawed:
            # Fine-tuned backbone weights must travel with the head, or the checkpoint silently
            # reverts to the pretrained backbone and the experiment's result cannot be shipped.
            state["backbone_state_dict"] = {k: v.cpu() for k, v in model.backbone.state_dict().items()}
            state["unfreeze_blocks"] = args.unfreeze_blocks
        torch.save({**state, "optimizer_state_dict": optimizer.state_dict()},
                   args.output_dir / f"decoder_ep{epoch:02d}_{tag}.pth")
        score = val.get("rmse", train_loss)
        if score < best_rmse:
            best_rmse = score
            torch.save(state, args.output_dir / f"decoder_best_{tag}.pth")
            log.info("New best checkpoint (val RMSE %.3f m)", best_rmse)

    log.info("Done. Best val RMSE: %.3f m → %s", best_rmse, args.output_dir / f"decoder_best_{tag}.pth")
    return 0


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    from heimdall.eval.metrics import RunningMetrics
    model.eval()
    acc = RunningMetrics(max_pixels_per_sample=50_000)
    for batch in loader:
        pred = model(batch["image"].to(device),
                     precomputed_depth=batch["depth"].to(device) if "depth" in batch else None)["pred_height"].float().cpu().numpy()
        ref = batch["height"].numpy()
        for p, r in zip(pred, ref):
            r = r[0].copy()
            r[(r < -100) | (r > 1000)] = np.nan
            acc.add("val", np.clip(p[0], 0, None), r)
    model.train()
    return acc.report().get("val", {})


if __name__ == "__main__":
    sys.exit(main())
