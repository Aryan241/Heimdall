#!/usr/bin/env python3
"""
Stage 3 — Training CLI Script for Domain Adaptation

Trains the ASPP decoder head on top of a frozen Depth Anything V2 or V3 backbone.
Supports GAMUS (HDF5), DFC2019, and ISPRS datasets.

Usage:
  python scripts/train_decoder.py --dataset-dir data/GAMUS --gpu 0 --model da3-metric-l --batch-size 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Ensure the root directory is in sys.path so 'heimdall' module can be found
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

def main() -> int:
    parser = argparse.ArgumentParser(description="Train Decoder Head for Heimdall")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Path to training dataset.")
    parser.add_argument("--strata", type=str, default=None, help="Optional strata filter (e.g. 'urban').")
    parser.add_argument("--gpu", type=int, default=None, help="GPU index to use (defaults to 0 if available).")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (keep small for 8GB VRAM).")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--model", type=str, default="da3-metric-l",
        choices=["vit-s", "vit-b", "vit-l", "da3-metric-l", "da3-mono-l"],
        help="Depth Anything backbone variant (default: da3-metric-l)."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/decoder"))
    parser.add_argument("--patch-size", type=int, default=512, help="Patch size for random crops.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger("train")

    # Defer heavy imports
    from heimdall.decoder.dataset import RemoteSensingHeightDataset
    from heimdall.decoder.model import DomainAdaptationWrapper
    from heimdall.device import get_device, supports_amp

    # Setup device
    if args.gpu is not None and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = get_device()
    logger.info("Using device: %s", device)
    
    use_amp = supports_amp(device)
    if use_amp:
        logger.info("AMP (Automatic Mixed Precision) enabled.")

    # Setup Dataset & DataLoader
    logger.info("Loading dataset from %s ...", args.dataset_dir)
    train_ds = RemoteSensingHeightDataset(
        args.dataset_dir, split="train", strata=args.strata, patch_size=args.patch_size
    )
    if len(train_ds) == 0:
        logger.error(
            "Dataset is empty! Check that %s contains images/ and heights/ subdirectories "
            "with train/val splits. For GAMUS, ensure .h5 files are present.",
            args.dataset_dir
        )
        return 1
    
    logger.info("Dataset loaded: %d training samples.", len(train_ds))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )

    # Setup Model
    logger.info("Initializing Model (Backbone: %s) ...", args.model)
    model = DomainAdaptationWrapper(model_key=args.model, device_str=str(device)).to(device)
    
    trainable_params = sum(p.numel() for p in model.head.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("Trainable params: %.2f M / Total: %.2f M", trainable_params / 1e6, total_params / 1e6)

    # Optimizer (only train the head)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    
    logger.info("=" * 60)
    logger.info("Starting training for %d epochs (%d batches/epoch)", args.epochs, len(train_loader))
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_silog = 0.0
        epoch_start = time.time()
        
        accumulate_grad_batches = 4
        optimizer.zero_grad(set_to_none=True)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", file=sys.stdout)
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)
            heights = batch["height"].to(device)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, target_height=heights)
                    loss = outputs["loss"] / accumulate_grad_batches
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                outputs = model(images, target_height=heights)
                loss = outputs["loss"] / accumulate_grad_batches
                loss.backward()
                
                if (batch_idx + 1) % accumulate_grad_batches == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.head.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                
            # Log exact unscaled values for monitoring
            l_val = outputs["loss"].item()
            silog_val = outputs["silog"].item() if "silog" in outputs and outputs["silog"] is not None else 0.0
            
            epoch_loss += l_val
            epoch_silog += silog_val
            
            pbar.set_postfix(
                loss=f"{l_val:.2f}", 
                silog=f"{silog_val:.2f}",
                lr=f"{optimizer.param_groups[0]['lr']:.2e}"
            )
        
        scheduler.step()
            
        avg_loss = epoch_loss / len(train_loader)
        avg_silog = epoch_silog / len(train_loader)
        elapsed = time.time() - epoch_start
        logger.info(
            "Epoch %d/%d complete | Avg Loss: %.4f | Avg SILog: %.4f | Time: %.1fs | LR: %.2e",
            epoch, args.epochs, avg_loss, avg_silog, elapsed, optimizer.param_groups[0]['lr']
        )
        
        # Save checkpoint
        ckpt_path = args.output_dir / f"decoder_ep{epoch:02d}_{args.strata or 'all'}.pth"
        torch.save({
            "epoch": epoch,
            "head_state_dict": model.head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "model_key": args.model,
        }, ckpt_path)
        logger.info("Checkpoint saved: %s", ckpt_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = args.output_dir / f"decoder_best_{args.strata or 'all'}.pth"
            torch.save({
                "epoch": epoch,
                "head_state_dict": model.head.state_dict(),
                "loss": avg_loss,
                "model_key": args.model,
            }, best_path)
            logger.info("New best model saved: %s (loss: %.4f)", best_path, best_loss)
        
    logger.info("=" * 60)
    logger.info("Training complete! Best loss: %.4f", best_loss)
    logger.info("Best checkpoint: %s", args.output_dir / f"decoder_best_{args.strata or 'all'}.pth")
    logger.info("=" * 60)
    return 0

if __name__ == "__main__":
    sys.exit(main())
