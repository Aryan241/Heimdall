#!/usr/bin/env python3
"""
Stage 3 — Training CLI Script for Domain Adaptation

Runs parallel, single-GPU training jobs for the decoder head on top of the
frozen Depth Anything V2 backbone.

Usage:
  python scripts/train_decoder.py --dataset-dir data/dfc2019 --strata urban --gpu 0
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

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
    parser.add_argument("--model", type=str, default="vit-b", choices=["vit-s", "vit-b", "vit-l"])
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoints/decoder"))
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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
        logger.info("AMP is supported on this device. Using torch.cuda.amp.")

    # Setup Dataset & DataLoader
    train_ds = RemoteSensingHeightDataset(args.dataset_dir, split="train", strata=args.strata, patch_size=512)
    if len(train_ds) == 0:
        logger.error("Dataset is empty. Ensure images/ and heights/ exist in %s/train", args.dataset_dir)
        return 1

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Setup Model
    logger.info("Initializing Model (Backbone: %s) ...", args.model)
    model = DomainAdaptationWrapper(model_key=args.model).to(device)
    
    # Optimizer (only train the head)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # Training Loop
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training for %d epochs...", args.epochs)
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            images = batch["image"].to(device)
            heights = batch["height"].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images, target_height=heights)
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images, target_height=heights)
                loss = outputs["loss"]
                loss.backward()
                optimizer.step()
                
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = epoch_loss / len(train_loader)
        logger.info("Epoch %d complete. Avg Loss: %.4f", epoch, avg_loss)
        
        # Save checkpoint
        ckpt_path = args.output_dir / f"decoder_ep{epoch:02d}_{args.strata or 'all'}.pth"
        torch.save(model.head.state_dict(), ckpt_path)
        
    logger.info("Training complete. Final checkpoint saved to %s", ckpt_path)
    return 0

if __name__ == "__main__":
    sys.exit(main())
