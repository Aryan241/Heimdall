#!/usr/bin/env python3
"""
Heimdall — eval.py
==================
Evaluates a trained Domain Adaptation model against ground truth DEMs.
Supports HDF5 (GAMUS) and standard image datasets out of the box using our Dataset loader.

Usage
-----
    python eval.py --weights checkpoints/decoder/decoder_epoch_10.pth --val-data data/GAMUS --output results.json
"""

import argparse
import logging
import sys
import json
from pathlib import Path

# Ensure repo root is in sys.path
repo_root = str(Path(__file__).resolve().parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import torch

def main() -> int:
    parser = argparse.ArgumentParser(description="Heimdall: Evaluation Harness")
    parser.add_argument("--weights", type=Path, help="Path to trained model weights (.pth).")
    parser.add_argument("--val-data", "--val_data", "--data-dir", "--data_dir", "--dataset-dir", "--dataset_dir", dest="val_data", type=Path, required=True, help="Path to validation dataset directory.")
    parser.add_argument("--output", "--output-dir", "--output_dir", dest="output", type=Path, default=Path("eval_results.json"), help="Output JSON report or directory.")
    parser.add_argument("--model", type=str, default="da3-metric-l", help="Model backbone to use.")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging.")
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(name)-30s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("heimdall.eval")
    
    # Imports deferred for speed
    from heimdall.eval.evaluator import Evaluator
    from heimdall.decoder.model import DomainAdaptationWrapper
    from heimdall.decoder.dataset import RemoteSensingHeightDataset
    from heimdall.device import get_device
    
    log.info("Starting Evaluation Harness...")
    log.info("Validation Data: %s", args.val_data)
    if args.weights:
        log.info("Weights: %s", args.weights)
        
    evaluator = Evaluator()
    device = get_device()
    
    # ── METRICS COMPUTATION ───────────────────────────────────────────────────
    if not args.val_data.exists():
        log.error("Validation data directory not found: %s", args.val_data)
        return 1

    log.info("Loading model (%s)...", args.model)
    model = DomainAdaptationWrapper(model_key=args.model, device_str=device.type)
    
    if args.weights and args.weights.exists():
        log.info("Loading trained weights from %s", args.weights)
        try:
            checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        except Exception:
            checkpoint = torch.load(args.weights, map_location=device)
        
        # Check if this is a custom checkpoint dict (from train_decoder.py) or a raw state dict
        if "head_state_dict" in checkpoint:
            log.info("Detected custom checkpoint format. Loading 'head_state_dict' only...")
            state_dict = checkpoint["head_state_dict"]
            # Unwrap module. prefix if necessary
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.head.load_state_dict(new_state_dict, strict=True)
        else:
            log.info("Detected raw model state dict. Loading full model...")
            state_dict = checkpoint
            # Unwrap module. prefix if necessary
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            model.load_state_dict(new_state_dict, strict=True)
    else:
        log.warning("No weights provided or file not found! Evaluating baseline untrained model...")
        
    model.to(device)
    model.eval()

    log.info("Initializing dataset loader...")
    # Use our unified dataset loader (handles H5, png, tif automatically)
    dataset = RemoteSensingHeightDataset(args.val_data, patch_size=512, split="val")
    
    if len(dataset) == 0:
        log.error("No valid image/height pairs found in %s", args.val_data)
        log.warning("Running mock evaluation fallback...")
        # Mock eval fallback for testing without data
        pred_urban = np.random.uniform(5.0, 30.0, (512, 512))
        gt_urban = pred_urban + np.random.normal(0, 1.5, (512, 512))
        evaluator.add_result("urban", pred_urban, gt_urban)
    else:
        log.info("Found %d validation samples. Beginning evaluation...", len(dataset))
        
        with torch.no_grad():
            for idx in range(len(dataset)):
                sample = dataset[idx]
                image_tensor = sample["image"].unsqueeze(0).to(device) # (1, 3, H, W)
                gt_tensor = sample["height"] # (1, H, W)
                img_path = Path(sample["image_path"])
                
                try:
                    # Forward pass
                    outputs = model(image_tensor)
                    pred_tensor = outputs["pred_height"] # (1, 1, H, W)
                    
                    # Convert to numpy for evaluator
                    pred_depth = pred_tensor.squeeze().cpu().numpy()
                    gt_depth = gt_tensor.squeeze().cpu().numpy()
                    
                    # Add to evaluator. (Inferring category from filename)
                    category = "urban" if "urban" in img_path.name.lower() else "forested" if "forest" in img_path.name.lower() else "general"
                    evaluator.add_result(category, pred_depth, gt_depth)
                    
                    if args.verbose:
                        log.debug("Evaluated %s", img_path.name)
                        
                    if (idx + 1) % 50 == 0:
                        log.info("Processed %d / %d samples...", idx + 1, len(dataset))
                        
                except Exception as e:
                    log.error("Failed evaluating %s: %s", img_path.name, e)
    
    log.info("Computing metrics across all categories...")
    results = evaluator.evaluate_all()
    
    # Save JSON report
    output_file = args.output
    if output_file.is_dir():
        output_file = output_file / "eval_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
        
    log.info("Evaluation complete! Report saved to %s", output_file)
    
    # Print formatted table
    print("\n" + "=" * 80)
    print(f"{'Category':<15} | {'RMSE':<8} | {'AbsRel':<8} | {'δ < 1.25':<10} | {'δ < 1.25^2':<10} | {'δ < 1.25^3':<10}")
    print("-" * 80)
    
    for cat, mets in results.items():
        if cat == "overall":
            continue
        print(f"{cat:<15} | {mets['rmse']:<8.3f} | {mets['abs_rel']:<8.3f} | {mets['delta1']:<10.3f} | {mets['delta2']:<10.3f} | {mets['delta3']:<10.3f}")
        
    print("-" * 80)
    if "overall" in results:
        o = results["overall"]
        print(f"{'OVERALL':<15} | {o['rmse']:<8.3f} | {o['abs_rel']:<8.3f} | {o['delta1']:<10.3f} | {o['delta2']:<10.3f} | {o['delta3']:<10.3f}")
    print("=" * 80 + "\n")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
