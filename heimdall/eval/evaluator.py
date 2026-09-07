import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

def compute_metrics(pred: np.ndarray, target: np.ndarray, min_depth=1e-3, max_depth=80.0) -> Dict[str, float]:
    """
    Computes standard depth estimation metrics: RMSE, AbsRel, log10, and delta inliers.
    Ignores non-positive values or values outside min/max depth.
    """
    valid_mask = (target > min_depth) & (target < max_depth) & (pred > min_depth) & (pred < max_depth)
    
    if not valid_mask.any():
        logger.warning("No valid pixels found for evaluation.")
        return {
            "rmse": 0.0,
            "abs_rel": 0.0,
            "log10": 0.0,
            "delta1": 0.0,
            "delta2": 0.0,
            "delta3": 0.0
        }
        
    p = pred[valid_mask]
    t = target[valid_mask]
    
    # AbsRel
    abs_rel = np.mean(np.abs(p - t) / t)
    
    # RMSE
    rmse = np.sqrt(np.mean((p - t) ** 2))
    
    # Log10 error
    log10 = np.mean(np.abs(np.log10(p) - np.log10(t)))
    
    # Threshold metrics (Delta inliers)
    thresh = np.maximum((p / t), (t / p))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    
    return {
        "rmse": float(rmse),
        "abs_rel": float(abs_rel),
        "log10": float(log10),
        "delta1": float(d1),
        "delta2": float(d2),
        "delta3": float(d3)
    }

class Evaluator:
    def __init__(self):
        self.results = {}
        
    def add_result(self, category: str, pred: np.ndarray, target: np.ndarray):
        """Adds a single image pair to the category evaluation."""
        if category not in self.results:
            self.results[category] = {"preds": [], "targets": []}
            
        self.results[category]["preds"].append(pred)
        self.results[category]["targets"].append(target)
        
    def evaluate_all(self) -> Dict[str, Dict[str, float]]:
        """Computes metrics for each category and an overall aggregate."""
        metrics_by_category = {}
        
        all_preds = []
        all_targets = []
        
        for category, data in self.results.items():
            cat_preds = np.concatenate([p.flatten() for p in data["preds"]])
            cat_targets = np.concatenate([t.flatten() for t in data["targets"]])
            
            metrics_by_category[category] = compute_metrics(cat_preds, cat_targets)
            
            all_preds.append(cat_preds)
            all_targets.append(cat_targets)
            
        if all_preds:
            total_preds = np.concatenate(all_preds)
            total_targets = np.concatenate(all_targets)
            metrics_by_category["overall"] = compute_metrics(total_preds, total_targets)
            
        return metrics_by_category
