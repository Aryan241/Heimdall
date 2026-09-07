"""
Stage 4 — Semantic Segmentation (Ground Masking)

Uses a pre-trained SegFormer model to classify pixels.
Extracts a binary 'ground mask' to feed into the RANSAC calibration (Stage 5),
ensuring that buildings and trees do not skew the height calibration.
"""

from __future__ import annotations

import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)

class GroundSegmenter:
    """
    Wrapper for HuggingFace SegFormer.
    Defaulting to ADE20K finetuned B0 for fast inference.
    """
    def __init__(self, model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512", device: torch.device | str = "cpu"):
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
        self.device = device
        logger.info("Loading SegFormer model %s to %s", model_id, self.device)
        self.processor = SegformerImageProcessor.from_pretrained(model_id)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_id).to(self.device)
        self.model.eval()
        
        # ADE20K labels that we consider "ground" or flat terrain suitable for DEM matching
        # 6: road, 9: grass, 11: sidewalk, 13: earth/ground, 29: field, 46: sand
        self.ground_labels = {6, 9, 11, 13, 29, 46}

    @torch.no_grad()
    def get_ground_mask(self, image: np.ndarray | torch.Tensor) -> torch.Tensor:
        """
        Args:
            image: (H, W, 3) numpy array or (3, H, W) tensor representing the RGB image.
        Returns:
            (H, W) boolean tensor where True = Ground.
        """
        # If input is a numpy array (H, W, 3) from ingestion, just pass to processor
        # If it's already a tensor, we assume it's scaled or we convert back to PIL/numpy for processor.
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs.pixel_values.to(self.device)
        
        outputs = self.model(pixel_values=pixel_values)
        logits = outputs.logits  # (1, num_labels, H/4, W/4)
        
        # Upsample logits to original image size
        if isinstance(image, np.ndarray):
            target_size = image.shape[:2]
        else:
            target_size = image.shape[1:]
            
        upsampled_logits = F.interpolate(
            logits, size=target_size, mode="bilinear", align_corners=False
        )
        
        # Get highest probability class
        predicted_classes = upsampled_logits.argmax(dim=1).squeeze() # (H, W)
        
        # Create boolean mask
        mask = torch.zeros_like(predicted_classes, dtype=torch.bool)
        for label_id in self.ground_labels:
            mask = mask | (predicted_classes == label_id)
            
        logger.debug("Ground mask generated. Ground pixels: %.1f%%", (mask.sum().item() / mask.numel()) * 100)
        return mask
