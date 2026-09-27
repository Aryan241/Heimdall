"""
Device detection utility — single source of truth for the entire project.
"""

import logging
import os

import torch

# Make torch's device indices match nvidia-smi's. Without this, torch sorts by capability
# ("fastest first"), so cuda:1 on a mixed 1080/3070/1080 box is not the card you meant.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    Priority: explicit preference > CUDA > MPS > CPU.
    If *preferred* is given but unavailable, falls back gracefully.
    """
    if preferred:
        preferred = preferred.lower()
        if preferred.startswith("cuda") and torch.cuda.is_available():
            # Accept "cuda" and "cuda:N" (workstations with several GPUs).
            index = int(preferred.split(":", 1)[1]) if ":" in preferred else None
            if index is None:
                return torch.device("cuda")
            if index < torch.cuda.device_count():
                return torch.device(f"cuda:{index}")
            logging.getLogger(__name__).warning(
                "cuda:%d requested but only %d CUDA device(s) present — using cuda:0.",
                index, torch.cuda.device_count())
            return torch.device("cuda:0")
        if preferred == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if preferred == "cpu":
            return torch.device("cpu")
        # requested device not available — fall through

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def supports_amp(device: torch.device) -> bool:
    """Return True only if the device supports torch.cuda.amp (Ampere+ CUDA)."""
    if device.type != "cuda":
        return False
    cap = torch.cuda.get_device_capability(device)
    return cap[0] >= 7  # Volta+ for basic AMP, Ampere+ for full benefit
