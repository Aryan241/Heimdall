"""
Device detection utility — single source of truth for the entire project.
"""

import torch


def get_device(preferred: str | None = None) -> torch.device:
    """Return the best available torch device.

    Priority: explicit preference > CUDA > MPS > CPU.
    If *preferred* is given but unavailable, falls back gracefully.
    """
    if preferred:
        preferred = preferred.lower()
        if preferred == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
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
