import torch
import torch.nn as nn
import torch.nn.functional as F

class SILogLoss(nn.Module):
    """
    Scale-Invariant Logarithmic (SILog) Loss.
    Standard metric depth estimation loss (Eigen et al.)
    It penalizes the variance of the log error, focusing on the structural relationships
    rather than being heavily penalized by absolute scale mistakes initially.
    """
    def __init__(self, variance_focus=0.85, eps=1e-6):
        super().__init__()
        self.variance_focus = variance_focus
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Apply mask and ensure values are strictly positive for log
        p = pred[mask].clamp(min=self.eps)
        t = target[mask].clamp(min=self.eps)

        # Log difference
        d = torch.log(p) - torch.log(t)

        # Loss = sqrt(mean(d^2) - lambda * (mean(d))^2)
        # Often scaled by 10 or 100 for better gradient magnitudes
        loss = torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0
        return loss

class GradientMatchingLoss(nn.Module):
    """
    Gradient matching loss to preserve sharp edges and structural details.
    """
    def __init__(self, scales=4):
        super().__init__()
        self.scales = scales

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss = 0.0
        # Compute gradient matching at multiple scales
        for i in range(self.scales):
            step = 2 ** i
            
            # Compute differences (x and y gradients)
            diff_pred_x = pred[:, :, :, step:] - pred[:, :, :, :-step]
            diff_pred_y = pred[:, :, step:, :] - pred[:, :, :-step, :]
            
            diff_target_x = target[:, :, :, step:] - target[:, :, :, :-step]
            diff_target_y = target[:, :, step:, :] - target[:, :, :-step, :]

            # Valid masks at this scale
            mask_x = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_y = mask[:, :, step:, :] & mask[:, :, :-step, :]

            # We use L1 loss on the gradients
            if mask_x.sum() > 0:
                loss += F.l1_loss(diff_pred_x[mask_x], diff_target_x[mask_x])
            if mask_y.sum() > 0:
                loss += F.l1_loss(diff_pred_y[mask_y], diff_target_y[mask_y])

        return loss

class MetricDepthLoss(nn.Module):
    """
    Combined State-of-the-Art Loss for Metric Depth Estimation.
    Combines SILog Loss (structure), Gradient Matching (edges), and a small L1 term for absolute anchoring.
    """
    def __init__(self, alpha=1.0, beta=0.5, gamma=0.1):
        super().__init__()
        self.silog = SILogLoss(variance_focus=0.85)
        self.grad = GradientMatchingLoss(scales=4)
        self.l1 = nn.L1Loss()
        
        self.alpha = alpha  # SILog weight
        self.beta = beta    # Gradient matching weight
        self.gamma = gamma  # Absolute L1 anchor weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Expects pred and target shapes: (B, 1, H, W)
        """
        # Create validity mask (ignore missing data / negative heights)
        mask = (target > -1000) & (~torch.isnan(target))
        
        if mask.sum() == 0:
            zero = torch.tensor(0.0, device=pred.device, requires_grad=True)
            return {"loss": zero, "silog": zero, "grad": zero, "l1": zero}

        l_silog = self.silog(pred, target, mask)
        l_grad = self.grad(pred, target, mask)
        l_l1 = self.l1(pred[mask], target[mask])

        total_loss = self.alpha * l_silog + self.beta * l_grad + self.gamma * l_l1

        return {
            "loss": total_loss,
            "silog": l_silog.detach(),
            "grad": l_grad.detach(),
            "l1": l_l1.detach()
        }
