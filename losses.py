# coding=utf-8
import torch
import math

def equirectangular_area_weights(height, device, dtype=torch.float32):
    pixel_h = math.pi / height
    colatitude = torch.linspace(pixel_h / 2, math.pi - pixel_h / 2, height, device=device, dtype=dtype)
    # Shape [1, 1, H, 1] to broadcast with [B, C, H, W]
    return torch.sin(colatitude).view(1, 1, height, 1)

def direction_loss(v_pred, v_true):
    """
    v_pred: [BATCH, 3]
    v_true: [BATCH, 3]
    Returns scalar
    """
    # Assuming already unit vectors
    return -torch.mean(torch.sum(v_pred * v_true, dim=-1))

def distribution_loss(p_pred, p_true):
    """
    p_pred: [BATCH, C, HEIGHT, WIDTH]
    p_true: [BATCH, C, HEIGHT, WIDTH]
    """
    batch, channels, height, width = p_pred.shape
    weights = equirectangular_area_weights(height, device=p_pred.device, dtype=p_pred.dtype)
    return torch.mean(weights * (p_pred - p_true)**2)

def spread_loss(v_pred):
    """
    v_pred: [BATCH, 3]
    """
    return 1 - torch.mean(torch.norm(v_pred, dim=-1))
