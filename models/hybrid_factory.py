import torch
import torch.nn as nn
from models.hybrid_model import HybridCNNTransformer

def create_hybrid_model(vol_size=(182, 218, 182), in_channels=1, num_classes=2,
                        feature_dim=256, transformer_dim=256, num_heads=8, 
                        depth=4, mlp_ratio=4.0, drop_rate=0.1):
    """
    Parameters:
    -----------
    vol_size : tuple
        Input volume size (D, H, W)
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes
    feature_dim : int
        Dimension of CNN features
    transformer_dim : int
        Dimension for transformer blocks
    num_heads : int
        Number of attention heads
    depth : int
        Number of transformer blocks
    mlp_ratio : float
        Ratio of MLP hidden dimension to transformer dimension
    drop_rate : float
        Dropout rate
    
    Returns:
    --------
    torch.nn.Module
        The configured Hybrid CNN-Transformer model
    """
   
    if depth < 1:
        raise ValueError(f"Depth must be at least 1, got {depth}")
    
    if num_heads < 1:
        raise ValueError(f"Number of heads must be at least 1, got {num_heads}")
    

    if transformer_dim % num_heads != 0:
        raise ValueError(f"Transformer dimension ({transformer_dim}) must be divisible by number of heads ({num_heads})")
    

    model = HybridCNNTransformer(
        vol_size=vol_size,
        in_channels=in_channels,
        feature_dim=feature_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        depth=depth,
        mlp_ratio=mlp_ratio,
        drop_rate=drop_rate,
        num_classes=num_classes
    )
    
    return model
