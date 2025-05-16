import torch
import torch.nn as nn
from models.vision_transformer import VisionTransformer3D

def create_vit_model(vol_size=(182, 218, 182), in_channels=1, num_classes=2, 
                     patch_size=16, embed_dim=768, depth=12, num_heads=12, 
                     mlp_ratio=4.0, drop_rate=0.1, attn_drop_rate=0.1):
    """
    Create a Vision Transformer model with customizable parameters.
    
    Parameters:
    -----------
    vol_size : tuple
        Input volume size (D, H, W)
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes
    patch_size : int
        Size of patches to divide the input volume
    embed_dim : int
        Embedding dimension
    depth : int
        Number of transformer blocks
    num_heads : int
        Number of attention heads
    mlp_ratio : float
        Ratio of MLP hidden dimension to embedding dimension
    drop_rate : float
        Dropout rate
    attn_drop_rate : float
        Attention dropout rate
        
    Returns:
    --------
    torch.nn.Module
        The configured Vision Transformer model
    """
    if depth < 1:
        raise ValueError(f"Depth must be at least 1, got {depth}")
    
    if num_heads < 1:
        raise ValueError(f"Number of heads must be at least 1, got {num_heads}")
        
    if embed_dim % num_heads != 0:
        raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
    
    
    model = VisionTransformer3D(
        vol_size=vol_size,
        patch_size=patch_size,
        in_channels=in_channels,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        qkv_bias=True,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        num_classes=num_classes
    )
    
    return model
