# models/cnn_minit_factory.py

from models.cnn_minit_hybrid import CNNMINiTHybrid

def create_cnn_minit_model(
    target_size=64,
    minit_variant='minit',
    minit_block_size=16,
    minit_patch_size=4,
    minit_embed_dim=256,
    minit_depth=6,
    minit_heads=8,
    minit_mlp_ratio=4,
    minit_dropout=0.1,
    cnn_dropout=0.3,
    num_classes=2,
    amp_enabled=False,
    **kwargs
):
    """
    Factory function for CNN-MINiT hybrid models.
    
    Parameters:
    -----------
    target_size : int
        Target size for CNN downsampling (64, 80, or 96)
    minit_variant : str
        MINiT variant to use ('nit', 'minit', 'mignit')
    ... (other parameters)
    """
    
    # Validate inputs
    if target_size not in [64, 80, 96]:
        raise ValueError(f"target_size must be 64, 80, or 96, got {target_size}")
    
    if minit_variant not in ['nit', 'minit', 'mignit']:
        raise ValueError(f"minit_variant must be 'nit', 'minit', or 'mignit', got {minit_variant}")
    
    # Validate MINiT parameters based on target size
    if minit_variant in ['minit', 'mignit']:
        if target_size % minit_block_size != 0:
            raise ValueError(f"target_size {target_size} must be divisible by block_size {minit_block_size}")
        if minit_block_size % minit_patch_size != 0:
            raise ValueError(f"block_size {minit_block_size} must be divisible by patch_size {minit_patch_size}")
    elif minit_variant == 'nit':
        if target_size % minit_patch_size != 0:
            raise ValueError(f"target_size {target_size} must be divisible by patch_size {minit_patch_size}")
    
    return CNNMINiTHybrid(
        target_size=target_size,
        minit_variant=minit_variant,
        minit_block_size=minit_block_size,
        minit_patch_size=minit_patch_size,
        minit_embed_dim=minit_embed_dim,
        minit_depth=minit_depth,
        minit_heads=minit_heads,
        minit_mlp_ratio=minit_mlp_ratio,
        minit_dropout=minit_dropout,
        cnn_dropout=cnn_dropout,
        num_classes=num_classes,
        amp_enabled=amp_enabled
    )
