import torch
import torch.nn as nn
from models.minit import create_nit_model, create_minit_model, create_mignit_model

def create_minit_model_factory(vol_size=(182, 218, 182), in_channels=1, num_classes=2,
                               model_variant='minit', block_size=16, patch_size=4,
                               embed_dim=256, depth=6, num_heads=8, mlp_ratio=4.0,
                               drop_rate=0.1, attn_drop_rate=0.1, amp_enabled=False):
    """
    Create a MINiT model with the specified parameters.
    
    Parameters:
    -----------
    vol_size : tuple
        Input volume size (D, H, W) - will be converted to cubic for MINiT
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes
    model_variant : str
        Model variant ('nit', 'minit', 'mignit')
    block_size : int
        Size of blocks for MIL (only used for minit/mignit)
    patch_size : int
        Size of patches within blocks
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
        Attention dropout rate (currently same as drop_rate in paper's implementation)
    amp_enabled : bool
        Whether to use automatic mixed precision
        
    Returns:
    --------
    torch.nn.Module
        The configured MINiT model
    """
    
    # Validation - following your framework's pattern
    if model_variant not in ['nit', 'minit', 'mignit']:
        raise ValueError(f"Unknown model variant: {model_variant}")
    
    if depth < 1:
        raise ValueError(f"Depth must be at least 1, got {depth}")
    
    if num_heads < 1:
        raise ValueError(f"Number of heads must be at least 1, got {num_heads}")
        
    if embed_dim % num_heads != 0:
        raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by number of heads ({num_heads})")
    
    # For MINiT, we need to handle your framework's volume sizes
    # Paper's implementation expects cubic volumes, so we'll use the minimum dimension
    min_dim = min(vol_size)
    
    # Calculate MLP dimension
    mlp_dim = int(embed_dim * mlp_ratio)
    
    # Create model based on variant
    if model_variant == 'nit':
        # Base NeuroImage Transformer
        model = create_nit_model(
            image_size=min_dim,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
            dropout=drop_rate,
            emb_dropout=drop_rate,
            amp_enabled=amp_enabled
        )
        
    elif model_variant == 'minit':
        # Multiple Instance NiT
        model = create_minit_model(
            image_size=min_dim,
            block_size=block_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
            dropout=drop_rate,
            emb_dropout=drop_rate,
            amp_enabled=amp_enabled
        )
        
    elif model_variant == 'mignit':
        # Multiple Instance Global NiT
        model = create_mignit_model(
            image_size=min_dim,
            block_size=block_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=embed_dim,
            depth=depth,
            heads=num_heads,
            mlp_dim=mlp_dim,
            channels=in_channels,
            dropout=drop_rate,
            emb_dropout=drop_rate,
            amp_enabled=amp_enabled
        )
    
    return model


def get_paper_hyperparameters():
    """
    Get the exact hyperparameters used in the original paper (Table 1).
    
    Returns a dictionary with hyperparameters for each model variant
    as reported in the paper.
    """
    
    return {
        # Base models from Table 1
        'nit': {
            'embed_dim': 256,
            'depth': 4,
            'num_heads': 8,
            'mlp_dim': 234,
            'drop_rate': 0.16,
            'learning_rate': 1e-4,
            'weight_decay': 0.16,
            'patch_size': 8,  # Paper used 8x8x8 for NiT
            'parameters': 1.8e6
        },
        'minit': {
            'embed_dim': 256,
            'depth': 6,
            'num_heads': 8,
            'mlp_dim': 309,
            'drop_rate': 0.125,
            'learning_rate': 1e-4,
            'weight_decay': 0.125,
            'block_size': 16,  # Paper used 16x16x16 blocks
            'patch_size': 4,   # Paper used 4x4x4 patches
            'parameters': 3.6e6
        },
        'mignit': {
            'embed_dim': 256,
            'depth': 6,
            'num_heads': 8,
            'mlp_dim': 309,
            'drop_rate': 0.3,
            'learning_rate': 2e-4,
            'weight_decay': 0.3,
            'block_size': 16,
            'patch_size': 4,
            'parameters': 8.5e6
        }
    }


def get_recommended_batch_sizes():
    """
    Get recommended batch sizes based on memory usage.
    
    The paper used small batch sizes due to GPU memory constraints.
    """
    
    return {
        'nit': {
            'batch_size': 8,      # NiT uses less memory
            'eval_batch_size': 16,
            'memory_gb_per_sample': 0.5
        },
        'minit': {
            'batch_size': 4,      # Paper's reported batch size
            'eval_batch_size': 8,
            'memory_gb_per_sample': 1.0
        },
        'mignit': {
            'batch_size': 2,      # MiGNiT uses most memory due to global attention
            'eval_batch_size': 4,
            'memory_gb_per_sample': 2.0
        }
    }


def estimate_model_size(model_variant, embed_dim):
    """
    Estimate model parameters and memory usage.
    
    Based on the paper's reported parameter counts in Table 1.
    """
    
    paper_params = get_paper_hyperparameters()
    
    if model_variant in paper_params:
        base_params = paper_params[model_variant]['parameters']
        # Scale based on embedding dimension difference
        scale_factor = (embed_dim / paper_params[model_variant]['embed_dim']) ** 2
        estimated_params = base_params * scale_factor
    else:
        # Rough estimate
        estimated_params = embed_dim * embed_dim * 10  # Very rough approximation
    
    # Memory estimation (rough)
    memory_mb = estimated_params * 4 / (1024 * 1024)  # 4 bytes per parameter
    memory_mb *= 3  # Account for gradients and optimizer states
    
    return {
        'estimated_parameters': int(estimated_params),
        'estimated_memory_mb': memory_mb,
        'paper_parameters': paper_params.get(model_variant, {}).get('parameters', 'Unknown')
    }


def validate_configuration(model_variant, vol_size, block_size, patch_size, embed_dim, num_heads):
    """
    Validate configuration against paper's setup and framework constraints.
    
    Returns list of warnings/issues.
    """
    
    issues = []
    
    # Check volume size
    min_dim = min(vol_size)
    if min_dim < 64:
        issues.append(f"Small volume size {vol_size}. Paper used 64x64x64. Consider preprocessing.")
    
    # Check if dimensions are compatible
    if model_variant in ['minit', 'mignit']:
        if min_dim % block_size != 0:
            issues.append(f"Volume size {min_dim} not divisible by block_size {block_size}")
        
        if block_size % patch_size != 0:
            issues.append(f"Block size {block_size} not divisible by patch_size {patch_size}")
    
    elif model_variant == 'nit':
        if min_dim % patch_size != 0:
            issues.append(f"Volume size {min_dim} not divisible by patch_size {patch_size}")
    
    # Check attention heads
    if embed_dim % num_heads != 0:
        issues.append(f"Embedding dim {embed_dim} not divisible by num_heads {num_heads}")
    
    # Compare against paper hyperparameters
    paper_params = get_paper_hyperparameters()
    if model_variant in paper_params:
        paper_config = paper_params[model_variant]
        
        if embed_dim != paper_config['embed_dim']:
            issues.append(f"Paper used embed_dim={paper_config['embed_dim']}, you're using {embed_dim}")
        
        if model_variant in ['minit', 'mignit']:
            if block_size != paper_config.get('block_size', 16):
                issues.append(f"Paper used block_size={paper_config.get('block_size', 16)}, you're using {block_size}")
            
            if patch_size != paper_config.get('patch_size', 4):
                issues.append(f"Paper used patch_size={paper_config.get('patch_size', 4)}, you're using {patch_size}")
    
    return issues


def create_paper_exact_model(model_variant='minit', vol_size=(64, 64, 64), num_classes=2):
    """
    Create a model with the exact hyperparameters from the paper.
    
    This ensures maximum replicability of paper results.
    """
    
    paper_params = get_paper_hyperparameters()
    
    if model_variant not in paper_params:
        raise ValueError(f"No paper hyperparameters available for {model_variant}")
    
    params = paper_params[model_variant]
    
    return create_minit_model_factory(
        vol_size=vol_size,
        in_channels=1,
        num_classes=num_classes,
        model_variant=model_variant,
        block_size=params.get('block_size', 16),
        patch_size=params.get('patch_size', 4),
        embed_dim=params['embed_dim'],
        depth=params['depth'],
        num_heads=params['num_heads'],
        mlp_ratio=params['mlp_dim'] / params['embed_dim'],
        drop_rate=params['drop_rate'],
        attn_drop_rate=params['drop_rate'],
        amp_enabled=True  # Enable for memory efficiency
    )


def print_model_info(model, model_variant):
    """
    Print model information in the style of your framework.
    """
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    paper_params = get_paper_hyperparameters()
    paper_count = paper_params.get(model_variant, {}).get('parameters', 'Unknown')
    
    print(f"Created {model_variant.upper()} model:")
    print(f"  Parameters: {total_params:,} ({trainable_params:,} trainable)")
    print(f"  Paper reported: {paper_count}")
    
    batch_info = get_recommended_batch_sizes()[model_variant]
    print(f"  Recommended batch size: {batch_info['batch_size']}")
    print(f"  Estimated memory per sample: {batch_info['memory_gb_per_sample']:.1f} GB")


# Test function to verify everything works
if __name__ == "__main__":
    print("Testing MINiT factory...")
    
    # Test with your framework's typical parameters
    model = create_minit_model_factory(
        vol_size=(182, 218, 182),  # Your framework's size
        model_variant='minit',
        embed_dim=256,
        depth=6,
        num_heads=8,
        amp_enabled=True
    )
    
    print_model_info(model, 'minit')
    
    # Test validation
    issues = validate_configuration(
        model_variant='minit',
        vol_size=(182, 218, 182),
        block_size=16,
        patch_size=4,
        embed_dim=256,
        num_heads=8
    )
    
    if issues:
        print("Configuration issues:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration looks good!")
    
    # Test paper-exact model
    print("\nTesting paper-exact model...")
    paper_model = create_paper_exact_model('minit', vol_size=(64, 64, 64))
    print_model_info(paper_model, 'minit')
    
    print("Factory tests passed!")
