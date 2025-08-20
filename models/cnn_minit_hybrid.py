# models/cnn_minit_hybrid.py (updated constructor)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.minit import create_nit_model, create_minit_model, create_mignit_model
from models.cnn_3d import ResidualBlock3D

class CNNMINiTHybrid(nn.Module):
    """
    Hybrid CNN-MINiT model with learned downsampling.
    Maintains full configurability of the MINiT backend.
    """
    
    def __init__(self, 
                 # CNN frontend parameters
                 target_size=64,  # 64, 80, or 96
                 cnn_dropout=0.3,
                 
                 # MINiT backend parameters - matching base MINiT flexibility
                 minit_variant='minit',  # 'nit', 'minit', or 'mignit'
                 minit_block_size=16,
                 minit_patch_size=4,
                 minit_embed_dim=256,
                 minit_depth=6,
                 minit_heads=8,
                 minit_mlp_ratio=1.2,  # Note: matching the default 1.2 from base MINiT
                 minit_dropout=0.125,
                 minit_attn_dropout=0.125,
                 
                 # General parameters
                 num_classes=2,
                 amp_enabled=False):
        super().__init__()
        
        self.target_size = target_size
        self.minit_variant = minit_variant
        
        # Build appropriate CNN frontend based on target size
        if target_size == 64:
            self.cnn_frontend = self._build_cnn_to_64(cnn_dropout)
        elif target_size == 80:
            self.cnn_frontend = self._build_cnn_to_80(cnn_dropout)
        elif target_size == 96:
            self.cnn_frontend = self._build_cnn_to_96(cnn_dropout)
        else:
            raise ValueError(f"Unsupported target size: {target_size}. Choose 64, 80, or 96.")
        
        # Build MINiT backend with full configurability
        minit_kwargs = {
            'image_size': target_size,
            'patch_size': minit_patch_size,
            'num_classes': num_classes,
            'dim': minit_embed_dim,
            'depth': minit_depth,
            'heads': minit_heads,
            'mlp_dim': int(minit_embed_dim * minit_mlp_ratio),
            'channels': 1,  # CNN outputs single channel
            'dropout': minit_dropout,
            'emb_dropout': minit_dropout,  # Paper uses same for both
            'drophead': minit_attn_dropout,  # This maps to attention dropout
            'amp_enabled': amp_enabled
        }
        
        if minit_variant == 'nit':
            self.minit_backend = create_nit_model(**minit_kwargs)
        elif minit_variant == 'minit':
            minit_kwargs['block_size'] = minit_block_size
            self.minit_backend = create_minit_model(**minit_kwargs)
        elif minit_variant == 'mignit':
            minit_kwargs['block_size'] = minit_block_size
            self.minit_backend = create_mignit_model(**minit_kwargs)
        else:
            raise ValueError(f"Unknown MINiT variant: {minit_variant}")
        
        # Log configuration
        self._log_configuration()
    
    def _log_configuration(self):
        """Log the hybrid model configuration for debugging."""
        print(f"Created CNN-{self.minit_variant.upper()} Hybrid:")
        print(f"  CNN: 182×218×182 -> {self.target_size}³")
        print(f"  MINiT variant: {self.minit_variant}")
        if self.minit_variant in ['minit', 'mignit']:
            print(f"  MINiT blocks: {self.target_size}³ with {self.minit_backend.block_size}³ blocks")
        print(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _build_cnn_to_64(self, dropout_rate):
        """
        CNN to downsample from 182×218×182 to 64×64×64.
        Total downsampling needed: ~2.84× per dimension
        Strategy: Use 3 stages with strategic strides and padding
        """
        return nn.Sequential(
            # Stage 1: 182×218×182 -> 91×109×91 (stride 2)
            nn.Conv3d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32, dropout_rate=dropout_rate),
            
            # Stage 2: 91×109×91 -> 45×54×45 (stride 2 with some cropping)
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64, dropout_rate=dropout_rate),
            
            # Stage 3: Careful downsampling to exactly 64×64×64
            # We'll use adaptive pooling for precise control
            nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            ResidualBlock3D(128, 128, dropout_rate=dropout_rate),
            
            # Final projection and adaptive pooling to exact size
            nn.Conv3d(128, 1, kernel_size=1),
            nn.AdaptiveAvgPool3d((64, 64, 64))
        )
    
    def _build_cnn_to_80(self, dropout_rate):
        """
        CNN to downsample from 182×218×182 to 80×80×80.
        Total downsampling needed: ~2.27× per dimension
        """
        return nn.Sequential(
            # Stage 1: 182×218×182 -> 91×109×91 (stride 2)
            nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32, dropout_rate=dropout_rate),
            
            # Stage 2: 91×109×91 -> 80×80×80 (using adaptive pooling)
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64, dropout_rate=dropout_rate),
            
            # Final projection and precise resizing
            nn.Conv3d(64, 1, kernel_size=1),
            nn.AdaptiveAvgPool3d((80, 80, 80))
        )
    
    def _build_cnn_to_96(self, dropout_rate):
        """
        CNN to downsample from 182×218×182 to 96×96×96.
        Total downsampling needed: ~1.89× per dimension
        """
        return nn.Sequential(
            # Stage 1: 182×218×182 -> 96×96×96 (nearly 2x downsample)
            # We'll use stride 2 then adaptive pool
            nn.Conv3d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            ResidualBlock3D(32, 32, dropout_rate=dropout_rate),
            
            # Add depth without changing resolution much
            nn.Conv3d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            ResidualBlock3D(64, 64, dropout_rate=dropout_rate),
            
            # Final projection and adaptive pooling
            nn.Conv3d(64, 1, kernel_size=1),
            nn.AdaptiveAvgPool3d((96, 96, 96))
        )
    
    def forward(self, x):
        # CNN feature extraction and downsampling
        x = self.cnn_frontend(x)
        
        # MINiT processing
        x = self.minit_backend(x)
        
        return x
