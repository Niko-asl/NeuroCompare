import torch
import torch.nn as nn
import torch.nn.functional as F


# 3D CNN frontent
class CNNFeatureExtractor(nn.Module):
    
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(64)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm3d(128)
        self.conv4 = nn.Conv3d(128, feature_dim, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm3d(feature_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        return x

# Hybridized with a ViT backend
class HybridCNNTransformer(nn.Module):
    
    def __init__(self, vol_size=(128, 218, 128), in_channels=1, feature_dim=256, 
                 transformer_dim=256, num_heads=8, depth=4, mlp_ratio=4, 
                 drop_rate=0.1, num_classes=2):
        super().__init__()
        
        # CNN feature extractor
        self.feature_extractor = CNNFeatureExtractor(
            in_channels=in_channels, feature_dim=feature_dim
        )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *vol_size)
            cnn_output = self.feature_extractor(dummy_input)
            self.feature_size = cnn_output.shape[2:]  # D, H, W
            self.num_patches = cnn_output.shape[2] * cnn_output.shape[3] * cnn_output.shape[4]
        
        # Projection layer to transformer dimension
        self.projection = nn.Linear(feature_dim, transformer_dim)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, transformer_dim))
        
        # Transformer blocks
        transformer_blocks = []
        for _ in range(depth):
            transformer_blocks.append(
                nn.TransformerEncoderLayer(
                    batch_first=True,
                    d_model=transformer_dim,
                    nhead=num_heads,
                    dim_feedforward=int(transformer_dim * mlp_ratio),
                    dropout=drop_rate,
                    activation='gelu'
                )
            )
        self.transformer = nn.Sequential(*transformer_blocks)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(transformer_dim)
        self.classifier = nn.Linear(transformer_dim, num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):

        features = self.feature_extractor(x)  # B, C, D, H, W
        
        B, C, D, H, W = features.shape
        features = features.permute(0, 2, 3, 4, 1)  # B, D, H, W, C
        features = features.reshape(B, D*H*W, C)  # B, DHW, C
        
        # Project to transformer dimension
        features = self.projection(features)  # B, DHW, transformer_dim
        
        features = features + self.pos_embed

        features = self.transformer(features)  # B, DHW, transformer_dim
        
        features = features.permute(0, 2, 1)  # B, transformer_dim, DHW
        features = self.global_pool(features).squeeze(-1)  # B, transformer_dim
        
        features = self.norm(features)
        output = self.classifier(features)
        
        return output
