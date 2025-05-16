import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cnn_3d import CNN3D, ResidualBlock3D

def create_cnn_model(in_channels=1, num_classes=2, num_layers=3, base_filters=32, dropout_rate=0.5):
    """
    Create a CNN model with the specified number of layers.
    
    Parameters:
    -----------
    in_channels : int
        Number of input channels
    num_classes : int
        Number of output classes
    num_layers : int
        Number of residual blocks (2-6)
    base_filters : int
        Number of base filters
    dropout_rate : float
        Dropout rate
        
    Returns:
    --------
    torch.nn.Module
        The configured CNN model
    """

    # Right now we have support for 2-6 layers
    # Anything else really isnt common so this should be fine but can be modified
    if num_layers < 2 or num_layers > 6:
        raise ValueError(f"Number of layers must be between 2 and 6, got {num_layers}")
    
    # For the default CNN3D, we just return the standard 3 stage model
    if num_layers == 3:
        return CNN3D(in_channels=in_channels, num_classes=num_classes, 
                     base_filters=base_filters, dropout_rate=dropout_rate)
    
    # For other layer counts - generate model on fly
    class CustomCNN3D(nn.Module):
        def __init__(self, in_channels, num_classes, num_layers, base_filters, dropout_rate):
            super(CustomCNN3D, self).__init__()
            
            self.dropout_rate = dropout_rate
            
            self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=3, 
                                  stride=2, padding=1, bias=False)
            self.bn1 = nn.BatchNorm3d(base_filters, momentum=0.1)
            self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
            
            self.residual_blocks = nn.ModuleList()
            self.pooling_layers = nn.ModuleList()
            
            current_filters = base_filters
            
            for i in range(num_layers):
                if i == 0:
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, current_filters, dropout_rate=dropout_rate/2)
                    )
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, current_filters, dropout_rate=dropout_rate/2)
                    )
                elif i == 1:
                    # Second block - double filters
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, current_filters*2, stride=1, dropout_rate=dropout_rate/2)
                    )
                    current_filters *= 2
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, current_filters, dropout_rate=dropout_rate/2)
                    )
                else:
                    new_filters = min(current_filters*2, 512)  
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, new_filters, stride=1, dropout_rate=dropout_rate/2)
                    )
                    current_filters = new_filters
                    self.residual_blocks.append(
                        ResidualBlock3D(current_filters, current_filters, dropout_rate=dropout_rate/2)
                    )
                
                self.pooling_layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            
            fc1_size = min(current_filters, 512)
            fc2_size = max(256, fc1_size // 2)
            
            self.fc1 = nn.Linear(current_filters, fc1_size)
            self.dropout1 = nn.Dropout(dropout_rate)
            self.fc2 = nn.Linear(fc1_size, fc2_size)
            self.dropout2 = nn.Dropout(dropout_rate)
            self.fc3 = nn.Linear(fc2_size, num_classes)
            
            self._init_weights()
        
        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    if m == self.fc3:
                        nn.init.normal_(m.weight, std=0.001)
                        nn.init.constant_(m.bias, 0)
                    else: 
                        nn.init.normal_(m.weight, std=0.01)
                        nn.init.constant_(m.bias, 0)
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            
            # Apply residual blocks with pooling layers
            for i in range(num_layers):
                block_idx = i * 2
                x = self.residual_blocks[block_idx](x)
                x = self.residual_blocks[block_idx + 1](x)
                x = self.pooling_layers[i](x)
            
            x = self.avg_pool(x)
            x = x.view(x.size(0), -1)
            
            x = self.fc1(x)
            x = F.relu(x)
            x = self.dropout1(x)
            x = self.fc2(x)
            x = F.relu(x)
            x = self.dropout2(x)
            x = self.fc3(x)
            
            return x
    
    return CustomCNN3D(in_channels, num_classes, num_layers, base_filters, dropout_rate)
