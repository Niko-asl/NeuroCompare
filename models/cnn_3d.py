import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock3D(nn.Module):
    #3D Residual block with two convolutions and a skip connection
    
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.3):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.dropout1 = nn.Dropout3d(dropout_rate)
        
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.dropout2 = nn.Dropout3d(dropout_rate)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, 
                          stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += self.shortcut(residual)
        out = F.relu(out)
        out = self.dropout2(out)
        
        return out


# Debug originally, turned out useful
def check_output_values(outputs, name="layer"):
        has_nan = torch.isnan(outputs).any()
        has_inf = torch.isinf(outputs).any()
        
        if has_nan or has_inf:
            print(f"WARNING: {name} has {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} values")
            return False
        min_val=outputs.min().item()
        max_val=outputs.max().item()
        mean_val=outputs.mean().item()
        std_val=outputs.std().item()
        
        if min_val == max_val:
            print(f"WARNING: {name} has constant value: {min_val}")
            return False
        if std_val < 1e-6:
            print(f"WARNING {name} has small variance: std={std_val}")
            return False
        
        # Do not uncomment unless everything is on fire and you dont know why 
        #if not has_nan and not has_inf:
         #   print(f"No NaN or Inf values detected in model")
          #  return False
        return True
        
        
def track_parameters(model, step):
    if step % 100 == 0:
        # Track the last layer's weights and biases
        fc_layer = model.fc3 if hasattr(model, 'fc3') else model.fc
        w_norm = torch.norm(fc_layer.weight).item()
        if fc_layer.bias is not None:
            b_norm = torch.norm(fc_layer.bias).item()
            b_mean = fc_layer.bias.mean().item()
            print(f"Step {step}: Weight norm={w_norm:.4f}, Bias norm={b_norm:.4f}, Bias mean={b_mean:.4f}")
        else:
            print(f"Step {step}: Weight norm={w_norm:.4f}, No bias")




class CNN3D(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=2, base_filters=32, dropout_rate=0.5):
        super(CNN3D, self).__init__()
        
        self.dropout_rate = dropout_rate
        

        self.conv1 = nn.Conv3d(in_channels, base_filters, kernel_size=3, 
                              stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters, momentum=0.1)  # More stable momentum - could be added to arguments?
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # first stage
        self.res1a = ResidualBlock3D(base_filters, base_filters, dropout_rate=dropout_rate/2)
        self.res1b = ResidualBlock3D(base_filters, base_filters, dropout_rate=dropout_rate/2)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # second stage
        self.res2a = ResidualBlock3D(base_filters, base_filters*2, stride=1, dropout_rate=dropout_rate/2)
        self.res2b = ResidualBlock3D(base_filters*2, base_filters*2, stride=1, dropout_rate=dropout_rate/2)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # third stage
        self.res3a = ResidualBlock3D(base_filters*2, base_filters*4, stride=1, dropout_rate=dropout_rate/2)
        self.res3b = ResidualBlock3D(base_filters*4, base_filters*4, stride=1, dropout_rate=dropout_rate/2)
        self.pool4 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # global avg pool
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(base_filters*4, 512)  
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, 256)  
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(256, num_classes)
        
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
        check_output_values(x, "conv1")
        x = self.bn1(x)
        check_output_values(x, "bn1")
        x = F.relu(x)
        x = self.pool1(x)
        check_output_values(x, "pool1")
        
        # Stage 1
        x = self.res1a(x)
        check_output_values(x, "res1a")
        x = self.res1b(x)
        check_output_values(x, "res1b")
        x = self.pool2(x)
        check_output_values(x, "pool2")
        
        # Stage 2
        x = self.res2a(x)
        check_output_values(x, "res2a")
        x = self.res2b(x)
        check_output_values(x, "res2b")
        x = self.pool3(x)
        check_output_values(x, "pool3")
        
        # Stage 3
        x = self.res3a(x)
        check_output_values(x, "res3a")
        x = self.res3b(x)
        check_output_values(x, "res3b")
        x = self.pool4(x)
        check_output_values(x, "pool4")
        
        # Global avg pool
        x = self.avg_pool(x)
        check_output_values(x, "avg_pool")
        x = x.view(x.size(0), -1)  # Flatten
        
        # Classifier
        x = self.fc1(x)
        check_output_values(x, "fc1")
        x = F.relu(x)
        x = self.dropout1(x)
        check_output_values(x, "dropout1")
        x = self.fc2(x)
        check_output_values(x, "fc2")
        x = F.relu(x)
        x = self.dropout2(x)
        check_output_values(x, "dropout2")
        x = self.fc3(x)
        check_output_values(x, "fc3")
        
        return x


