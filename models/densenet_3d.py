import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class _DenseLayer(nn.Module):
    """Individual dense layer with growth rate."""
    
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(num_input_features, bn_size * growth_rate,
                              kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(bn_size * growth_rate, growth_rate,
                              kernel_size=3, stride=1, padding=1, bias=False)
        
        self.drop_rate = float(drop_rate)

    def bn_function(self, inputs):
        """Bottleneck function for memory efficiency."""
        if isinstance(inputs, torch.Tensor):
            prev_features = [inputs]
        else:
            prev_features = inputs
            
        concated_features = torch.cat(prev_features, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def forward(self, input):
        if isinstance(input, torch.Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                   training=self.training)
        return new_features


class _DenseBlock(nn.Module):
    """Dense block consisting of multiple dense layers."""
    
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Module):
    """Transition layer between dense blocks."""
    
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(num_input_features, num_output_features,
                             kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out


class DenseNet3D(nn.Module):
    """3D DenseNet implementation matching the paper's architecture."""
    
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0.0, num_classes=2):
        super(DenseNet3D, self).__init__()

        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv3d(1, num_init_features, kernel_size=7, stride=2,
                               padding=3, bias=False)),
            ('norm0', nn.BatchNorm3d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool3d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                   num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool3d(out, (1, 1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def densenet121_3d(**kwargs):
    """DenseNet-121 3D model"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 24, 16), **kwargs)


def densenet169_3d(**kwargs):
    """DenseNet-169 3D model"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 32, 32), **kwargs)


def densenet201_3d(**kwargs):
    """DenseNet-201 3D model"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 48, 32), **kwargs)


def densenet264_3d(**kwargs):
    """DenseNet-264 3D model - matches the paper exactly"""
    return DenseNet3D(growth_rate=32, block_config=(6, 12, 64, 48), **kwargs)


# For compatibility with your factory pattern
def create_densenet_model(densenet_type='264', num_classes=2, drop_rate=0.0):
    """Factory function to create DenseNet models."""
    
    if densenet_type == '121':
        return densenet121_3d(num_classes=num_classes, drop_rate=drop_rate)
    elif densenet_type == '169':
        return densenet169_3d(num_classes=num_classes, drop_rate=drop_rate)
    elif densenet_type == '201':
        return densenet201_3d(num_classes=num_classes, drop_rate=drop_rate)
    elif densenet_type == '264':
        return densenet264_3d(num_classes=num_classes, drop_rate=drop_rate)
    else:
        raise ValueError(f"Unknown DenseNet type: {densenet_type}")
