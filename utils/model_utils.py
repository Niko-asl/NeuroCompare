import torch
import numpy as np
import logging

logger = logging.getLogger(__name__)

def track_parameters(model, step, model_type='cnn', log_frequency=100):
    """
    Track and print parameter statistics during training.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to track
    step : int
        Current training step
    model_type : str
        Type of model (cnn, vit, hybrid)
    log_frequency : int
        How often to log parameter statistics
    """
    if step % log_frequency != 0:
        return  # Only log at specified frequency
    
    if model_type == 'cnn':
        _track_cnn_parameters(model, step)
    elif model_type == 'vit':
        _track_vit_parameters(model, step)
    elif model_type == 'hybrid':
        _track_hybrid_parameters(model, step)
    else:
        logger.warning(f"Unknown model type for parameter tracking: {model_type}")

def _track_cnn_parameters(model, step):
    """Track CNN model parameters."""
    # Track the last layer's weights and biases
    if hasattr(model, 'fc3'):
        fc_layer = model.fc3
    elif hasattr(model, 'fc'):
        fc_layer = model.fc
    else:
        logger.warning("Could not find final FC layer for parameter tracking")
        return
    
    w_norm = torch.norm(fc_layer.weight).item()
    w_mean = fc_layer.weight.mean().item()
    w_std = fc_layer.weight.std().item()
    
    output = f"Step {step}: CNN final layer - Weight stats: norm={w_norm:.4f}, mean={w_mean:.4f}, std={w_std:.4f}"
    
    if fc_layer.bias is not None:
        b_norm = torch.norm(fc_layer.bias).item()
        b_mean = fc_layer.bias.mean().item()
        if fc_layer.bias.shape[0] >= 2:
            b_diff = fc_layer.bias[1].item() - fc_layer.bias[0].item()
            output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}, class_diff={b_diff:.4f}"
        else:
            output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}"
    else:
        output += ", No bias"
        
    logger.info(output)
    
    # Also check first conv layer
    if hasattr(model, 'conv1'):
        w_norm = torch.norm(model.conv1.weight).item()
        w_mean = model.conv1.weight.mean().item()
        w_std = model.conv1.weight.std().item()
        logger.info(f"Step {step}: CNN first conv - Weight stats: norm={w_norm:.4f}, mean={w_mean:.4f}, std={w_std:.4f}")

def _track_vit_parameters(model, step):
    """Track Vision Transformer model parameters."""
    # Track the classifier head
    if hasattr(model, 'head'):
        head = model.head
        w_norm = torch.norm(head.weight).item()
        w_mean = head.weight.mean().item()
        w_std = head.weight.std().item()
        
        output = f"Step {step}: ViT head - Weight stats: norm={w_norm:.4f}, mean={w_mean:.4f}, std={w_std:.4f}"
        
        if head.bias is not None:
            b_norm = torch.norm(head.bias).item()
            b_mean = head.bias.mean().item()
            if head.bias.shape[0] >= 2:
                b_diff = head.bias[1].item() - head.bias[0].item()
                output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}, class_diff={b_diff:.4f}"
            else:
                output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}"
        
        logger.info(output)
    
    # Track attention weights in first transformer block
    if hasattr(model, 'blocks') and len(model.blocks) > 0:
        attn = model.blocks[0].attn
        if hasattr(attn, 'qkv'):
            qkv_norm = torch.norm(attn.qkv.weight).item()
            qkv_mean = attn.qkv.weight.mean().item()
            qkv_std = attn.qkv.weight.std().item()
            logger.info(f"Step {step}: ViT first attn QKV - Weight stats: norm={qkv_norm:.4f}, mean={qkv_mean:.4f}, std={qkv_std:.4f}")

def _track_hybrid_parameters(model, step):
    """Track Hybrid CNN-Transformer model parameters."""
    # Track the classifier
    if hasattr(model, 'classifier'):
        classifier = model.classifier
        w_norm = torch.norm(classifier.weight).item()
        w_mean = classifier.weight.mean().item()
        w_std = classifier.weight.std().item()
        
        output = f"Step {step}: Hybrid classifier - Weight stats: norm={w_norm:.4f}, mean={w_mean:.4f}, std={w_std:.4f}"
        
        if classifier.bias is not None:
            b_norm = torch.norm(classifier.bias).item()
            b_mean = classifier.bias.mean().item()
            if classifier.bias.shape[0] >= 2:
                b_diff = classifier.bias[1].item() - classifier.bias[0].item()
                output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}, class_diff={b_diff:.4f}"
            else:
                output += f", Bias: norm={b_norm:.4f}, mean={b_mean:.4f}"
        
        logger.info(output)
    
    # Track feature extractor
    if hasattr(model, 'feature_extractor') and hasattr(model.feature_extractor, 'conv1'):
        fe_conv = model.feature_extractor.conv1
        w_norm = torch.norm(fe_conv.weight).item()
        w_mean = fe_conv.weight.mean().item()
        w_std = fe_conv.weight.std().item()
        logger.info(f"Step {step}: Hybrid CNN first conv - Weight stats: norm={w_norm:.4f}, mean={w_mean:.4f}, std={w_std:.4f}")

def check_output_values(outputs, name="layer"):
    """
    Check for problematic values in the outputs of a layer.
    
    Parameters:
    -----------
    outputs : torch.Tensor
        Output tensor from a layer
    name : str
        Name of the layer for logging
        
    Returns:
    --------
    bool
        True if values are acceptable, False otherwise
    """
    has_nan = torch.isnan(outputs).any()
    has_inf = torch.isinf(outputs).any()
    
    if has_nan or has_inf:
        logger.warning(f"{name} has {'NaN' if has_nan else ''} {'Inf' if has_inf else ''} values")
        return False
    
    min_val = outputs.min().item()
    max_val = outputs.max().item()
    mean_val = outputs.mean().item()
    std_val = outputs.std().item()
    
    if min_val == max_val:
        logger.warning(f"{name} has constant value: {min_val}")
        return False
    
    if std_val < 1e-6:
        logger.warning(f"{name} has small variance: std={std_val}")
        return False
    
    return True
