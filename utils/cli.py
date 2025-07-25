import argparse
import os
import yaml
import json
from pathlib import Path
from types import SimpleNamespace

def create_parser():
    parser = argparse.ArgumentParser(
        description="Argument Parser for the end to end Deep Learning MRI framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    add_config_arguments(parser)
    add_data_arguments(parser)
    add_model_arguments(parser) 
    add_training_arguments(parser)
    add_augmentation_arguments(parser)
    add_optimization_arguments(parser)
    add_loss_arguments(parser)
    add_evaluation_arguments(parser)
    add_logging_arguments(parser)
    add_hardware_arguments(parser)
    
    return parser

def add_config_arguments(parser):

    config_group = parser.add_argument_group('Configuration File')
    
    config_group.add_argument('--config', type=str, default=None,
                             help='Path to YAML/JSON configuration file')
    config_group.add_argument('--save_config', type=str, default=None,
                             help='Save current configuration to specified file')
    config_group.add_argument('--print_config', action='store_true',
                             help='Print the current configuration and exit')

def load_config(config_path):

    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            try:
                import yaml
                config = yaml.safe_load(f)
            except ImportError:
                raise ImportError("PyYAML not installed. Install with: pip install PyYAML")
        elif config_path.suffix.lower() == '.json':
            config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return config

def flatten_config(config, prefix=''):

    flattened = {}
    
    for key, value in config.items():
        if prefix:
            full_key = f"{prefix}_{key}"
        else:
            full_key = key
            
        if isinstance(value, dict):
            # Recursively flatten nested dicts
            flattened.update(flatten_config(value, full_key))
        else:
            flattened[full_key] = value
    
    return flattened

def config_to_args(config):

    # Flatten the nested config
    flat_config = flatten_config(config)
    
    # Map config keys to CLI argument names
    key_mapping = {

        'data_data': 'data',           
        'data_data_path': 'data',      
        'data_path': 'data',           
        'data_base_dir': 'base_dir',
        'data_batch_size': 'batch_size',
        'data_balanced_sampling': 'balanced_sampling',
        'data_num_workers': 'num_workers',
        'data_pin_memory': 'pin_memory',
        
        # Model
        'model_type': 'model',
        'model_cnn_layers': 'cnn_layers',
        'model_cnn_base_filters': 'cnn_base_filters',
        'model_cnn_dropout': 'cnn_dropout',
        'model_vit_patch_size': 'vit_patch_size',
        'model_vit_depth': 'vit_depth',
        'model_vit_heads': 'vit_heads',
        'model_vit_embed_dim': 'vit_embed_dim',
        'model_vit_mlp_ratio': 'vit_mlp_ratio',
        'model_vit_dropout': 'vit_dropout',
        'model_hybrid_feature_dim': 'hybrid_feature_dim',
        'model_hybrid_transformer_dim': 'hybrid_transformer_dim',
        'model_hybrid_num_heads': 'hybrid_num_heads',
        'model_hybrid_depth': 'hybrid_depth',
        'model_hybrid_mlp_ratio': 'hybrid_mlp_ratio',
        'model_hybrid_dropout': 'hybrid_dropout',
        
        # Training
        'training_epochs': 'epochs',
        'training_early_stopping': 'early_stopping',
        'training_monitor': 'monitor',
        'training_initial_freeze': 'initial_freeze',
        'training_margin': 'margin',
        'training_grad_clip': 'grad_clip',
        'training_warmup_epochs': 'warmup_epochs',
        'training_warmup_method': 'warmup_method',
        'training_monitor_gradients': 'monitor_gradients',
        'training_track_parameters': 'track_parameters',
        
        # Augmentation
        'augmentation_enabled': 'augmentation_enabled',
        'augmentation_strategy': 'augmentation_strategy',
        'augmentation_probability': 'aug_prob',
        'augmentation_techniques': 'augment',
        'augmentation_params_rotation_angle': 'rotation_angle',
        'augmentation_params_translation_voxels': 'translation_voxels',
        'augmentation_params_scale_range': 'scale_range',
        'augmentation_params_noise_percent': 'noise_percent',
        'augmentation_params_deformation_sigma': 'deformation_sigma',
        'augmentation_params_deformation_magnitude': 'deformation_magnitude',
        'augmentation_params_bias_coeff': 'bias_coeff',
        'augmentation_params_ghosting_max': 'ghosting_max',
        'augmentation_params_gamma_range': 'gamma_range',
        'augmentation_params_crop_size': 'crop_size',
        'augmentation_params_cutout_size_range': 'cutout_size_range',
        
        # Optimization
        'optimization_learning_rate': 'lr',
        'optimization_weight_decay': 'weight_decay',
        'optimization_optimizer': 'optimizer',
        'optimization_scheduler_type': 'scheduler',
        'optimization_scheduler_plateau_factor': 'plateau_factor',
        'optimization_scheduler_plateau_patience': 'plateau_patience',
        'optimization_scheduler_cosine_min_lr': 'cosine_min_lr',
        'optimization_scheduler_step_size': 'step_size',
        'optimization_scheduler_step_gamma': 'step_gamma',
        
        # Loss
        'loss_type': 'loss',
        'loss_class_weights': 'class_weights',
        'loss_focal_gamma': 'focal_gamma',
        'loss_focal_alpha': 'focal_alpha',
        'loss_cb_focal_beta': 'beta',
        'loss_asymmetric_gamma_pos': 'gamma_pos',
        'loss_asymmetric_gamma_neg': 'gamma_neg',
        'loss_margin_pos_margin': 'pos_margin',
        'loss_margin_neg_margin': 'neg_margin',
        'loss_margin_pos_weight': 'pos_weight',
        
        # Evaluation
        'evaluation_eval_batch_size': 'eval_batch_size',
        'evaluation_decision_margin': 'decision_margin',
        'evaluation_save_predictions': 'save_predictions',
        'evaluation_export_features': 'export_features',
        'evaluation_test_time_augmentation': 'test_time_augmentation',
        'evaluation_tta_samples': 'tta_samples',
        
        # Logging - Fixed to match your YAML structure
        'logging_output_dir': 'output',
        'logging_save_freq': 'save_freq',
        'logging_debug': 'debug',
        'logging_verbose': 'verbose',
        'logging_tensorboard': 'tensorboard',
        'logging_wandb': 'wandb',
        'logging_wandb_project': 'wandb_project',
        'logging_wandb_entity': 'wandb_entity',
        
        # Hardware
        'hardware_gpu': 'gpu',
        'hardware_mixed_precision': 'mixed_precision',
        'hardware_distributed': 'distributed',
        'hardware_world_size': 'world_size',
        'hardware_dist_url': 'dist_url',
        
        # Experiment metadata
        'experiment_name': 'exp_name',
    }
    
    # Convert to args format
    args_dict = {}
    for config_key, value in flat_config.items():
        if config_key in key_mapping:
            arg_name = key_mapping[config_key]
            args_dict[arg_name] = value
    
    return args_dict

def merge_config_and_cli(config_args, cli_args):
    """Merge config file args with CLI args (CLI takes precedence)."""
    # Start with config defaults
    merged = config_args.copy()
    
    # Override with any CLI arguments that were explicitly set
    cli_dict = vars(cli_args)
    
    # Create parser to get defaults
    parser = create_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    
    for key, value in cli_dict.items():
        # Only override if CLI argument was explicitly set
        if key in defaults and value != defaults[key]:
            merged[key] = value
        elif key not in defaults:
            merged[key] = value
    
    return merged

def handle_augmentation_config(args):
   
    if hasattr(args, 'augmentation_enabled') and not args.augmentation_enabled:
        args.no_augmentation = True
    elif hasattr(args, 'augmentation_strategy'):
        if args.augmentation_strategy == 'none':
            args.no_augmentation = True
        elif args.augmentation_strategy == 'all':
            args.all_augmentation = True
        elif args.augmentation_strategy == 'custom' and hasattr(args, 'augment'):
            
            pass
        else: 
            # Let the training script use its defaults
            pass

def parse_args(args=None):
    
    parser = create_parser()
    cli_args = parser.parse_args(args)
    
    # Handle config file
    if cli_args.config:
        print(f"Loading configuration from: {cli_args.config}")
        config_dict = load_config(cli_args.config)
        config_args = config_to_args(config_dict)
        
        # Merge config and CLI args (CLI takes precedence)
        merged_args = merge_config_and_cli(config_args, cli_args)
        
        # Convert back to namespace
        for key, value in merged_args.items():
            setattr(cli_args, key, value)
    
    # Handle augmentation configuration
    handle_augmentation_config(cli_args)
    
    if not getattr(cli_args, 'data', None):
        parser.error("--data is required (either via CLI or config file)")
    
    default_attrs = {
        'save_freq': 0,
        'debug': False,
        'verbose': False,
        'tensorboard': False,
        'wandb': False,
        'wandb_project': 'alzheimers-mri',
        'wandb_entity': None,
        'mixed_precision': False,
        'distributed': False,
        'world_size': 1,
        'dist_url': 'env://',
        'monitor_gradients': False,
        'track_parameters': False,
        'num_workers': 4,
        'pin_memory': False,
        'balanced_sampling': False,
        'base_dir': None,
        'initial_freeze': 0,
        'margin': 0.0,
        'warmup_epochs': 0,
        'warmup_method': 'linear',
        'plateau_factor': 0.5,
        'plateau_patience': 7,
        'cosine_min_lr': 1e-6,
        'step_size': 30,
        'step_gamma': 0.1,
        'class_weights': False,
        'focal_gamma': 2.0,
        'focal_alpha': 0.25,
        'beta': 0.9999,
        'gamma_pos': 1.0,
        'gamma_neg': 4.0,
        'pos_margin': 0.5,
        'neg_margin': 0.5,
        'pos_weight': 2.0,
        'eval_batch_size': 4,
        'decision_margin': 0.0,
        'save_predictions': False,
        'export_features': False,
        'test_time_augmentation': False,
        'tta_samples': 5,
        'rotation_angle': 10,
        'translation_voxels': 5,
        'scale_range': [0.9, 1.1],
        'noise_percent': 0.03,
        'deformation_sigma': 3,
        'deformation_magnitude': 10,
        'bias_coeff': 0.5,
        'ghosting_max': 3,
        'gamma_range': [0.7, 1.5],
        'crop_size': 80,
        'cutout_size_range': [20, 40],
    }
    
    for attr, default_value in default_attrs.items():
        if not hasattr(cli_args, attr):
            setattr(cli_args, attr, default_value)
    
    if cli_args.print_config:
        print("\nCurrent Configuration:")
        print(json.dumps(vars(cli_args), indent=2, default=str))
        exit(0)

    if getattr(cli_args, 'save_config', None):
        save_config(cli_args, cli_args.save_config)
        print(f"Configuration saved to: {cli_args.save_config}")
    
    # Create directories and handle experiment naming (existing logic)
    os.makedirs(cli_args.output, exist_ok=True)
    
    if getattr(cli_args, 'exp_name', None) is None:
        import time
        timestamp = time.strftime("%d%m%Y-%H%M%S")
        cli_args.exp_name = f"{cli_args.model}_{timestamp}"
    
    cli_args.exp_dir = os.path.join(cli_args.output, cli_args.exp_name)
    os.makedirs(cli_args.exp_dir, exist_ok=True)
    
    cli_args.ckpt_dir = os.path.join(cli_args.exp_dir, 'checkpoints')
    cli_args.log_dir = os.path.join(cli_args.exp_dir, 'logs')
    
    os.makedirs(cli_args.ckpt_dir, exist_ok=True)
    os.makedirs(cli_args.log_dir, exist_ok=True)
    
    return cli_args

# FOr reverse args -> config file 
def save_config(args, config_path):
    config_path = Path(config_path)
    

    config = {
        'experiment': {
            'name': getattr(args, 'exp_name', None),
        },
        'data': {
            'data': args.data,  # Match your YAML structure
            'base_dir': getattr(args, 'base_dir', None),
            'batch_size': args.batch_size,
            'balanced_sampling': getattr(args, 'balanced_sampling', False),
            'num_workers': getattr(args, 'num_workers', 4),
            'pin_memory': getattr(args, 'pin_memory', False),
        },
        'model': {
            'type': args.model,
            'cnn': {
                'layers': getattr(args, 'cnn_layers', 3),
                'base_filters': getattr(args, 'cnn_base_filters', 32),
                'dropout': getattr(args, 'cnn_dropout', 0.5),
            },
            'vit': {
                'patch_size': getattr(args, 'vit_patch_size', 16),
                'depth': getattr(args, 'vit_depth', 12),
                'heads': getattr(args, 'vit_heads', 12),
                'embed_dim': getattr(args, 'vit_embed_dim', 768),
                'mlp_ratio': getattr(args, 'vit_mlp_ratio', 4.0),
                'dropout': getattr(args, 'vit_dropout', 0.1),
            },
            'hybrid': {
                'feature_dim': getattr(args, 'hybrid_feature_dim', 256),
                'transformer_dim': getattr(args, 'hybrid_transformer_dim', 256),
                'num_heads': getattr(args, 'hybrid_num_heads', 8),
                'depth': getattr(args, 'hybrid_depth', 4),
                'mlp_ratio': getattr(args, 'hybrid_mlp_ratio', 4.0),
                'dropout': getattr(args, 'hybrid_dropout', 0.1),
            },
        },
        ######## Test this part then come back to finish it #############
    }
    

    def remove_none(obj):
        if isinstance(obj, dict):
            return {k: remove_none(v) for k, v in obj.items() if v is not None}
        else:
            return obj
    
    config = remove_none(config)
    
    # Save to file
    with open(config_path, 'w') as f:
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            json.dump(config, f, indent=2)


def add_data_arguments(parser):
    """Add data configuration arguments."""
    data_group = parser.add_argument_group('Data Configuration')
    
    data_group.add_argument('--data', type=str, required=False,
                           help='Path to dataset_splits.csv file')
    data_group.add_argument('--base_dir', type=str, default=None,
                           help='Base directory for resolving file paths in CSV')
    data_group.add_argument('--batch_size', type=int, default=4,
                           help='Batch size for training and evaluation')
    data_group.add_argument('--balanced_sampling', action='store_true',
                           help='Use balanced sampling for training batches')
######### Future model args go here ##############
def add_model_arguments(parser):
    """Add model architecture arguments."""
    model_group = parser.add_argument_group('Model Architecture')
    
    model_group.add_argument('--model', type=str, default='cnn', 
                            choices=['cnn', 'vit', 'hybrid', 'densenet'],
                            help='Model architecture to train')
    
    cnn_group = parser.add_argument_group('CNN Architecture Options')
    cnn_group.add_argument('--cnn_layers', type=int, default=3, choices=range(2, 7),
                          help='Number of residual blocks in CNN (2-6)')
    cnn_group.add_argument('--cnn_base_filters', type=int, default=32,
                          help='Number of base filters in CNN')
    cnn_group.add_argument('--cnn_dropout', type=float, default=0.5,
                          help='Dropout rate for CNN')
                          
    # DenseNet Parameters
    densenet_group = parser.add_argument_group('DenseNet Architecture Options')
    densenet_group.add_argument('--densenet_type', type=str, default='264',
                               choices=['121', '169', '201', '264'],
                               help='DenseNet architecture type')
    densenet_group.add_argument('--densenet_growth_rate', type=int, default=32,
                               help='Growth rate for DenseNet')
    densenet_group.add_argument('--densenet_dropout', type=float, default=0.0,
                               help='Dropout rate for DenseNet')
    densenet_group.add_argument('--densenet_bn_size', type=int, default=4,
                               help='Bottleneck size multiplier for DenseNet')
              
    # Vision Transformer architecture params
    vit_group = parser.add_argument_group('Vision Transformer Architecture Options')
    vit_group.add_argument('--vit_patch_size', type=int, default=16,
                          help='Patch size for Vision Transformer')
    vit_group.add_argument('--vit_depth', type=int, default=12,
                          help='Number of transformer blocks')
    vit_group.add_argument('--vit_heads', type=int, default=12,
                          help='Number of attention heads')
    vit_group.add_argument('--vit_embed_dim', type=int, default=768,
                          help='Embedding dimension')
    vit_group.add_argument('--vit_mlp_ratio', type=float, default=4.0,
                          help='MLP ratio for transformer blocks')
    vit_group.add_argument('--vit_dropout', type=float, default=0.1,
                          help='Dropout rate for ViT')

    # Hybrid model params
    hybrid_group = parser.add_argument_group('Hybrid Model Architecture Options')
    hybrid_group.add_argument('--hybrid_feature_dim', type=int, default=256,
                             help='Feature dimension for hybrid model')
    hybrid_group.add_argument('--hybrid_transformer_dim', type=int, default=256,
                             help='Transformer dimension for hybrid model')
    hybrid_group.add_argument('--hybrid_num_heads', type=int, default=8,
                             help='Number of attention heads for hybrid model')
    hybrid_group.add_argument('--hybrid_depth', type=int, default=4,
                             help='Number of transformer blocks for hybrid model')
    hybrid_group.add_argument('--hybrid_mlp_ratio', type=float, default=4.0,
                             help='MLP ratio for hybrid model')
    hybrid_group.add_argument('--hybrid_dropout', type=float, default=0.1,
                             help='Dropout rate for hybrid model')

def add_training_arguments(parser):
    """Add training configuration arguments."""
    train_group = parser.add_argument_group('Training Configuration')
    
    train_group.add_argument('--epochs', type=int, default=50,
                            help='Number of epochs to train')
    train_group.add_argument('--early_stopping', type=int, default=15,
                            help='Number of epochs to wait for improvement before early stopping')
    train_group.add_argument('--monitor', type=str, default='balanced_acc',
                            choices=['auc', 'f1', 'balanced_acc'],
                            help='Metric to monitor for early stopping and model selection')
    train_group.add_argument('--initial_freeze', type=int, default=0,
                            help='Number of epochs to freeze feature extractor initially')
    train_group.add_argument('--margin', type=float, default=0.0,
                            help='Decision margin for predictions (0.5 ± margin)')
    train_group.add_argument('--grad_clip', type=float, default=1.0,
                            help='Max norm for gradient clipping (0 to disable)')
    train_group.add_argument('--warmup_epochs', type=int, default=0,
                            help='Number of epochs for learning rate warmup')
    train_group.add_argument('--warmup_method', type=str, default='linear',
                            choices=['linear', 'exponential'],
                            help='Warmup scale method (linear or exponential)')
    train_group.add_argument('--monitor_gradients', action='store_true',
                            help='Enable gradient monitoring during training')
    train_group.add_argument('--track_parameters', action='store_true',
                            help='Track model parameters during training')

def add_augmentation_arguments(parser):
    """Add data augmentation arguments."""
    aug_group = parser.add_argument_group('Data Augmentation')
    
    # Control overall augmentation strategy
    aug_control = aug_group.add_mutually_exclusive_group()
    aug_control.add_argument('--no_augmentation', action='store_true',
                           help='Disable all data augmentation')
    aug_control.add_argument('--all_augmentation', action='store_true',
                           help='Enable all data augmentations')
    aug_control.add_argument('--augment', type=str, nargs='+', choices=[
                            'rotation', 'translation', 'scale', 'flip', 
                            'deformation', 'noise', 'bias', 'ghosting',
                            'gamma', 'contrast', 'crop', 'cutout'],
                           help='List of specific augmentations to apply')
    
    aug_group.add_argument('--aug_prob', type=float, default=0.5,
                         help='Probability of applying each augmentation')
    
    # Augmentation parameters
    aug_group.add_argument('--rotation_angle', type=float, default=10,
                         help='Maximum rotation angle in degrees')
    aug_group.add_argument('--translation_voxels', type=int, default=5,
                         help='Maximum translation in voxels')
    aug_group.add_argument('--scale_range', type=float, nargs=2, default=[0.9, 1.1],
                         help='Range for random scaling (min max)')
    aug_group.add_argument('--noise_percent', type=float, default=0.03,
                         help='Noise standard deviation as percent of max intensity')
    aug_group.add_argument('--deformation_sigma', type=float, default=3,
                         help='Elastic deformation sigma parameter')
    aug_group.add_argument('--deformation_magnitude', type=float, default=10,
                         help='Elastic deformation magnitude parameter')
    aug_group.add_argument('--bias_coeff', type=float, default=0.5,
                         help='Bias field coefficient')
    aug_group.add_argument('--ghosting_max', type=int, default=3,
                         help='Maximum number of ghost artifacts')
    aug_group.add_argument('--gamma_range', type=float, nargs=2, default=[0.7, 1.5],
                         help='Range for random gamma correction (min max)')
    aug_group.add_argument('--crop_size', type=int, default=80,
                         help='Size of random crop in voxels')
    aug_group.add_argument('--cutout_size_range', type=int, nargs=2, default=[20, 40],
                         help='Size range for cutout in voxels (min max)')

def add_optimization_arguments(parser):
    """Add optimization configuration arguments."""
    optim_group = parser.add_argument_group('Optimization Configuration')
    
    optim_group.add_argument('--lr', type=float, default=0.0001,
                           help='Learning rate')
    optim_group.add_argument('--weight_decay', type=float, default=1e-4,
                           help='Weight decay (L2 regularization)')
    optim_group.add_argument('--optimizer', type=str, default='adamw',
                           choices=['adam', 'adamw', 'sgd'],
                           help='Optimizer type')
    optim_group.add_argument('--scheduler', type=str, default='plateau',
                           choices=['plateau', 'cosine', 'step', 'none'],
                           help='Learning rate scheduler')
    optim_group.add_argument('--plateau_factor', type=float, default=0.5,
                           help='Factor to reduce LR by on plateau')
    optim_group.add_argument('--plateau_patience', type=int, default=7,
                           help='Patience for plateau scheduler')
    optim_group.add_argument('--cosine_min_lr', type=float, default=1e-6,
                           help='Minimum LR for cosine scheduler')
    optim_group.add_argument('--step_size', type=int, default=30,
                           help='Step size for StepLR scheduler')
    optim_group.add_argument('--step_gamma', type=float, default=0.1,
                           help='Gamma for StepLR scheduler')

def add_loss_arguments(parser):
    """Add loss function configuration arguments."""
    loss_group = parser.add_argument_group('Loss Function Configuration')
    
    loss_group.add_argument('--loss', type=str, default='weighted_ce',
                         choices=['ce', 'weighted_ce', 'focal', 'cb_focal', 
                                  'asymmetric', 'margin', 'bce_logits'],
                         help='Loss function type')
    loss_group.add_argument('--class_weights', action='store_true',
                         help='Use class weights for loss function to handle imbalance')
    loss_group.add_argument('--focal_gamma', type=float, default=2.0,
                         help='Gamma parameter for focal loss')
    loss_group.add_argument('--focal_alpha', type=float, default=0.25,
                         help='Alpha parameter for focal loss')
    loss_group.add_argument('--beta', type=float, default=0.9999,
                         help='Beta parameter for class-balanced weighting')
    loss_group.add_argument('--gamma_pos', type=float, default=1.0,
                         help='Focusing parameter for positive examples in asymmetric loss')
    loss_group.add_argument('--gamma_neg', type=float, default=4.0,
                         help='Focusing parameter for negative examples in asymmetric loss')
    loss_group.add_argument('--pos_margin', type=float, default=0.5,
                         help='Margin for positive class predictions')
    loss_group.add_argument('--neg_margin', type=float, default=0.5,
                         help='Margin for negative class predictions')
    loss_group.add_argument('--pos_weight', type=float, default=2.0,
                         help='Weight for positive class predictions')

def add_evaluation_arguments(parser):
    """Add evaluation configuration arguments."""
    eval_group = parser.add_argument_group('Evaluation Configuration')
    
    eval_group.add_argument('--eval_batch_size', type=int, default=4,
                          help='Batch size for evaluation')
    eval_group.add_argument('--decision_margin', type=float, default=0.0,
                          help='Decision margin for evaluation predictions')
    eval_group.add_argument('--save_predictions', action='store_true',
                          help='Save individual predictions for analysis')
    eval_group.add_argument('--export_features', action='store_true',
                          help='Export extracted features for visualization')
    eval_group.add_argument('--test_time_augmentation', action='store_true',
                          help='Use test-time augmentation')
    eval_group.add_argument('--tta_samples', type=int, default=5,
                          help='Number of augmented samples for test-time augmentation')

def add_logging_arguments(parser):
    """Add logging and output arguments."""
    log_group = parser.add_argument_group('Logging and Output')
    
    log_group.add_argument('--output', type=str, default='output',
                         help='Output directory')
    log_group.add_argument('--exp_name', type=str, default=None,
                         help='Experiment name (defaults to {model_type}_{timestamp})')
    log_group.add_argument('--save_freq', type=int, default=0,
                         help='Save checkpoint every N epochs (0 for best only)')
    log_group.add_argument('--debug', action='store_true',
                         help='Enable debug mode with extra logging')
    log_group.add_argument('--verbose', action='store_true',
                         help='Enable verbose output')
    log_group.add_argument('--tensorboard', action='store_true',
                         help='Enable TensorBoard logging')
    log_group.add_argument('--wandb', action='store_true',
                         help='Enable Weights & Biases logging')
    log_group.add_argument('--wandb_project', type=str, default='alzheimers-mri',
                         help='Weights & Biases project name')
    log_group.add_argument('--wandb_entity', type=str, default=None,
                         help='Weights & Biases entity name')

def add_hardware_arguments(parser):
    """Add hardware configuration arguments."""
    hw_group = parser.add_argument_group('Hardware Configuration')
    
    hw_group.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    hw_group.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    hw_group.add_argument('--pin_memory', action='store_true',
                        help='Use pin memory for faster data transfer to GPU')
    hw_group.add_argument('--mixed_precision', action='store_true',
                        help='Use mixed precision training (FP16)')
    hw_group.add_argument('--distributed', action='store_true',
                        help='Enable distributed training')
    hw_group.add_argument('--world_size', type=int, default=1,
                        help='Number of processes for distributed training')
    hw_group.add_argument('--dist_url', type=str, default='env://',
                        help='URL for distributed training')

if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(vars(args), indent=2, default=str))
