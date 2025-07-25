import os
import yaml
import json
from pathlib import Path
from typing import Dict, Any, List
import shutil

class ConfigManager:
    #Manage configuration files
    
    def __init__(self, config_dir="configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
    
    def list_configs(self) -> List[str]:
      
        configs = []
        for file_path in self.config_dir.glob("*.yaml"):
            configs.append(file_path.stem)
        for file_path in self.config_dir.glob("*.yml"):
            configs.append(file_path.stem)
        for file_path in self.config_dir.glob("*.json"):
            configs.append(file_path.stem)
        return sorted(configs)
    
    def get_config_path(self, config_name: str) -> Path:

        for ext in ['.yaml', '.yml', '.json']:
            config_path = self.config_dir / f"{config_name}{ext}"
            if config_path.exists():
                return config_path
        
        raise FileNotFoundError(f"Configuration '{config_name}' not found in {self.config_dir}")
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        config_path = self.get_config_path(config_name)
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                return yaml.safe_load(f)
            else:
                return json.load(f)
    
    def save_config(self, config: Dict[str, Any], config_name: str, format: str = "yaml"):
        if format.lower() in ['yaml', 'yml']:
            config_path = self.config_dir / f"{config_name}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
        else:
            config_path = self.config_dir / f"{config_name}.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
        
        return config_path
    
    def create_from_args(self, args, config_name: str, format: str = "yaml"):
        config = self._args_to_config(args)
        return self.save_config(config, config_name, format)
    
    def _args_to_config(self, args) -> Dict[str, Any]:
        #Convert argparse namespace to nested config structure
        config = {
            'experiment': {
                'name': getattr(args, 'exp_name', None),
                'description': f"Configuration generated from CLI args",
            },
            'data': {
                'data_path': args.data,
                'base_dir': getattr(args, 'base_dir', None),
                'batch_size': args.batch_size,
                'balanced_sampling': getattr(args, 'balanced_sampling', False),
                'num_workers': getattr(args, 'num_workers', 4),
                'pin_memory': getattr(args, 'pin_memory', False),
            },
            'model': {
                'type': args.model,
            },
            'training': {
                'epochs': args.epochs,
                'early_stopping': args.early_stopping,
                'monitor': args.monitor,
                'initial_freeze': getattr(args, 'initial_freeze', 0),
                'margin': getattr(args, 'margin', 0.0),
                'grad_clip': getattr(args, 'grad_clip', 1.0),
                'warmup_epochs': getattr(args, 'warmup_epochs', 0),
                'warmup_method': getattr(args, 'warmup_method', 'linear'),
                'monitor_gradients': getattr(args, 'monitor_gradients', False),
                'track_parameters': getattr(args, 'track_parameters', False),
            },
            'optimization': {
                'learning_rate': args.lr,
                'weight_decay': args.weight_decay,
                'optimizer': args.optimizer,
                'scheduler': {
                    'type': args.scheduler,
                    'plateau_factor': getattr(args, 'plateau_factor', 0.5),
                    'plateau_patience': getattr(args, 'plateau_patience', 7),
                    'cosine_min_lr': getattr(args, 'cosine_min_lr', 1e-6),
                    'step_size': getattr(args, 'step_size', 30),
                    'step_gamma': getattr(args, 'step_gamma', 0.1),
                }
            },
            'loss': {
                'type': args.loss,
                'class_weights': getattr(args, 'class_weights', False),
                'focal': {
                    'gamma': getattr(args, 'focal_gamma', 2.0),
                    'alpha': getattr(args, 'focal_alpha', 0.25),
                },
                'cb_focal': {
                    'beta': getattr(args, 'beta', 0.9999),
                },
                'asymmetric': {
                    'gamma_pos': getattr(args, 'gamma_pos', 1.0),
                    'gamma_neg': getattr(args, 'gamma_neg', 4.0),
                },
                'margin': {
                    'pos_margin': getattr(args, 'pos_margin', 0.5),
                    'neg_margin': getattr(args, 'neg_margin', 0.5),
                    'pos_weight': getattr(args, 'pos_weight', 2.0),
                }
            },
            'hardware': {
                'gpu': args.gpu,
                'mixed_precision': getattr(args, 'mixed_precision', False),
                'distributed': getattr(args, 'distributed', False),
            }
        }
        
        if args.model == 'cnn':
            config['model']['cnn'] = {
                'layers': getattr(args, 'cnn_layers', 3),
                'base_filters': getattr(args, 'cnn_base_filters', 32),
                'dropout': getattr(args, 'cnn_dropout', 0.5),
            }
        elif args.model == 'vit':
            config['model']['vit'] = {
                'patch_size': getattr(args, 'vit_patch_size', 16),
                'depth': getattr(args, 'vit_depth', 12),
                'heads': getattr(args, 'vit_heads', 12),
                'embed_dim': getattr(args, 'vit_embed_dim', 768),
                'mlp_ratio': getattr(args, 'vit_mlp_ratio', 4.0),
                'dropout': getattr(args, 'vit_dropout', 0.1),
            }
        elif args.model == 'hybrid':
            config['model']['hybrid'] = {
                'feature_dim': getattr(args, 'hybrid_feature_dim', 256),
                'transformer_dim': getattr(args, 'hybrid_transformer_dim', 256),
                'num_heads': getattr(args, 'hybrid_num_heads', 8),
                'depth': getattr(args, 'hybrid_depth', 4),
                'mlp_ratio': getattr(args, 'hybrid_mlp_ratio', 4.0),
                'dropout': getattr(args, 'hybrid_dropout', 0.1),
            }
        
        # Handle augmentation
        if getattr(args, 'no_augmentation', False):
            config['augmentation'] = {'enabled': False}
        elif getattr(args, 'all_augmentation', False):
            config['augmentation'] = {
                'enabled': True,
                'strategy': 'all',
                'probability': getattr(args, 'aug_prob', 0.5),
            }
        elif getattr(args, 'augment', None):
            config['augmentation'] = {
                'enabled': True,
                'strategy': 'custom',
                'probability': getattr(args, 'aug_prob', 0.5),
                'techniques': args.augment,
                'params': {
                    'rotation_angle': getattr(args, 'rotation_angle', 10),
                    'translation_voxels': getattr(args, 'translation_voxels', 5),
                    'scale_range': getattr(args, 'scale_range', [0.9, 1.1]),
                    'noise_percent': getattr(args, 'noise_percent', 0.03),
                    'deformation_sigma': getattr(args, 'deformation_sigma', 3),
                    'deformation_magnitude': getattr(args, 'deformation_magnitude', 10),
                    'bias_coeff': getattr(args, 'bias_coeff', 0.5),
                    'ghosting_max': getattr(args, 'ghosting_max', 3),
                    'gamma_range': getattr(args, 'gamma_range', [0.7, 1.5]),
                    'crop_size': getattr(args, 'crop_size', 80),
                    'cutout_size_range': getattr(args, 'cutout_size_range', [20, 40]),
                }
            }
        else:
            config['augmentation'] = {
                'enabled': True,
                'strategy': 'default',
                'probability': getattr(args, 'aug_prob', 0.5),
            }
        
        config = self._remove_none_values(config)
        
        return config
    
    def _remove_none_values(self, obj):
        if isinstance(obj, dict):
            return {k: self._remove_none_values(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self._remove_none_values(item) for item in obj if item is not None]
        else:
            return obj
    
    def compare_configs(self, config1_name: str, config2_name: str) -> Dict[str, Any]:
        config1 = self.load_config(config1_name)
        config2 = self.load_config(config2_name)
        
        return self._deep_diff(config1, config2)
    
    def _deep_diff(self, dict1, dict2, path=""):
        differences = {}
        
        # Check keys in dict1
        for key in dict1:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict2:
                differences[current_path] = {'status': 'removed', 'value': dict1[key]}
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                nested_diff = self._deep_diff(dict1[key], dict2[key], current_path)
                differences.update(nested_diff)
            elif dict1[key] != dict2[key]:
                differences[current_path] = {
                    'status': 'changed',
                    'old': dict1[key],
                    'new': dict2[key]
                }
        
        # Check keys only in dict2
        for key in dict2:
            current_path = f"{path}.{key}" if path else key
            if key not in dict1:
                differences[current_path] = {'status': 'added', 'value': dict2[key]}
        
        return differences
    
    def validate_config(self, config_name: str) -> List[str]:
        try:
            config = self.load_config(config_name)
        except Exception as e:
            return [f"Failed to load config: {e}"]
        
        issues = []
        
        # Required sections
        required_sections = ['data', 'model', 'training', 'optimization']
        for section in required_sections:
            if section not in config:
                issues.append(f"Missing required section: {section}")
        
        # Validate data section
        if 'data' in config:
            data_config = config['data']
            if 'data_path' not in data_config:
                issues.append("Missing required field: data.data_path")
            if 'batch_size' in data_config and data_config['batch_size'] <= 0:
                issues.append("Invalid batch_size: must be > 0")
        
        # Validate model section
        if 'model' in config:
            model_config = config['model']
            if 'type' not in model_config:
                issues.append("Missing required field: model.type")
            elif model_config['type'] not in ['cnn', 'vit', 'hybrid']:
                issues.append(f"Invalid model type: {model_config['type']}")
        
        # Validate training section
        if 'training' in config:
            training_config = config['training']
            if 'epochs' in training_config and training_config['epochs'] <= 0:
                issues.append("Invalid epochs: must be > 0")
            if 'monitor' in training_config and training_config['monitor'] not in ['auc', 'f1', 'balanced_acc']:
                issues.append(f"Invalid monitor metric: {training_config['monitor']}")
        
        return issues
    
    def create_template(self, model_type: str, template_name: str = None) -> Path:
        if template_name is None:
            template_name = f"{model_type}_template"
        
        templates = {
            'cnn': {
                'experiment': {
                    'name': f"{model_type}_experiment",
                    'description': f"Template configuration for {model_type.upper()} model",
                },
                'data': {
                    'data_path': "dataset_splits.csv",
                    'batch_size': 8,
                    'balanced_sampling': True,
                    'num_workers': 4,
                },
                'model': {
                    'type': 'cnn',
                    'cnn': {
                        'layers': 3,
                        'base_filters': 32,
                        'dropout': 0.5,
                    }
                },
                'training': {
                    'epochs': 50,
                    'early_stopping': 15,
                    'monitor': 'balanced_acc',
                },
                'optimization': {
                    'learning_rate': 0.0001,
                    'weight_decay': 0.0001,
                    'optimizer': 'adamw',
                    'scheduler': {'type': 'plateau'},
                },
                'loss': {
                    'type': 'weighted_ce',
                    'class_weights': True,
                },
                'augmentation': {
                    'enabled': True,
                    'strategy': 'default',
                    'probability': 0.5,
                },
            },
            'vit': {
                'experiment': {
                    'name': f"{model_type}_experiment",
                    'description': f"Template configuration for {model_type.upper()} model",
                },
                'data': {
                    'data_path': "dataset_splits.csv",
                    'batch_size': 4,  # Smaller for ViT
                    'balanced_sampling': True,
                    'num_workers': 4,
                },
                'model': {
                    'type': 'vit',
                    'vit': {
                        'patch_size': 16,
                        'depth': 12,
                        'heads': 12,
                        'embed_dim': 768,
                        'mlp_ratio': 4.0,
                        'dropout': 0.1,
                    }
                },
                'training': {
                    'epochs': 100, 
                    'early_stopping': 20,
                    'monitor': 'auc',
                    'warmup_epochs': 5,
                },
                'optimization': {
                    'learning_rate': 0.00005,
                    'weight_decay': 0.001,
                    'optimizer': 'adamw',
                    'scheduler': {'type': 'cosine'},
                },
                'loss': {
                    'type': 'focal',
                    'focal': {'gamma': 2.0, 'alpha': 0.25},
                },
                'augmentation': {
                    'enabled': True,
                    'strategy': 'all',
                    'probability': 0.6,
                },
            },
            'hybrid': {
                'experiment': {
                    'name': f"{model_type}_experiment",
                    'description': f"Template configuration for {model_type.upper()} model",
                },
                'data': {
                    'data_path': "dataset_splits.csv",
                    'batch_size': 6,  
                    'balanced_sampling': True,
                    'num_workers': 4,
                },
                'model': {
                    'type': 'hybrid',
                    'hybrid': {
                        'feature_dim': 256,
                        'transformer_dim': 256,
                        'num_heads': 8,
                        'depth': 4,
                        'mlp_ratio': 4.0,
                        'dropout': 0.1,
                    }
                },
                'training': {
                    'epochs': 75,
                    'early_stopping': 15,
                    'monitor': 'balanced_acc',
                    'warmup_epochs': 3,
                },
                'optimization': {
                    'learning_rate': 0.0001,
                    'weight_decay': 0.0005,
                    'optimizer': 'adamw',
                    'scheduler': {'type': 'plateau'},
                },
                'loss': {
                    'type': 'cb_focal',
                    'cb_focal': {'beta': 0.9999},
                },
                'augmentation': {
                    'enabled': True,
                    'strategy': 'custom',
                    'probability': 0.5,
                    'techniques': ['rotation', 'translation', 'flip', 'noise'],
                },
            }
        }
        
        if model_type not in templates:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return self.save_config(templates[model_type], template_name)


def create_config_cli():

    import argparse
    
    parser = argparse.ArgumentParser(description="MRI Classification Config Manager")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List configs
    subparsers.add_parser('list', help='List available configurations')
    
    # Create template
    template_parser = subparsers.add_parser('template', help='Create a template configuration')
    template_parser.add_argument('model_type', choices=['cnn', 'vit', 'hybrid'],
                                help='Model type for template')
    template_parser.add_argument('--name', type=str, help='Template name')
    
    # Validate config
    validate_parser = subparsers.add_parser('validate', help='Validate a configuration')
    validate_parser.add_argument('config_name', help='Name of config to validate')
    
    # Compare configs
    compare_parser = subparsers.add_parser('compare', help='Compare two configurations')
    compare_parser.add_argument('config1', help='First configuration')
    compare_parser.add_argument('config2', help='Second configuration')
    
    # Convert args to config
    convert_parser = subparsers.add_parser('convert', help='Convert CLI args to config file')
    convert_parser.add_argument('--output', '-o', required=True, help='Output config name')
    convert_parser.add_argument('args', nargs=argparse.REMAINDER, help='CLI arguments to convert')
    
    args = parser.parse_args()
    manager = ConfigManager()
    
    if args.command == 'list':
        configs = manager.list_configs()
        print("Available configurations:")
        for config in configs:
            print(f"  - {config}")
    
    elif args.command == 'template':
        template_path = manager.create_template(args.model_type, args.name)
        print(f"Template created: {template_path}")
    
    elif args.command == 'validate':
        issues = manager.validate_config(args.config_name)
        if issues:
            print(f"Validation issues for '{args.config_name}':")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print(f"Configuration '{args.config_name}' is valid!")
    
    elif args.command == 'compare':
        differences = manager.compare_configs(args.config1, args.config2)
        if differences:
            print(f"Differences between '{args.config1}' and '{args.config2}':")
            for path, diff in differences.items():
                print(f"  {path}: {diff}")
        else:
            print("Configurations are identical!")
    
    elif args.command == 'convert':
        # This would require importing the full CLI parser
        print("Convert functionality requires integration with main CLI parser")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    create_config_cli()
