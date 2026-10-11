#!/usr/bin/env python3
"""
Simplified External Dataset Model Evaluation Script
Auto-detects model architecture from experiment folder structure
"""

import sys
import os
import argparse
import logging
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.dataset import create_data_loaders
from models.cnn_factory import create_cnn_model
from models.vit_factory import create_vit_model
from models.hybrid_factory import create_hybrid_model
from models.minit_factory import create_minit_model_factory
from models.densenet_3d import create_densenet_model

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import unpack_batch

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained models on external dataset with auto-detection")
    
    # Only essential arguments
    parser.add_argument('--model_folder', type=str, required=True,
                       help='Path to model experiment folder containing JSON and checkpoint')
    parser.add_argument('--external_csv', type=str, required=True,
                       help='Path to external_dataset.csv file')
    parser.add_argument('--base_dir', type=str, default=None,
                       help='Base directory for resolving file paths in CSV')
    
    # Optional evaluation settings
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for evaluation')
    parser.add_argument('--output_dir', type=str, default='external_evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save individual predictions to CSV')
    
    return parser.parse_args()

def discover_model_files(model_folder):
    """Auto-discover JSON config and checkpoint files in model folder"""
    
    model_folder = Path(model_folder)
    
    if not model_folder.exists():
        raise FileNotFoundError(f"Model folder not found: {model_folder}")
    
    # Find JSON config file
    json_files = list(model_folder.glob("*final_results.json"))
    if not json_files:
        raise FileNotFoundError(f"No *final_results.json found in {model_folder}")
    elif len(json_files) > 1:
        print(f"Multiple JSON files found: {json_files}")
        print(f"Using: {json_files[0]}")
    
    json_file = json_files[0]
    
    # Find checkpoint file
    checkpoint_folder = model_folder / "checkpoints"
    if not checkpoint_folder.exists():
        raise FileNotFoundError(f"Checkpoints folder not found: {checkpoint_folder}")
    
    checkpoint_files = list(checkpoint_folder.glob("*_best.pth"))
    if not checkpoint_files:
        # Try alternative patterns
        checkpoint_files = list(checkpoint_folder.glob("*.pth"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {checkpoint_folder}")
    
    if len(checkpoint_files) > 1:
        print(f"Multiple checkpoint files found: {checkpoint_files}")
        # Prefer *_best.pth files
        best_files = [f for f in checkpoint_files if 'best' in f.name]
        if best_files:
            checkpoint_file = best_files[0]
        else:
            checkpoint_file = checkpoint_files[0]
        print(f"Using: {checkpoint_file}")
    else:
        checkpoint_file = checkpoint_files[0]
    
    return json_file, checkpoint_file

def load_config_from_json(json_file):
    """Load training configuration from JSON file"""
    
    with open(json_file, 'r') as f:
        config = json.load(f)
    
    if 'training_config' not in config:
        raise ValueError(f"No training_config found in {json_file}")
    
    training_config = config['training_config']
    
    # Extract model type
    model_type = training_config.get('model')
    if not model_type:
        raise ValueError(f"No model type found in training config")
    
    return model_type, training_config, config

def create_model_from_config(model_type, config):
    """Create model from training configuration"""
    
    # Handle input shape
    if config.get('resize'):
        input_shape = tuple(config['resize'])
    else:
        input_shape = (182, 218, 182)  # Default
    
    print(f"Creating {model_type} model with input shape: {input_shape}")
    
    if model_type == 'cnn':
        model = create_cnn_model(
            in_channels=1,
            num_classes=2,
            num_layers=config.get('cnn_layers', 3),
            base_filters=config.get('cnn_base_filters', 32),
            dropout_rate=config.get('cnn_dropout', 0.5)
        )
    
    elif model_type == 'vit':
        model = create_vit_model(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            patch_size=config.get('vit_patch_size', 16),
            embed_dim=config.get('vit_embed_dim', 768),
            depth=config.get('vit_depth', 12),
            num_heads=config.get('vit_heads', 12),
            mlp_ratio=config.get('vit_mlp_ratio', 4.0),
            drop_rate=config.get('vit_dropout', 0.1)
        )
    
    elif model_type == 'hybrid':
        model = create_hybrid_model(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            feature_dim=config.get('hybrid_feature_dim', 256),
            transformer_dim=config.get('hybrid_transformer_dim', 256),
            num_heads=config.get('hybrid_num_heads', 8),
            depth=config.get('hybrid_depth', 4),
            mlp_ratio=config.get('hybrid_mlp_ratio', 4.0),
            drop_rate=config.get('hybrid_dropout', 0.1)
        )
    
    elif model_type == 'minit':
        model = create_minit_model_factory(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            model_variant=config.get('minit_variant', 'minit'),
            block_size=config.get('minit_block_size', 16),
            patch_size=config.get('minit_patch_size', 4),
            embed_dim=config.get('minit_embed_dim', 256),
            depth=config.get('minit_depth', 6),
            num_heads=config.get('minit_heads', 8),
            mlp_ratio=config.get('minit_mlp_ratio', 4.0),
            drop_rate=config.get('minit_dropout', 0.125)
        )
    
    elif model_type == 'densenet':
        model = create_densenet_model(
            densenet_type=config.get('densenet_type', '264'),
            num_classes=2,
            drop_rate=config.get('densenet_dropout', 0.0)
        )
    
    elif model_type == 'cnn_minit':
        try:
            from models.cnn_minit_factory import create_cnn_minit_model
           
            model = create_cnn_minit_model(
                target_size=config.get('cnn_minit_target_size', 64),
                frontend_dropout=config.get('cnn_minit_frontend_dropout', 0.2),
                minit_variant=config.get('cnn_minit_variant', 'minit'),
                minit_block_size=config.get('cnn_minit_block_size', 16),
                minit_patch_size=config.get('cnn_minit_patch_size', 4),
                minit_embed_dim=config.get('cnn_minit_embed_dim', 256),
                minit_depth=config.get('cnn_minit_depth', 6),
                minit_heads=config.get('cnn_minit_heads', 8),
                minit_mlp_ratio=config.get('cnn_minit_mlp_ratio', 4),
                minit_dropout=config.get('cnn_minit_dropout', 0.125),
                minit_attn_dropout=config.get('cnn_minit_attn_dropout', 0.125),
                cnn_dropout=config.get('cnn_minit_frontend_dropout', 0.1)
            )
        except ImportError:
            print("CNN-MINiT factory not found. You may need to implement this model factory.")
            raise ValueError(f"Unknown model type: {model_type}")
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model, input_shape

def load_model_checkpoint(model, checkpoint_file):
    """Load model weights from checkpoint"""
    
    try:
        checkpoint = torch.load(checkpoint_file, map_location='cpu', weights_only=False)
    except Exception:
        try:
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
        except Exception as e:
            raise ValueError(f"Could not load checkpoint: {e}")
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'unknown')
        metrics = checkpoint.get('metrics', {})
    else:
        # Assume the checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        epoch = 'unknown'
        metrics = {}
    
    print(f"Successfully loaded model weights from epoch {epoch}")
    if metrics:
        print(f"Best validation metrics: {metrics}")
    
    return model, checkpoint

def evaluate_model(model, dataloader, device):
    """Evaluate model on external dataset"""
    model.eval()
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    all_indices = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            inputs, labels, _, idx = unpack_batch(batch)
            all_indices.extend(idx.tolist() if hasattr(idx, 'tolist') else list(idx))
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of AD class
            all_labels.extend(labels.cpu().numpy())
    
    dataloader.last_indices = np.array(all_indices)
    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)

def calculate_metrics(y_true, y_pred, y_prob):
    """Calculate comprehensive evaluation metrics"""
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Additional metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    metrics = {
        'accuracy': accuracy,
        'auc': auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'balanced_accuracy': balanced_acc,
        'confusion_matrix': {
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    }
    
    return metrics

def plot_results(y_true, y_pred, y_prob, metrics, output_dir, model_name):
    """Create evaluation plots"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    axes[0].set_xticklabels(['Control', 'AD'])
    axes[0].set_yticklabels(['Control', 'AD'])
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[1].plot(fpr, tpr, label=f'ROC Curve (AUC = {metrics["auc"]:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('ROC Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    # Probability Distribution
    axes[2].hist(y_prob[y_true == 0], bins=20, alpha=0.7, label='Control', density=True)
    axes[2].hist(y_prob[y_true == 1], bins=20, alpha=0.7, label='AD', density=True)
    axes[2].set_xlabel('Predicted Probability (AD)')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Probability Distribution')
    axes[2].legend()
    axes[2].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_evaluation_plots.png'), dpi=300)
    plt.close()

def main():
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("external_evaluation.log"),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    try:
        # Auto-discover model files
        logger.info(f"Auto-discovering model files in: {args.model_folder}")
        json_file, checkpoint_file = discover_model_files(args.model_folder)
        logger.info(f"Found JSON config: {json_file}")
        logger.info(f"Found checkpoint: {checkpoint_file}")
        
        # Load configuration from JSON
        logger.info("Loading model configuration from JSON...")
        model_type, training_config, full_config = load_config_from_json(json_file)
        logger.info(f"Detected model type: {model_type}")
        
        # Create model from configuration
        logger.info("Creating model from configuration...")
        model, input_shape = create_model_from_config(model_type, training_config)
        
        # Load model weights
        logger.info("Loading model weights...")
        model, checkpoint = load_model_checkpoint(model, checkpoint_file)
        model = model.to(device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model loaded with {total_params:,} parameters")
        
        # Load external dataset
        logger.info(f"Loading external dataset from: {args.external_csv}")
        if not os.path.exists(args.external_csv):
            raise FileNotFoundError(f"External dataset CSV not found: {args.external_csv}")
        
        # Create temporary CSV with split information for data loader
        external_df = pd.read_csv(args.external_csv)
        external_df['split'] = 'test'  # Mark all as test data
        
        temp_csv_path = os.path.join(args.output_dir, "temp_external_splits.csv")
        external_df.to_csv(temp_csv_path, index=False)
        
        # Create data loader - only for test data to avoid empty train/val sets
        logger.info("Creating data loader for external test data...")
        
        # Create a single dataset and dataloader for test data only
        from data.dataset import AlzheimerDataset
        from torch.utils.data import DataLoader
        
        test_dataset = AlzheimerDataset(
            split_csv_path=temp_csv_path,
            split='test',
            transform=None,
            base_dir=args.base_dir,
            resize=list(input_shape) if input_shape != (182, 218, 182) else None
        )
        
        external_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
        logger.info(f"External dataset loaded with {len(external_loader.dataset)} samples")
        
        # Extract model name for output files
        model_name = f"{model_type}_{Path(args.model_folder).name}"
        
        # Evaluate model
        logger.info("Starting model evaluation...")
        predictions, probabilities, labels = evaluate_model(model, external_loader, device)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        metrics = calculate_metrics(labels, predictions, probabilities)
        
        # Log results
        logger.info("="*60)
        logger.info("EXTERNAL DATASET EVALUATION RESULTS")
        logger.info("="*60)
        logger.info(f"Model: {model_type}")
        logger.info(f"Experiment: {args.model_folder}")
        logger.info(f"External dataset: {args.external_csv}")
        logger.info(f"Total samples: {len(labels)}")
        logger.info(f"AD samples: {np.sum(labels == 1)}")
        logger.info(f"Control samples: {np.sum(labels == 0)}")
        logger.info("-"*60)
        logger.info(f"Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f} ({metrics['balanced_accuracy']*100:.2f}%)")
        logger.info(f"AUC: {metrics['auc']:.4f}")
        logger.info(f"Sensitivity (Recall): {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity: {metrics['specificity']:.4f}")
        logger.info(f"Precision: {metrics['precision']:.4f}")
        logger.info(f"F1 Score: {metrics['f1']:.4f}")
        logger.info("-"*60)
        cm = metrics['confusion_matrix']
        logger.info(f"Confusion Matrix:")
        logger.info(f"  TN: {cm['tn']}, FP: {cm['fp']}")
        logger.info(f"  FN: {cm['fn']}, TP: {cm['tp']}")
        logger.info("="*60)
        
        # Save detailed results
        results = {
            'model_info': {
                'model_type': model_type,
                'model_folder': str(args.model_folder),
                'json_file': str(json_file),
                'checkpoint_file': str(checkpoint_file),
                'total_parameters': total_params,
                'input_shape': list(input_shape)
            },
            'training_config': training_config,
            'original_test_metrics': full_config.get('test_metrics', {}),
            'external_evaluation': {
                'dataset_csv': args.external_csv,
                'total_samples': int(len(labels)),
                'ad_samples': int(np.sum(labels == 1)),
                'control_samples': int(np.sum(labels == 0)),
                'metrics': {k: float(v) if isinstance(v, (np.float64, np.float32)) else v 
                           for k, v in metrics.items() if k != 'confusion_matrix'},
                'confusion_matrix': metrics['confusion_matrix']
            }
        }
        
        # Save results as JSON
        results_file = os.path.join(args.output_dir, f'{model_name}_external_evaluation.json')
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {results_file}")
        
        # Save predictions if requested
        if args.save_predictions:
            id_df = test_dataset.df.iloc[external_loader.last_indices]
            id_cols = [c for c in ('subject_id', 'cohort', 'dataset', 'Dataset', 'scan_id', 'final_file', 'source')
                       if c in id_df.columns]
            predictions_df = pd.DataFrame({
                **{c: id_df[c].values for c in id_cols},
                'true_label': labels,
                'predicted_label': predictions,
                'predicted_probability': probabilities,
                'correct': labels == predictions
            })
            
            pred_file = os.path.join(args.output_dir, f'{model_name}_predictions.csv')
            predictions_df.to_csv(pred_file, index=False)
            logger.info(f"Individual predictions saved to: {pred_file}")
        
        # Create plots
        logger.info("Generating evaluation plots...")
        plot_results(labels, predictions, probabilities, metrics, args.output_dir, model_name)
        logger.info(f"Evaluation plots saved to: {os.path.join(args.output_dir, f'{model_name}_evaluation_plots.png')}")
        
        # Classification report
        report = classification_report(labels, predictions, target_names=['Control', 'AD'])
        logger.info("Classification Report:")
        logger.info("\n" + report)
        
        # Save classification report
        with open(os.path.join(args.output_dir, f'{model_name}_classification_report.txt'), 'w') as f:
            f.write(f"External Dataset Evaluation - {model_name}\n")
            f.write("="*60 + "\n")
            f.write(f"Model folder: {args.model_folder}\n")
            f.write(f"External dataset: {args.external_csv}\n")
            f.write(f"Training accuracy: {full_config.get('test_metrics', {}).get('accuracy', 'N/A')}\n")
            f.write(f"External accuracy: {metrics['accuracy']:.4f}\n\n")
            f.write(report)
        
        # Clean up temporary file
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
        
        logger.info("External dataset evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
