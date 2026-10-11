#900 lines of code and 10 years off my lifespan

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, roc_curve

from utils.cli import parse_args
from utils.losses import LossFactory
from utils.model_utils import track_parameters, check_output_values
from data.dataset import (create_data_loaders, create_evaluation_dataloader, unpack_batch,
                          set_seed, build_domain_map, cohort_column)
from models.domain_adversarial import FeatureHook, CohortAdversary, ganin_lambda, HEAD_ATTR
from data.augmentation import MRIAugmentation
from models.cnn_factory import create_cnn_model
from models.vision_transformer import VisionTransformer3D
from models.hybrid_model import HybridCNNTransformer
from models.minit_factory import create_minit_model_factory, print_model_info, validate_configuration

def train_epoch(model, dataloader, criterion, optimizer, device, args, hook=None, adversary=None, epoch=0):
    """
    Train for one epoch with optional explicit L2 regularization.
    
    Parameters:
    -----------
    model : torch.nn.Module
        The model to train
    dataloader : torch.utils.data.DataLoader
        DataLoader for training data
    criterion : torch.nn.Module
        Loss function
    optimizer : torch.optim.Optimizer
        Optimizer
    device : torch.device
        Device to train on
    args : argparse.Namespace
        Command-line arguments
    """
    model.train()
    if adversary is not None:
        adversary.train()
    running_loss = 0.0
    correct = 0
    total = 0
    dom_loss_sum, dom_correct, dom_n, lam = 0.0, 0, 0, 0.0
    n_batches = len(dataloader)
    
    all_labels = []
    all_preds = []
    
    global_step = 0
    
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for step, batch in enumerate(progress_bar):
        inputs, labels, domains, _ = unpack_batch(batch)
        inputs, labels, domains = inputs.to(device), labels.to(device), domains.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        
        # Weight decay is applied once, by the optimizer (an extra explicit L2 term used to be added here too).
        
        # Cohort-adversarial term: gradient reversal makes the backbone remove cohort information
        if adversary is not None:
            lam = ganin_lambda((epoch + step / n_batches) / args.epochs, args.dann_weight)
            d_loss, d_correct, d_used = adversary(hook.features, labels, domains, lam)
            loss = loss + d_loss
            dom_loss_sum += d_loss.item() * d_used
            dom_correct += d_correct
            dom_n += d_used
        
        loss.backward()
        
        global_step += 1
        
        # Track parameters if specified in command
        if args.track_parameters:
            track_parameters(model, global_step, model_type=args.model)
        
        # Clip gradients if spec in com
        if args.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            if args.verbose and global_step % 10 == 0: 
                logging.info(f"Gradient norm before clipping: {grad_norm:.4f}")
        
        # Monitor gradients if spec in com
        if args.monitor_gradients and global_step % 10 == 0:
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    if np.isnan(grad_norm) or np.isinf(grad_norm):
                        logging.warning(f"{name} has {grad_norm} gradient")
                    elif grad_norm > 10.0:
                        logging.warning(f"{name} has large gradient: {grad_norm:.4f}")
                    elif grad_norm < 1e-8 and grad_norm > 0:
                        logging.warning(f"{name} has tiny gradient: {grad_norm:.10f}")
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.detach().max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Save for metrics
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        
        # Update progress bar
        progress_bar.set_postfix({'loss': loss.item(), 'acc': 100.*correct/total})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = 100. * correct / total
    
    # Calculate prediction dist
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    logging.info(f"Training class distribution - True labels: {np.bincount(all_labels)}")
    logging.info(f"Training class distribution - Predictions: {np.bincount(all_preds)}")
    
    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    logging.info(f"Training Confusion Matrix:\n{cm}")
    
    # Calculate balanced accuracy
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    balanced_acc = (sensitivity + specificity) / 2
    
    logging.info(f"Training Balanced Accuracy: {balanced_acc * 100:.2f}%")
    
    dom_stats = {}
    if adversary is not None and dom_n:
        dom_stats = {'domain_loss': dom_loss_sum / dom_n, 'domain_acc': 100. * dom_correct / dom_n, 'lambda': lam}
        logging.info(f"Cohort adversary: loss {dom_stats['domain_loss']:.4f}, "
                     f"train cohort acc {dom_stats['domain_acc']:.2f}%, lambda {lam:.3f}")
    
    return epoch_loss, epoch_acc, dom_stats


def validate_with_margin(model, dataloader, criterion, device, margin=0.0):
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation', leave=False):
            inputs, labels, _, _ = unpack_batch(batch)
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            
            probs = torch.softmax(outputs, dim=1)
            
            # This maybe help class imbalance maybe not
            #Honestly probably not worth using
            # Apply margin-based decision rule
            ad_probs = probs[:, 1]  
            
            # Use threshold with margin
            preds = torch.zeros_like(labels)
            preds[ad_probs > (0.5 + margin)] = 1  # Predict AD if prob > 0.5+margin
            preds[ad_probs < (0.5 - margin)] = 0  # Predict Control if prob < 0.5-margin
            # If prob is within margin of 0.5, use standard rule
            middle_mask = ((ad_probs >= (0.5 - margin)) & (ad_probs <= (0.5 + margin)))
            if middle_mask.any():
                preds[middle_mask] = (ad_probs[middle_mask] > 0.5).long()
            
            # Store for metrics calculation
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(ad_probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    epoch_loss = running_loss / len(dataloader.dataset)
    
    accuracy = 100. * (all_preds == all_labels).mean()
    
    # Check class distribution
    logging.info(f"Validation class distribution - True labels: {np.bincount(all_labels)}")
    logging.info(f"Validation class distribution - Predictions: {np.bincount(all_preds)}")
    
    # Raw predictions for first 10 samples of val set
    # Mostly for debug - if these are all the same the network isn't learning
    logging.info(f"Sample outputs (first 10):")
    for i in range(min(10, len(all_labels))):
        logging.info(f"  Sample {i}: True={all_labels[i]}, Pred={all_preds[i]}, Prob(AD)={all_probs[i]:.4f}")
    
    # ROC-AUC
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        auc = 0.0
    
    # confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    logging.info(f"Confusion Matrix:\n{cm}")
    
    try:
        tn, fp, fn, tp = cm.ravel()
        
        # Calculate precision, recall, F1, specificity, balanced accuracy
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        balanced_acc = (recall + specificity) / 2
    
    # This should never pop but who knows
    except Exception as e:
        logging.warning(f"Error calculating metrics: {e}")
        precision, recall, f1, specificity, balanced_acc = 0, 0, 0, 0, 0.5
    
    # metrics to dict
    metrics = {
        'loss': epoch_loss,
        'accuracy': accuracy,
        'balanced_acc': balanced_acc * 100,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': {
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        }
    }
    
    return metrics


def plot_history(history, output_dir, model_name):
    """Plot training and validation metrics with enhanced visualization."""
    plt.figure(figsize=(15, 10))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['val_balanced_acc'], label='Val Balanced Acc', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Metrics')
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history['val_auc'], label='Val AUC', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    plt.title('ROC AUC')
    
    # Plot F1, Recall, Precision
    plt.subplot(2, 2, 4)
    plt.plot(history['val_f1'], label='F1 Score', color='purple')
    plt.plot(history['val_recall'], label='Recall', color='orange')
    plt.plot(history['val_precision'], label='Precision', color='brown')
    plt.plot(history['val_specificity'], label='Specificity', color='pink')
    plt.xlabel('Epoch')
    plt.ylabel('Metric Value')
    plt.legend()
    plt.title('Classification Metrics')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_history.png'))
    plt.close()


def generate_confusion_matrix(model, dataloader, device, output_dir, model_name):

    model.eval()
    y_true = []
    y_pred = []
    y_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            inputs, labels, _, _ = unpack_batch(batch)
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_probs.extend(probs[:, 1].cpu().numpy())
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)
    
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    try:
        auc = roc_auc_score(y_true, y_probs)
    except Exception as e:
        logging.warning(f"Could not calculate AUC: {e}")
        auc = 0.0
    
    plt.figure(figsize=(10, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Control', 'AD']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    
    # Metrics to text file
    metrics_file = os.path.join(output_dir, f'{model_name}_test_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Test Set Metrics for {model_name}\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Accuracy: {accuracy:.2f}%\n")
        f.write(f"Balanced Accuracy: {balanced_acc*100:.2f}%\n")
        f.write(f"AUC: {auc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall (Sensitivity): {recall:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(f"TN: {tn}, FP: {fp}\n")
        f.write(f"FN: {fn}, TP: {tp}\n")
    
    return {
        'accuracy': accuracy,
        'balanced_acc': balanced_acc*100,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'confusion_matrix': cm
    }


ID_COLUMNS = ('subject_id', 'cohort', 'dataset', 'Dataset', 'scan_id', 'final_file', 'source')


def predict(model, dataloader, device, hook=None):
    """Per-scan predictions joined to subject/cohort/scan IDs, plus penultimate features if hooked."""
    model.eval()
    probs, labels, idxs, feats = [], [], [], []
    with torch.no_grad():
        for batch in dataloader:
            inputs, y, _, idx = unpack_batch(batch)
            outputs = model(inputs.to(device))
            probs.append(torch.softmax(outputs, dim=1)[:, 1].cpu())
            labels.append(torch.as_tensor(y))
            idxs.append(torch.as_tensor(idx))
            if hook is not None:
                feats.append(hook.features.detach().float().cpu())
    probs, labels, idxs = torch.cat(probs).numpy(), torch.cat(labels).numpy(), torch.cat(idxs).numpy()
    df = dataloader.dataset.df.iloc[idxs]
    table = df[[c for c in ID_COLUMNS if c in df.columns]].reset_index(drop=True)
    table['label'] = labels
    table['prob_ad'] = probs
    table['pred'] = (probs > 0.5).astype(int)
    return table, (torch.cat(feats).numpy() if feats else None)


def summary_metrics(table):
    y, p, yhat = table['label'].values, table['prob_ad'].values, table['pred'].values
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    try:
        auc = float(roc_auc_score(y, p))
    except ValueError:
        auc = float('nan')
    return {'n': int(len(y)), 'auc': auc, 'balanced_acc': 100 * (sens + spec) / 2,
            'sensitivity': sens, 'specificity': spec,
            'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}}


def main():
 
    # Numpy has a scope issue
   
    import numpy as np
    
    # parse_args does not have a scope issue
    args = parse_args()
    # Make. It. Make. Sense.
    set_seed(args.seed, args.deterministic)
    
 
    os.makedirs(args.exp_dir, exist_ok=True)
    
    # Logging
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.log_dir, f'{args.model}_training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f'Starting training for {args.model} model (seed {args.seed})')
    logger.info(f'Experiment: {args.exp_name}')
    
    dropout_rate = args.cnn_dropout if args.model == 'cnn' else args.vit_dropout if args.model == 'vit' else args.hybrid_dropout
    logger.info(f'Regularization: dropout={dropout_rate}, weight_decay={args.weight_decay}')
    logger.info(f'Monitoring {args.monitor} for early stopping and model selection')
    logger.info(f'Decision margin: {args.margin} (threshold = 0.5 ± {args.margin})')
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu >= 0 else 'cpu')
    logger.info(f'Using device: {device}')
    
    if args.no_augmentation:
        logger.info("Data augmentation disabled")
        transforms = {
            'train': None,
            'val': None,
            'test': None
        }
    elif args.all_augmentation:
        logger.info("Using all data augmentations")
        transforms = {
            'train': MRIAugmentation.compose(
                augmentations=None,  # Use all
                p=args.aug_prob,
                target_shape=(182, 218, 182)
            ),
            'val': None,
            'test': None
        }
    elif args.augment:
        logger.info(f"Using selected augmentations: {', '.join(args.augment)}")
        transforms = {
            'train': MRIAugmentation.compose(
                augmentations=args.augment,
                p=args.aug_prob,
                target_shape=(182, 218, 182)
            ),
            'val': None,
            'test': None
        }
    else:
        # Default subset of augmentations 
        default_augs = ['rotation', 'translation', 'scale', 'flip', 'noise']
        logger.info(f"Using default augmentations: {', '.join(default_augs)}")
        transforms = {
            'train': MRIAugmentation.compose(
                augmentations=default_augs,
                p=args.aug_prob,
                target_shape=(182, 218, 182)
            ),
            'val': None,
            'test': None
        }
    
    # Create data loaders
    logger.info(f'Loading data from {args.data}')
    if args.base_dir:
        logger.info(f'Using base directory: {args.base_dir}')
    logger.info("About to create data loaders...")
    logger.info(f"Number of workers: {args.num_workers}")


    try:
        dataloaders = create_data_loaders(
            args.data,
            batch_size=args.batch_size,
            transforms=transforms,
            base_dir=args.base_dir,
            use_balanced_sampler=args.balanced_sampling,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
            resize = args.resize,
            resize_method = args.resize_method,
            seed = args.seed,
            domain_map = build_domain_map(args.data),
            eval_batch_size = args.eval_batch_size
        )
    except FileNotFoundError as e:
        logger.error(f"Error loading dataset: {e}")
        logger.error("Please check if the paths in your CSV file are correct.")
        logger.error("If they're relative paths, use --base_dir to specify the root directory.")
        return
    
    train_df = pd.read_csv(args.data)
    train_df = train_df[train_df['split'] == 'train']
    train_labels = np.array([1 if g == 'AD' else 0 for g in train_df['group']])
    class_counts = np.bincount(train_labels, minlength=2)
    domain_map = build_domain_map(args.data)
    logger.info(f"Training cohorts (domain ids): {domain_map}")
    logger.info(f"Training set class distribution: {class_counts} (gang gang)")    
    
    # Calculate class weights for balanced loss if requested
    if args.class_weights:
        # Compute weights inversely proportional to class frequencies
        n_samples = len(train_labels)
        n_classes = len(class_counts)
        class_weights = n_samples / (n_classes * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(device)
        logger.info(f"Using class weights: {class_weights}")
    else:
        class_weights = None

    
    # Sample a batch to get input dimensions
    # This is mostly debug but also nice to know
#    try:
 #       logger.info("Fetching a sample batch...")
  #      sample_inputs, sample_labels = next(iter(dataloaders['train']))
#        logger.info(f"Sample batch shape: {sample_inputs.shape}")
#        input_shape = sample_inputs.shape[2:]  # D, H, W (excluding batch and channel dims)
#        logger.info(f'Input shape: {input_shape}')
    try:
        # Calculate input shape based on resize flag or use default
        if args.resize:
            input_shape = tuple(args.resize)
            logger.info(f'Using resized input shape: {input_shape}')
        else:
           input_shape = (182, 218, 182)  # Known preprocessing output shape
           logger.info(f'Using default input shape: {input_shape}')
    except Exception as e:
        logger.error(f"Error determining input shape: {e}")
        logger.error(f"Exception details: {str(e)}")
        return
 
    
    # This is the cool part
    # Dynamic model creation from command line parameters
    # Right now only implemented for the 3 "vanilla" models
    # Not sure if this can be adapted to more complex network types, but good for preliminary study
    logger.info(f"Creating {args.model} model with parameters from arguments...")
    
    if args.model == 'cnn':
        model = create_cnn_model(
            in_channels=1,
            num_classes=2,
            num_layers=args.cnn_layers,
            base_filters=args.cnn_base_filters,
            dropout_rate=args.cnn_dropout
        )
        
    elif args.model == 'densenet':
        # NEW: DenseNet creation
        from models.densenet_3d import create_densenet_model
        model = create_densenet_model(
            densenet_type=args.densenet_type,
            num_classes=2,
            drop_rate=args.densenet_dropout
        )
        logger.info(f"Created DenseNet-{args.densenet_type} with growth_rate={args.densenet_growth_rate}")    
        
    elif args.model == 'vit':
        from models.vit_factory import create_vit_model
        model = create_vit_model(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            patch_size=args.vit_patch_size,
            embed_dim=args.vit_embed_dim,
            depth=args.vit_depth,
            num_heads=args.vit_heads,
            mlp_ratio=args.vit_mlp_ratio,
            drop_rate=args.vit_dropout,
            attn_drop_rate=args.vit_dropout
        )
        
    elif args.model == 'minit':
        # Validate configuration
        issues = validate_configuration(
            model_variant=args.minit_variant,
            vol_size=input_shape,
            block_size=args.minit_block_size,
            patch_size=args.minit_patch_size,
            embed_dim=args.minit_embed_dim,
            num_heads=args.minit_heads
        )
        
        if issues and args.verbose:
            for issue in issues:
                logger.warning(f"MINiT config: {issue}")
        
        # Create MINiT model
        model = create_minit_model_factory(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            model_variant=args.minit_variant,
            block_size=args.minit_block_size,
            patch_size=args.minit_patch_size,
            embed_dim=args.minit_embed_dim,
            depth=args.minit_depth,
            num_heads=args.minit_heads,
            mlp_ratio=args.minit_mlp_ratio,
            drop_rate=args.minit_dropout,
            attn_drop_rate=args.minit_attn_dropout,
            amp_enabled=args.mixed_precision
        )
        
        logger.info(f"Created MINiT model variant: {args.minit_variant}")
        if args.verbose:
            print_model_info(model, args.minit_variant)
            
    
    
    elif args.model == 'cnn_minit':
        from models.cnn_minit_factory import create_cnn_minit_model
    
        model = create_cnn_minit_model(
            # CNN-specific parameters
            target_size=args.cnn_minit_target_size,
            cnn_dropout=args.cnn_minit_frontend_dropout,
        
            # MINiT backend parameters - using cnn_minit specific args
            minit_variant=args.cnn_minit_variant,
            minit_block_size=args.cnn_minit_block_size,
            minit_patch_size=args.cnn_minit_patch_size,
            minit_embed_dim=args.cnn_minit_embed_dim,
            minit_depth=args.cnn_minit_depth,
            minit_heads=args.cnn_minit_heads,
            minit_mlp_ratio=args.cnn_minit_mlp_ratio,
            minit_dropout=args.cnn_minit_dropout,
            minit_attn_dropout=args.cnn_minit_attn_dropout,
        
            num_classes=2,
            amp_enabled=args.mixed_precision
        )
    
        logger.info(f"Created CNN-MINiT hybrid: target_size={args.cnn_minit_target_size}, "
                    f"variant={args.cnn_minit_variant}")
        
    elif args.model == 'hybrid':
        from models.hybrid_factory import create_hybrid_model
        model = create_hybrid_model(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2,
            feature_dim=args.hybrid_feature_dim,
            transformer_dim=args.hybrid_transformer_dim,
            num_heads=args.hybrid_num_heads,
            depth=args.hybrid_depth,
            mlp_ratio=args.hybrid_mlp_ratio,
            drop_rate=args.hybrid_dropout
        )
    
    model = model.to(device)
    logger.info(f'Created {args.model} model')
    
    # Penultimate features: needed for the cohort adversary and for --export_features
    hook = None
    if args.dann or args.export_features:
        if args.model in HEAD_ATTR:
            hook = FeatureHook(model, args.model)
        elif args.dann:
            raise ValueError(f"--dann is not supported for '{args.model}' (no single penultimate feature vector)")
        else:
            logger.warning(f"--export_features is not supported for '{args.model}'; skipping feature export")
    
    adversary = None
    if args.dann:
        train_ds = dataloaders['train'].dataset
        counts = torch.zeros(2, len(domain_map))
        for y, d in zip(train_ds.labels, train_ds.domains):
            if d >= 0:
                counts[y, d] += 1
        adversary = CohortAdversary(hook.dim, len(domain_map), hidden=args.dann_hidden,
                                    conditional=args.dann_conditional, class_domain_counts=counts).to(device)
        logger.info(f"Cohort adversary on {hook.dim}-d features, {len(domain_map)} cohorts, "
                    f"{'class-conditional' if args.dann_conditional else 'unconditional'}, "
                    f"max lambda {args.dann_weight}. Train scans per class x cohort:\n{counts.int().tolist()}")
    adv_params = list(adversary.parameters()) if adversary is not None else []
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {total_params:,} parameters, {trainable_params:,} trainable")
    
    # Freeze if specified in command line
    if args.initial_freeze > 0:
        if args.model == 'cnn':
            # Freeze all layers except the final classifier
            for name, param in model.named_parameters():
                if 'fc' not in name:  # Freeze everything except FC layers
                    param.requires_grad = False
            
            # Only optimize the final classifier
            optimizer_params = filter(lambda p: p.requires_grad, model.parameters())
            logger.info(f"Freezing feature extractor for {args.initial_freeze} epochs")
        else:
            # For other models, just use all parameters
            optimizer_params = model.parameters()
    else:
        optimizer_params = model.parameters()
    optimizer_params = list(optimizer_params) + adv_params
    
    # loss function based on command line args
    loss_kwargs = {}

    if args.loss in ['cb_focal', 'bce_logits']:
        
        class_counts = {0: np.sum(train_labels == 0), 1: np.sum(train_labels == 1)}
        logger.info(f"Using class counts for loss weighting: {class_counts}")
        
        loss_kwargs['class_counts'] = class_counts
        loss_kwargs['device'] = device
        if args.loss == 'bce_logits' and args.class_weights:
            # bce_with_logits only reads pos_weight; class counts alone were silently ignored
            loss_kwargs['pos_weight'] = torch.tensor([class_counts[0] / class_counts[1]], device=device)
        
        # Look if you've read this far you get the point
        # Everything is controlled via command line flags
        if args.loss == 'cb_focal':
            loss_kwargs['gamma'] = args.focal_gamma
            loss_kwargs['beta'] = args.beta
    elif args.loss == 'focal':
        loss_kwargs['gamma'] = args.focal_gamma
        loss_kwargs['alpha'] = args.focal_alpha
        
        if class_weights is not None:
            loss_kwargs['pos_weight'] = class_weights[1] / class_weights[0]
    elif args.loss == 'weighted_ce':
        loss_kwargs['class_weights'] = class_weights
    elif args.loss == 'asymmetric':
        loss_kwargs['gamma_pos'] = args.gamma_pos
        loss_kwargs['gamma_neg'] = args.gamma_neg
    elif args.loss == 'margin':
        loss_kwargs['pos_margin'] = args.pos_margin
        loss_kwargs['neg_margin'] = args.neg_margin
        loss_kwargs['pos_weight'] = args.pos_weight

    # Create loss function using, you guessed it...
    criterion = LossFactory.get_loss(args.loss, **loss_kwargs)
    logger.info(f"Using loss function: {args.loss} with parameters: {loss_kwargs}")
    
    # Create optimizer. Guess how?
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            optimizer_params,
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    
    # Configure LR scheduler, same thing
    if args.warmup_epochs > 0:
        logger.info(f"Using {args.warmup_method} warmup for {args.warmup_epochs} epochs")
        
        if args.warmup_method == 'linear':
            def warmup_schedule(epoch):
                if epoch < args.warmup_epochs:
                    return float(epoch) / float(max(1, args.warmup_epochs))
                return 1.0
        else:  # exponential
            def warmup_schedule(epoch):
                if epoch < args.warmup_epochs:
                    return 0.1 ** (1.0 - float(epoch) / float(args.warmup_epochs))
                return 1.0
                
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=warmup_schedule
        )
    else:
        warmup_scheduler = None
    
    # Create main scheduler
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='max',
            factor=args.plateau_factor,
            patience=args.plateau_patience
        )
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.cosine_min_lr
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.step_size,
            gamma=args.step_gamma
        )
    else:  
        scheduler = None
    
    # Set up gradient monitoring if the command line says the magic word
    if args.monitor_gradients:
        from utils.gradient_monitor import GradientMonitor
        logger.info("Gradient monitor enabled")
        grad_monitor = GradientMonitor(
            model,
            log_dir=os.path.join(args.log_dir, 'gradient_logs'),
            log_interval=10
        ).register_hooks()
    else:
        grad_monitor = None
    
    # Training loop with early stopping
    best_val_metric = 0.0
    best_epoch = 0
    no_improvement = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_balanced_acc': [],
        'val_auc': [],
        'val_precision': [],
        'val_recall': [],
        'val_f1': [],
        'val_specificity': [],
        'train_domain_acc': []
    }
    
    for epoch in range(args.epochs):
        logger.info(f'Epoch {epoch+1}/{args.epochs}')
        
        # Apply warmup schedule
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            current_lr = optimizer.param_groups[0]['lr']
            warmup_scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            logger.info(f"Warmup: LR adjusted from {current_lr:.6f} to {new_lr:.6f}")
        
        # Unfreeze feature extractor after initial_freeze epochs
        if epoch == args.initial_freeze and args.initial_freeze > 0:
           logger.info("Unfreezing all model parameters")
           for param in model.parameters():
               param.requires_grad = True
           
           # Update optimizer to include all parameters
           if args.optimizer == 'adam':
               optimizer = torch.optim.Adam(
                   list(model.parameters()) + adv_params, 
                   lr=args.lr * 0.1,  # Lower learning rate after unfreezing
                   weight_decay=args.weight_decay
               )
           elif args.optimizer == 'adamw':
               optimizer = torch.optim.AdamW(
                   list(model.parameters()) + adv_params, 
                   lr=args.lr * 0.1,
                   weight_decay=args.weight_decay
               )
           elif args.optimizer == 'sgd':
               optimizer = torch.optim.SGD(
                   list(model.parameters()) + adv_params,
                   lr=args.lr * 0.1,
                   momentum=0.9,
                   weight_decay=args.weight_decay
               )
           
           # Update trainable parameter count - just for debug
           trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
           logger.info(f"Now training all {trainable_params:,} parameters")
       
        # Train one epoch
        train_loss, train_acc, dom_stats = train_epoch(
            model, dataloaders['train'], criterion, optimizer, device, args,
            hook=hook, adversary=adversary, epoch=epoch
        )
        history['train_domain_acc'].append(dom_stats.get('domain_acc'))
        logger.info(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
        
        # Validate with margin
        val_metrics = validate_with_margin(
            model, dataloaders['val'], criterion, device, margin=args.margin
        )
        
        # Log validation metrics
        logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                   f"Acc: {val_metrics['accuracy']:.2f}%, "
                   f"Balanced Acc: {val_metrics['balanced_acc']:.2f}%, "
                   f"AUC: {val_metrics['auc']:.4f}, "
                   f"F1: {val_metrics['f1']:.4f}, "
                   f"Recall: {val_metrics['recall']:.4f}, "
                   f"Precision: {val_metrics['precision']:.4f}")
        
        # Store current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update learning rate based on scheduler
        if scheduler is not None:
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics[args.monitor])
            else:
                scheduler.step()
        
        # Check if learning rate changed/log it
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            logger.info(f'Learning rate reduced from {current_lr} to {new_lr}')
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_balanced_acc'].append(val_metrics['balanced_acc'])
        history['val_auc'].append(val_metrics['auc'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_specificity'].append(val_metrics['specificity'])
        
        current_val_metric = val_metrics[args.monitor]
        
        # Save best model - check for early stopping
        if current_val_metric > best_val_metric:
            best_val_metric = current_val_metric
            best_epoch = epoch
            no_improvement = 0
            
            model_path = os.path.join(args.ckpt_dir, f'{args.model}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': val_metrics,
                'args': args,
                'domain_map': domain_map,
                'adversary_state_dict': adversary.state_dict() if adversary is not None else None,
            }, model_path)
            logger.info(f'Saved best model with {args.monitor}: {current_val_metric:.4f}')
            
            # Save intermediate checkpoint if requested
            if args.save_freq > 0:
                if (epoch + 1) % args.save_freq == 0:
                    checkpoint_path = os.path.join(args.ckpt_dir, f'{args.model}_epoch{epoch+1}.pth')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'metrics': val_metrics,
                        'args': args,
                    }, checkpoint_path)
                    logger.info(f'Saved checkpoint at epoch {epoch+1}')
        else:
            no_improvement += 1
            logger.info(f'No improvement for {no_improvement} epochs '
                       f'(best {args.monitor}: {best_val_metric:.4f} at epoch {best_epoch+1})')
            
            if no_improvement >= args.early_stopping:
                logger.info(f'Early stopping after {epoch+1} epochs')
                break
    
    
    # Load the best model
    # This has some weird bugs with torch pre/post 2.6
    # Apparently loading with safe globals helps?
    # I dunno someone smarter than me on stackoverflow said to do this 
    
    logger.info(f'Loading best model from epoch {best_epoch+1} for final evaluation')
    try:
        best_model_path = os.path.join(args.ckpt_dir, f'{args.model}_best.pth')
        checkpoint = torch.load(best_model_path, weights_only=False)
        logger.info("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        logger.warning(f"Error loading checkpoint: {e}")
        
        try:
            import numpy as np
            torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
            checkpoint = torch.load(best_model_path, weights_only=True)
            logger.info("Successfully loaded checkpoint with added safe globals")
        except Exception as e:
            logger.error(f"Failed to load checkpoint with both methods: {e}")
            logger.warning("Continuing evaluation without loading the best model")
            checkpoint = {
                'epoch': best_epoch,
                'model_state_dict': model.state_dict(),
                'metrics': {}
            }
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Model state successfully loaded from checkpoint")
    else:
        logger.warning("Checkpoint did not contain model_state_dict, using current model state")
    
    # Plot training history
    plot_history(history, args.exp_dir, args.model)
    
    # Close gradient monitor if used
    if grad_monitor is not None:
        grad_monitor.close()
    
    # Evaluate on val and every held-out split (test, external, ...), saving per-scan predictions
    # with subject / cohort / scan IDs so errors can be compared across models.
    split_metrics = {}
    for split, loader in dataloaders.items():
        if split == 'train':
            continue
        table, feats = predict(model, loader, device, hook if args.export_features else None)
        table.insert(0, 'split', split)
        table.to_csv(os.path.join(args.exp_dir, f'{args.model}_{split}_predictions.csv'), index=False)
        if feats is not None:
            np.save(os.path.join(args.exp_dir, f'{args.model}_{split}_features.npy'), feats)
        split_metrics[split] = summary_metrics(table)
        m = split_metrics[split]
        logger.info(f"{split}: n={m['n']}, AUC {m['auc']:.4f}, balanced acc {m['balanced_acc']:.2f}%")
        if split != 'val':
            generate_confusion_matrix(model, loader, device, args.exp_dir, f'{args.model}_{split}')
    
    if args.export_features and hook is not None:
        # Unaugmented, unshuffled pass over the training split, e.g. for cohort decodability checks
        train_eval = create_evaluation_dataloader(
            args.data, batch_size=args.eval_batch_size, base_dir=args.base_dir, num_workers=args.num_workers,
            resize=args.resize, resize_method=args.resize_method, split='train', domain_map=domain_map)
        table, feats = predict(model, train_eval, device, hook)
        table.insert(0, 'split', 'train')
        table.to_csv(os.path.join(args.exp_dir, f'{args.model}_train_predictions.csv'), index=False)
        np.save(os.path.join(args.exp_dir, f'{args.model}_train_features.npy'), feats)
    
    import json
    results_file = os.path.join(args.exp_dir, f'{args.model}_final_results.json')
    with open(results_file, 'w') as f:
        results = {
            'split_metrics': split_metrics,
            'test_metrics': split_metrics.get('test', {}),
            'best_validation': {
                'epoch': best_epoch + 1,
                'metric': args.monitor,
                'value': float(best_val_metric),
            },
            'domain_map': domain_map,
            'history': history,
            'training_config': vars(args)
        }
        json.dump(results, f, indent=2, default=str)
    
    logger.info('Training and evaluation complete!')
    logger.info(f'Results saved to {args.exp_dir}')
    
    return split_metrics
# I drank so much redbull I could win a race vs a horse
if __name__ == '__main__':
    main()                      
