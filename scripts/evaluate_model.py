# A little secondary evaluation script thats decoupled from the eval done after training

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.dataset import unpack_batch

def evaluate_model(model_path, data_csv, base_dir=None, device='cuda:0', margin=0.0, batch_size=4):
    import os
    import torch
    import logging
    import numpy as np
    from torch.utils.data import DataLoader
    from data.dataset import create_data_loaders
    from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load the model
    logger.info(f'Loading model from {model_path}')
    try:
        # Try with weights_only=False first
        checkpoint = torch.load(model_path, weights_only=False)
        logger.info("Successfully loaded checkpoint with weights_only=False")
    except Exception as e:
        logger.warning(f"Error loading with weights_only=False: {e}")
        try:
            # Try allowing numpy globals
            import numpy as np
            torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
            checkpoint = torch.load(model_path, weights_only=True)
            logger.info("Successfully loaded checkpoint with added safe globals")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    # Get the model architecture from the model path
    # This might not work if you move the model files to odd dirs
    model_name = os.path.basename(model_path).split('_')[0]
    logger.info(f'Detected model architecture: {model_name}')
    
    logger.info(f'Loading data from {data_csv}')
    if base_dir:
        logger.info(f'Using base directory: {base_dir}')
    
    dataloaders = create_data_loaders(
        data_csv,
        batch_size=batch_size,
        transforms={'train': None, 'val': None, 'test': None}, 
        base_dir=base_dir
    )
    

    test_dataloader = dataloaders['test']
    sample_inputs = unpack_batch(next(iter(test_dataloader)))[0]
    input_shape = sample_inputs.shape[2:]  # D, H, W (excluding batch and channel dims)
    
    # Import the model classes
    from models.cnn_3d import CNN3D
    from models.vision_transformer import VisionTransformer3D
    from models.hybrid_model import HybridCNNTransformer
    
    # Create model with appropriate architecture
    if model_name == 'cnn':
        model = CNN3D(in_channels=1, num_classes=2)
    elif model_name == 'vit':
        model = VisionTransformer3D(
            vol_size=input_shape,
            patch_size=16,
            in_channels=1,
            num_classes=2
        )
    elif model_name == 'hybrid':
        model = HybridCNNTransformer(
            vol_size=input_shape,
            in_channels=1,
            num_classes=2
        )
    else:
        logger.error(f"Unknown model architecture: {model_name}")
        return None
    
    # Load the model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate the model
    logger.info('Evaluating model on test set...')
    all_labels = []
    all_preds = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_dataloader:
            inputs, labels, _, _ = unpack_batch(batch)
            inputs = inputs.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            
            ad_probs = probs[:, 1]  # Probability of class 1 (AD)
            
            # Use threshold with margin
            preds = torch.zeros_like(labels)
            preds[ad_probs > (0.5 + margin)] = 1  # Predict AD if prob > 0.5+margin
            preds[ad_probs < (0.5 - margin)] = 0  # Predict Control if prob < 0.5-margin
            # If prob is within margin of 0.5, use standard rule
            middle_mask = ((ad_probs >= (0.5 - margin)) & (ad_probs <= (0.5 + margin)))
            if middle_mask.any():
                preds[middle_mask] = (ad_probs[middle_mask] > 0.5).long()
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(ad_probs.cpu().numpy())
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    logger.info(f"Confusion Matrix:\n{cm}")
    
    tn, fp, fn, tp = cm.ravel()
    accuracy = 100 * (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    balanced_acc = (recall + specificity) / 2
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception as e:
        logger.warning(f"Could not calculate AUC: {e}")
        auc = 0.0
    
    # Log results
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {accuracy:.2f}%")
    logger.info(f"  Balanced Accuracy: {balanced_acc*100:.2f}%")
    logger.info(f"  AUC: {auc:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Recall (Sensitivity): {recall:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Specificity: {specificity:.4f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title('Confusion Matrix')
    
    classes = ['Control', 'AD']
    tick_marks = np.arange(len(classes))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(classes, rotation=45)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(classes)
    s
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    axes[0].set_ylabel('True label')
    axes[0].set_xlabel('Predicted label')
    
    # ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    axes[1].plot(fpr, tpr, label=f'ROC curve (area = {auc:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Receiver Operating Characteristic')
    axes[1].legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('model_evaluation.png')
    plt.close()
    
    # Return metrics
    metrics = {
        'accuracy': accuracy,
        'balanced_acc': balanced_acc*100,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'specificity': specificity,
        'confusion_matrix': cm
    }
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to the saved model checkpoint')
    parser.add_argument('--data', type=str, required=True,
                      help='Path to dataset_splits.csv file')
    parser.add_argument('--base_dir', type=str, default=None,
                      help='Base directory for resolving file paths in CSV')
    parser.add_argument('--margin', type=float, default=0.0,
                      help='Decision margin for predictions')
    parser.add_argument('--batch_size', type=int, default=4,
                      help='Batch size for evaluation')
    
    args = parser.parse_args()
    
    metrics = evaluate_model(
        args.model, 
        args.data, 
        base_dir=args.base_dir,
        margin=args.margin,
        batch_size=args.batch_size
    )
    
    if metrics:
        print("\nEvaluation Summary:")
        print(f"  Accuracy: {metrics['accuracy']:.2f}%")
        print(f"  Balanced Accuracy: {metrics['balanced_acc']:.2f}%")
        print(f"  AUC: {metrics['auc']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
