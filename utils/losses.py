import torch
import torch.nn as nn
import torch.nn.functional as F

# Loss equivalent of the model factories
# Basically configs loss on the fly 
class LossFactory:

    
    @staticmethod
    def get_loss(loss_type, **kwargs):
        """
        Get the specified loss function.
        
        Parameters:
        -----------
        loss_type : str
            Type of loss function ('ce', 'weighted_ce', 'focal', 'asymmetric', 'margin')
        **kwargs : dict
            Additional parameters for the loss function
            
        Returns:
        --------
        loss_fn : callable
            Loss function
        """
        if loss_type == 'ce':
            return LossFactory.cross_entropy(**kwargs)
        elif loss_type == 'weighted_ce':
            return LossFactory.weighted_cross_entropy(**kwargs)
        elif loss_type == 'focal':
            return LossFactory.focal_loss(**kwargs)
        elif loss_type == 'cb_focal':
            return LossFactory.class_balanced_focal_loss(**kwargs)
        elif loss_type == 'asymmetric':
            return LossFactory.asymmetric_loss(**kwargs)
        elif loss_type == 'margin':
            return LossFactory.margin_loss(**kwargs)
        elif loss_type == 'bce_logits':
            return LossFactory.bce_with_logits(**kwargs)
            
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Stndrd cross entropy
    @staticmethod
    def cross_entropy(reduction='mean', **kwargs):
        return nn.CrossEntropyLoss(reduction=reduction)
    
    # as above, but with weights
    @staticmethod
    def weighted_cross_entropy(class_weights=None, reduction='mean', **kwargs):
        return nn.CrossEntropyLoss(weight=class_weights, reduction=reduction)
    
    # a spot of focal loss
    # (see what I did there?)
    @staticmethod
    def focal_loss(**kwargs):
        gamma = kwargs.get('gamma', 2.0)
        alpha = kwargs.get('alpha', 0.25)
        pos_weight = kwargs.get('pos_weight', None)
        
        return FocalBCEWithLogitsLoss(alpha=alpha, gamma=gamma, pos_weight=pos_weight)
    
    # asym loss - different params for neg vs pos samples
    # could be useful depending on degree of imbalance
    @staticmethod
    def asymmetric_loss(gamma_pos=1.0, gamma_neg=4.0, reduction='mean', **kwargs):
        return AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, reduction=reduction)
    
    # margin loss 
    # needs higher confidence interval to predict AD
    @staticmethod
    def margin_loss(pos_margin=0.7, neg_margin=0.3, pos_weight=2.0, reduction='mean', **kwargs):
        return MarginLoss(pos_margin=pos_margin, neg_margin=neg_margin, pos_weight=pos_weight, reduction=reduction)

    # binary cross entropy with logits
    # this was a weird implementation that needed a wrapper 
    # maybe review later
    @staticmethod
    def bce_with_logits(**kwargs):
        pos_weight = kwargs.get('pos_weight', None)
    
        class BCEWithLogitsWrapper(nn.Module):
            def __init__(self, pos_weight=None):
                super(BCEWithLogitsWrapper, self).__init__()
                self.pos_weight = pos_weight
            
            def forward(self, logits, targets):
                if logits.dim() > 1 and logits.shape[1] > 1:
                    # Extract logits for positive class
                    logits = logits[:, 1]
         
                targets = targets.float()
            
                return F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=self.pos_weight, reduction='mean'
                )
    
        return BCEWithLogitsWrapper(pos_weight=pos_weight)
    
    
    # Class balanced focal loss 
    @staticmethod
    def class_balanced_focal_loss(**kwargs):
        class_counts = kwargs.get('class_counts', [1, 1])
        gamma = kwargs.get('gamma', 2.0)
        beta = kwargs.get('beta', 0.9999)
        device = kwargs.get('device', 'cpu')
        
        pos_weight = calculate_class_balanced_weight(class_counts, beta, device)
        
        return FocalBCEWithLogitsLoss(gamma=gamma, pos_weight=pos_weight)
    
    

class FocalLoss(nn.Module):

    #Focal Loss implementation for binary and multi-class classification.
    
    #Inspired by Lin et al, with the great Kaiming He - https://arxiv.org/abs/1708.02002
   
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):

        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        
        log_softmax = F.log_softmax(inputs, dim=1)
        
        log_probs = torch.gather(log_softmax, dim=1, index=targets.unsqueeze(1))
        log_probs = log_probs.squeeze(1)
        
        probs = torch.exp(log_probs)
        
        focal_weights = (1 - probs) ** self.gamma
        
        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_weights = torch.gather(self.alpha, dim=0, index=targets)
            focal_weights = focal_weights * alpha_weights
        
        # Calculate focal loss
        focal_loss = -focal_weights * log_probs
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class AsymmetricLoss(nn.Module):
    
    def __init__(self, gamma_pos=1.0, gamma_neg=4.0, reduction='mean'):

        super(AsymmetricLoss, self).__init__()
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction
        self.eps = 1e-7  # Small epsilon to avoid numerical instability
    
    def forward(self, inputs, targets):

        # Convert to binary for multi-class case (class index -> one-hot)
        if len(targets.shape) == 1:
            num_classes = inputs.shape[1]
            # Convert targets to one-hot
            targets_one_hot = torch.zeros_like(inputs)
            targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)
            targets = targets_one_hot
        
        probs = torch.sigmoid(inputs)
        
        # Separate predictions for positive and negative examples
        pos_probs = probs * targets
        neg_probs = (1 - probs) * (1 - targets)
        
        # Calculate loss for positive examples
        pos_loss = targets * torch.pow(1 - pos_probs + self.eps, self.gamma_pos) * torch.log(pos_probs + self.eps)
        
        # Calculate loss for negative examples
        neg_loss = (1 - targets) * torch.pow(neg_probs + self.eps, self.gamma_neg) * torch.log(1 - probs + self.eps)
        
        # Combine losses
        loss = -1.0 * (pos_loss + neg_loss)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MarginLoss(nn.Module):
    """
    Margin Loss implementation for binary classification.
    
    Requires higher confidence for minority class predictions.
    """
    
    def __init__(self, pos_margin=0.7, neg_margin=0.3, pos_weight=2.0, reduction='mean'):
        """
        Initialize Margin Loss.
        
        Parameters:
        -----------
        pos_margin : float
            Margin for positive class (minority class) predictions
        neg_margin : float
            Margin for negative class (majority class) predictions
        pos_weight : float
            Weight for positive class (minority class) predictions
        reduction : str
            Reduction method ('mean', 'sum', 'none')
        """
        super(MarginLoss, self).__init__()
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass.
        
        Parameters:
        -----------
        inputs : torch.Tensor
            Model predictions (logits)
        targets : torch.Tensor
            Target labels
            
        Returns:
        --------
        loss : torch.Tensor
            Margin loss
        """
        # Convert targets to float for calculations
        targets = targets.float()
        
        # Get probabilities using sigmoid
        probs = torch.sigmoid(inputs[:, 1] - inputs[:, 0])  # P(class 1) - P(class 0)
        
        # Calculate loss for positive examples (target=1)
        pos_loss = targets * torch.clamp(self.pos_margin - probs, min=0)
        
        # Calculate loss for negative examples (target=0)
        neg_loss = (1 - targets) * torch.clamp(probs - self.neg_margin, min=0)
        
        # Combine losses with positive class weighting
        loss = self.pos_weight * pos_loss + neg_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


# Deprecated currently
# May revisit/reimplement
class CombinedLoss(nn.Module):
# combine multiple loss functions
    
    def __init__(self, losses, weights=None):
        """
        Initialize Combined Loss.
        
        Parameters:
        -----------
        losses : list
            List of loss functions
        weights : list, optional
            List of weights for each loss function (default: equal weighting)
        """
        super(CombinedLoss, self).__init__()
        self.losses = losses
        
        if weights is None:
            self.weights = [1.0 / len(losses)] * len(losses)
        else:
            assert len(weights) == len(losses), "Number of weights must match number of losses"
            self.weights = weights
    
    def forward(self, inputs, targets):

        total_loss = 0.0
        
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss += weight * loss_fn(inputs, targets)
        
        return total_loss
        
# This might not be used since its already wrapper but I'm terrified to remove it
class FocalBCEWithLogitsLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2.0, pos_weight=None, reduction='mean'):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        
    def forward(self, logits, targets):
        if logits.dim() > 1 and logits.shape[1] > 1:
            logits = logits[:, 1]
        
        targets = targets.float()
        
        if self.pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets, reduction='none'
            )
        
        p_t = torch.exp(-bce)  
        focal_factor = (1 - p_t) ** self.gamma
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_factor = alpha_t * focal_factor
        
        loss = focal_factor * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
            
def calculate_class_balanced_weight(class_counts, beta=0.9999, device='cpu'):
    # Ensure class_counts is a list or array of length 2
    if isinstance(class_counts, dict):
        class_counts = [class_counts.get(0, 0), class_counts.get(1, 0)]
    
    # Calculate effective number for each class
    eff_num = [1.0 - (beta ** n) for n in class_counts]
    weights = [(1.0 - beta) / (e + 1e-10) for e in eff_num]  # Add epsilon for numerical stability
    
    # Calculate pos_weight (weight of positive class relative to negative)
    pos_weight = weights[1] / weights[0]
    
    return torch.tensor([pos_weight]).to(device)
