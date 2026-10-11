"""
Penultimate-feature access and a cohort-adversarial head (DANN, Ganin et al. 2016).

FeatureHook captures the input to each model's final linear layer, so the same
features feed the adversary during training and --export_features at evaluation,
without changing any model class.

CohortAdversary predicts cohort from those features through a gradient reversal
layer. With conditional=True there is one cohort classifier per diagnostic class,
so the backbone is pushed to remove cohort information *within* AD and *within*
controls, rather than removing anything that differs between cohorts with
different AD prevalence or control definitions (which can include disease signal).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Final classification layer for each architecture. MINiT/MIGiT end in a layer
# that combines per-block class logits, so they have no shared feature vector.
HEAD_ATTR = {'cnn': 'fc3', 'densenet': 'classifier', 'vit': 'head', 'hybrid': 'classifier'}


def final_layer(model, model_type):
    if model_type not in HEAD_ATTR:
        raise ValueError(
            f"Feature access is supported for {sorted(HEAD_ATTR)}; '{model_type}' has no single "
            "penultimate feature vector (MINiT-family heads combine per-block logits)."
        )
    layer = getattr(model, HEAD_ATTR[model_type])
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"{model_type}.{HEAD_ATTR[model_type]} is {type(layer).__name__}, expected nn.Linear")
    return layer


class FeatureHook:
    """After each forward pass, .features holds the input to the final linear layer
    (still attached to the graph during training)."""

    def __init__(self, model, model_type):
        self.layer = final_layer(model, model_type)
        self.dim = self.layer.in_features
        self.features = None
        self._handle = self.layer.register_forward_pre_hook(self._store)

    def _store(self, module, inputs):
        self.features = inputs[0]

    def remove(self):
        self._handle.remove()


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lam * grad, None


def grad_reverse(x, lam):
    return _GradReverse.apply(x, lam)


def ganin_lambda(progress, max_weight=1.0, gamma=10.0):
    """Ramp from 0 to max_weight over training (progress in [0, 1])."""
    return max_weight * (2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)


class CohortAdversary(nn.Module):
    def __init__(self, in_dim, n_domains, hidden=256, dropout=0.2, conditional=False, n_classes=2,
                 class_domain_counts=None):
        """class_domain_counts: tensor (n_classes, n_domains) of training scan counts, used to weight
        the cohort loss so that large cohorts do not dominate the adversary."""
        super().__init__()
        if n_domains < 2:
            raise ValueError(f"Domain-adversarial training needs at least 2 training cohorts, got {n_domains}")
        self.conditional = conditional
        n_heads = n_classes if conditional else 1
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
                          nn.Linear(hidden, n_domains))
            for _ in range(n_heads)
        ])
        counts = torch.ones(n_classes, n_domains) if class_domain_counts is None else class_domain_counts.float()
        if not conditional:
            counts = counts.sum(0, keepdim=True)
        # inverse-frequency weights per head, normalised to mean 1 over observed cohorts
        w = torch.where(counts > 0, 1.0 / counts.clamp(min=1), torch.zeros_like(counts))
        w = w / w.sum(1, keepdim=True).clamp(min=1e-12) * (counts > 0).sum(1, keepdim=True)
        self.register_buffer('weights', w)

    def forward(self, features, labels, domains, lam):
        """Returns (loss, n_correct, n_used). Samples with domain -1 are ignored."""
        keep = domains >= 0
        if keep.sum() == 0:
            zero = features.sum() * 0.0
            return zero, 0, 0
        f, y, d = grad_reverse(features[keep], lam), labels[keep], domains[keep]
        if self.conditional:
            losses, correct = [], 0
            for c, head in enumerate(self.heads):
                m = y == c
                if m.any():
                    logits = head(f[m])
                    losses.append(F.cross_entropy(logits, d[m], weight=self.weights[c], reduction='sum'))
                    correct += (logits.argmax(1) == d[m]).sum().item()
            loss = torch.stack(losses).sum() / keep.sum()
        else:
            logits = self.heads[0](f)
            loss = F.cross_entropy(logits, d, weight=self.weights[0])
            correct = (logits.argmax(1) == d).sum().item()
        return loss, correct, int(keep.sum())
