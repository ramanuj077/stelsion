import torch
import torch.nn as nn

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Binary Focal Loss for highly imbalanced exoplanet classification.
        FL(pt) = -alpha * (1 - pt)^gamma * log(pt)
        """
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # Reshape to 1D vectors
        inputs = inputs.view(-1)
        targets = targets.view(-1).float()
        
        # Avoid log(0) and clamp probabilities
        eps = 1e-7
        inputs = torch.clamp(inputs, eps, 1.0 - eps)
        
        # Standard Binary Cross Entropy
        bce = - (targets * torch.log(inputs) + (1.0 - targets) * torch.log(1.0 - inputs))
        
        # pt represents model's probability of the correct class
        pt = targets * inputs + (1.0 - targets) * (1.0 - inputs)
        
        # Calculate focal weights: downweights easy examples where pt is close to 1
        focal_weight = (1.0 - pt) ** self.gamma
        
        # Class balance factor
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        
        # Fused loss
        loss = alpha_t * focal_weight * bce
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
