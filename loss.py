import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Myloss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(Myloss, self).__init__()
        self.epsilon = epsilon

    def forward(self, input_, label, weight):
        entropy = -label * torch.log(input_ + self.epsilon) - (1 - label) * torch.log(1 - input_ + self.epsilon)
        return torch.sum(entropy * weight) / 2

def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

def grl_hook(coeff):
    def hook(grad):
        return -coeff * grad.clone()
    return hook

def GVB(input_list, ad_net, coeff=None, myloss=Myloss(), GVBD=False):
    """
    Enhanced GVB function that can handle both softmax outputs and features
    
    Args:
        input_list: Can be either:
                   - [softmax_output, focals] (legacy format)
                   - [softmax_output, focals, features] (new format)
        ad_net: Adversarial network
        coeff: GRL coefficient
        myloss: Loss function
        GVBD: Whether to use GVBD
    """
    if len(input_list) == 3:
        # New format: [softmax_output, focals, features]
        softmax_output = input_list[0]  # For entropy calculation
        focals = input_list[1].view(-1)
        features = input_list[2]        # For adversarial network
        print(f"📋 GVB: Using separate features for adversarial network")
    else:
        # Legacy format: [softmax_output, focals]
        softmax_output = input_list[0]
        focals = input_list[1].view(-1)
        features = softmax_output       # Use softmax for both (less optimal)
        print(f"📋 GVB: Using softmax for both entropy and adversarial network")
    
    batch_size = softmax_output.size(0) // 2
    device = softmax_output.device

    # Use features for adversarial network (better for domain adaptation)
    ad_output = ad_net(features)
    
    if isinstance(ad_output, tuple) and len(ad_output) == 2:
        # Dual output case (for GVBD)
        ad_out, fc_out = ad_output
    else:
        # Single output case
        ad_out = ad_output
        fc_out = ad_output

    if GVBD:
        ad_out = torch.sigmoid(ad_out - fc_out)
    else:
        ad_out = torch.sigmoid(ad_out)

    # Domain classification targets: 1 for source, 0 for target
    dc_target = torch.tensor([[1.]] * batch_size + [[0.]] * batch_size, device=device)

    # Entropy weighting using softmax outputs
    entropy = Entropy(softmax_output)
    entropy.register_hook(grl_hook(coeff))
    entropy = torch.exp(-entropy)
    mean_entropy = entropy.mean()

    gvbg = torch.mean(torch.abs(focals))
    gvbd = torch.mean(torch.abs(fc_out))

    source_mask = torch.ones_like(entropy)
    source_mask[batch_size:] = 0
    source_weight = entropy * source_mask

    target_mask = torch.ones_like(entropy)
    target_mask[:batch_size] = 0
    target_weight = entropy * target_mask

    eps = 1e-6  # to prevent division by zero
    weight = (source_weight / (source_weight.sum().detach() + eps)) + \
             (target_weight / (target_weight.sum().detach() + eps))

    return myloss(ad_out, dc_target, weight.contiguous().view(-1, 1)), mean_entropy, gvbg, gvbd