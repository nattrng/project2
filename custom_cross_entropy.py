import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits,
    labels,
    perp_indices: torch.Tensor = None, #should be rank 1 tensor
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    if perp_indices is not None:        
        #we use torch.gather with the perp_indices as indices (batch_size, seq_len - 1, k). We use the output logits of shape (batch_size, seq_len - 1, vocab) 
        gathered_logit_probs = torch.gather(logits, dim=2, index=perp_indices)
        gathered_nll = -torch.log(gathered_logit_probs) # (batch_len, seq_len, k)
        surr_loss_term = torch.sum(gathered_nll, dim=-1) (batch_len, seq_len) keepdim should be false automatically.
        
        
        
        
        
        loss = surr_loss_term + F.cross_entropy(logits=logits, labels=labels, ignore_index=ignore_index, reduction='none') #reduction='none' returns tensor with losses. shoudl be in same shape as (batch_size, seq)
        if reduction == "mean":
            loss /= ((labels != ignore_index).sum().item() + len(weighted_perp_loss_tensor))

        if not compute_z_loss:
            return loss, None

        z_squared = logits.logsumexp(-1).pow(2) #we can keep this, doesn't do anything harmful
        if reduction == "mean":
            z_squared = (z_squared * (labels != ignore_index)).mean()
        elif reduction == "sum":
            z_squared = (z_squared * (labels != ignore_index)).sum() # I will not care for this portion. It is not in my project

        z_loss = z_loss_multiplier * z_squared

        return loss, z_loss
    
    else:
        
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

        if not compute_z_loss:
            return loss, None

        z_squared = logits.logsumexp(-1).pow(2)
        if reduction == "mean":
            z_squared = (z_squared * (labels != ignore_index)).mean()
        elif reduction == "sum":
            z_squared = (z_squared * (labels != ignore_index)).sum()

        z_loss = z_loss_multiplier * z_squared
