import torch
import torch.nn.functional as F

def cross_entropy_loss(
    logits,
    labels,
    weighted_perp_loss_tensor, #should be rank 1 tensor
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    loss = torch.sum(weighted_perp_loss_tensor) + F.cross_entropy(logits=logits, labels=labels, ignore_index=ignore_index, reduction='sum') 
    if reduction == "mean":
        loss /= torch.mean(((labels == ignore_index).sum().item() + len(weighted_perp_loss_tensor)))

    if not compute_z_loss:
        return loss, None

    z_squared = logits.logsumexp(-1).pow(2) #we can keep this, doesn't do anything harmful
    if reduction == "mean":
        z_squared = (z_squared * (labels != ignore_index)).mean()
    elif reduction == "sum":
        z_squared = (z_squared * (labels != ignore_index)).sum() # I will not care for this portion. It is not in my project

    z_loss = z_loss_multiplier * z_squared

    return loss, z_loss
