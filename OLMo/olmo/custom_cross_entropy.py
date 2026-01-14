import torch
import torch.nn.functional as F


def cross_entropy_loss(
    logits,
    labels,
    perp_values: torch.Tensor = None,
    perp_indices: torch.Tensor = None, #should be rank 1 tensor
    lookup_surrogate_to_self_tokens: torch.Tensor = None,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
):
    if perp_indices is not None:
        #we use torch.gather with the perp_indices as indices (batch_size, seq_len - 1, k). We use the output logits of shape (batch_size, seq_len - 1, vocab) 
        if logits.dim() == 2:
            batch_size, seq_len = perp_indices.shape[0], perp_indices.shape[1]
            logits = logits.view(batch_size, seq_len, logits.shape[-1])
        elif logits.dim() == 3:
            batch_size, seq_len = logits.shape[0], logits.shape[1]
        else:
            raise ValueError("logits must be rank 2 or 3 when perp_indices is set")

        labels_for_z = labels.view(batch_size, seq_len) if labels.dim() == 1 else labels
        labels_flat = labels.view(-1)
        k = (perp_indices.shape)[-1] #grabs last dim which is k.
        
        translated_perp_indices = lookup_surrogate_to_self_tokens[perp_indices]
        translated_perp_indices = translated_perp_indices.clamp(min=0)
        gathered_logits = torch.gather(logits, dim=2, index=translated_perp_indices)
        gathered_logit_probs = F.softmax(gathered_logits, dim=-1)
        gathered_nll = -torch.log(gathered_logit_probs + 1e-10) # (batch_len, seq_len, k)

        isinf_bool = torch.isinf(perp_values).all(dim=-1)
        non_included = (~isinf_bool).sum().item() * k
        mask = (~isinf_bool).unsqueeze(-1)

        safe_perp_values = perp_values.clone()
        safe_perp_values[isinf_bool] = 0
        masked_softmax_weight = F.softmax(-safe_perp_values, dim=-1) * mask
        
        weighted_loss_tensor = gathered_nll * masked_softmax_weight
        surr_loss_term = torch.sum(weighted_loss_tensor, dim=-1) #(batch_len, seq_len) keepdim should be false automatically.
        
        logits_flat = logits.view(-1, logits.shape[-1])
        loss = surr_loss_term.sum() + F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction="sum") #reduction='none' returns tensor with losses. shoudl be in same shape as (batch_size, seq)
        if reduction == "mean":
            loss /= ((labels_flat != ignore_index).sum().item() + (batch_size * seq_len * k - non_included))
            

        if not compute_z_loss:
            return loss, None

        z_squared = logits.logsumexp(-1).pow(2) #we can keep this, doesn't do anything harmful
        if reduction == "mean":
            z_squared = (z_squared * (labels_for_z != ignore_index)).mean()
        elif reduction == "sum":
            z_squared = (z_squared * (labels_for_z != ignore_index)).sum() # I will not care for this portion. It is not in my project

        z_loss = z_loss_multiplier * z_squared

        return loss, z_loss
    
    else:
        
        if logits.dim() == 3:
            logits = logits.view(-1, logits.shape[-1])
            labels = labels.view(-1)
        loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)

        if not compute_z_loss:
            return loss, None

        z_squared = logits.logsumexp(-1).pow(2)
        if reduction == "mean":
            z_squared = (z_squared * (labels != ignore_index)).mean()
        elif reduction == "sum":
            z_squared = (z_squared * (labels != ignore_index)).sum()

        z_loss = z_loss_multiplier * z_squared

        return loss, z_loss
