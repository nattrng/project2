import torch
import torch.nn.functional as F





# def create_neighbor_lookup_table(embed_matrix, k): # selects neighbors based on cos similarity and l2 distance now. we have achieved a dynamic set!
#     with torch.no_grad():
#         #cos vers
#         cosine_similarity_scores = F.normalize(embed_matrix, p=2, dim=-1) @ (F.normalize(embed_matrix, p=2, dim=-1)).T
#         cosine_similarity_scores = torch.clamp(cosine_similarity_scores, min=-1.0 + 1e-7, max=1.0-1e-7)
#         angles = torch.acos(cosine_similarity_scores)
#         angles.fill_diagonal_(float('inf')) #angle of vector with itself is 0. would be selected lmao.
#         cos_sorted = torch.topk(torch.acos(cosine_similarity_scores), k=k, dim=-1, largest=False, sorted=True)
#         cos_angles, cos_indices = cos_sorted[0], cos_sorted[1]
    
#         # embed_matrix = embed_matrix.unsqueeze(0) # (1, seq_len, dim)
#         # dists = torch.cdist(embed_matrix, embed_matrix, p=2).squeeze() # cdist necessitates batch dim for som e reason. creates it, does the cdist, then removes the auxiliary dim.
#         # # distance of a vector with itself is zero. fill it with float('inf') so that topk doesnl't select it
#         # dists.fill_diagonal_(float('inf'))
#         # sorted = torch.sort(dists, dim=-1)
#         # l2_indices = sorted[1]
#         # l2_distances = sorted[0]

#         # filter = cos_indicies == l2_indices

#         return cos_angles, cos_indices


# def SynonymCrossEntropy(outputs, labels, cos_angles, cos_indices, lr, tokenizer): #olmo tokenizer has decode func. 

#     #threshold ϵ [0,1] btw

#     logits = outputs.get('logits')
#     probs = F.softmax(logits, dim=-1)
#     loss = 0

#     decoded_targets = tokenizer.decode(labels) #should be in the same shape as the labels, albeit with strings
    


    
#     batch_size, seq_len = labels.shape
#     for b in range(batch_size):
#         for i in range(seq_len):
#             # target = embed_matrix[labels[b, i]] # do not normalize, location in embedding space matters, not just direction.
#             target_error = -torch.log(probs[b, i, labels[b, i]])
#             # dist = torch.norm(embed_matrix - target, dim=-1, p=2)
#             # dist[labels[b, i]] = float('inf')
#             # knn = dist.topk(num_neighbors, largest=False, sorted=True)
#             with torch.no_grad():
#                 neighbor_dist, neighbor_indices = cos_angles[labels[b, i]], cos_indices[labels[b, i]]
#                 neighbor_weighting = F.softmax(-neighbor_dist, dim=-1)

#             # scaling_factor = max_neighbor_weighting_threshold / (target_distances[0] + 1e-12)

#             neighbor_neg_log_likelihoods = -(torch.log(probs[b, i, neighbor_indices] + 1e-12)) * neighbor_weighting #softmax as weighting
#             loss += target_error + (lr*1000) * neighbor_neg_log_likelihoods.sum()
#             counter += 1 + len(neighbor_indices) 


#     return (outputs.get('logits'), loss / len(neighbor_indices[0]))
 # ----
    # loss = F.cross_entropy(logits, labels, ignore_index=ignore_index, reduction=reduction)
    # batch_size, seq_len = labels.shape
    # for b in range(batch_size):
    #     for i in range(seq_len):
    #         # target = embed_matrix[labels[b, i]] # do not normalize, location in embedding space matters, not just direction.
    #         target_error = -torch.log(probs[b, i, labels[b, i]])
    #         # dist = torch.norm(embed_matrix - target, dim=-1, p=2)
    #         # dist[labels[b, i]] = float('inf')
    #         # knn = dist.topk(num_neighbors, largest=False, sorted=True)
    #         with torch.no_grad():
    #             neighbor_dist, neighbor_indices = lookup_table[labels[b, i].item()][0], lookup_table[labels[b, i].item()][1]
    #             neighbor_exp = torch.exp((-1/80 * neighbor_dist), dim=-1)
    #             neighbor_sum = torch.sum(neighbor_exp, dim=-1, keepdim=True)
    #             neighbor_softmax = neighbor_exp / neighbor_sum

    #             # neighbor_weighting = F.softmax(-neighbor_distances, dim=-1)

    #         # scaling_factor = max_neighbor_weighting_threshold / (target_distances[0] + 1e-12)

    #         neighbor_neg_log_likelihoods = -(torch.log(probs[b, i, neighbor_indices] + 1e-12)) * neighbor_softmax #softmax as weighting
    #         loss += target_error + (lr * 1000) * neighbor_neg_log_likelihoods.sum()
    #         counter += 1 + len(neighbor_indices) 
 

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
