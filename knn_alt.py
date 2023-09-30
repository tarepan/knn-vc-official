"""Alternative kNN regression implementation."""

import torch
from torch import Tensor


# Prepare
topk = 4
query_seq    = torch.rand([200, 128])
matching_set = torch.rand([200, 128])
synth_set    = matching_set
device = torch.device("cpu")


# Original
def fast_cosine_dist(source_feats: Tensor, matching_pool: Tensor, device: str = 'cpu') -> Tensor:
    """ Like torch.cdist, but fixed dim=-1 and for cosine distance."""
    source_norms = torch.norm(source_feats, p=2, dim=-1).to(device)
    matching_norms = torch.norm(matching_pool, p=2, dim=-1)
    dotprod = -torch.cdist(source_feats[None].to(device), matching_pool[None], p=2)[0]**2 + source_norms[:, None]**2 + matching_norms[None]**2
    dotprod /= 2

    dists = 1 - ( dotprod / (source_norms[:, None] * matching_norms[None]) )
    return dists

dists = fast_cosine_dist(query_seq, matching_set, device=device)
best = dists.topk(k=topk, largest=False, dim=-1)
out_feats = synth_set[best.indices].mean(dim=1)


# Alt
import torchmetrics

dists2 = torchmetrics.functional.pairwise_cosine_similarity(query_seq, matching_set)
dists2 = 1 - dists2
best2 = dists2.topk(k=topk, largest=False, dim=-1)
out_feats2 = synth_set[best2.indices].mean(dim=1)


# Alt2
from torchmetrics.functional import pairwise_cosine_similarity

def knn_alt2(query_series: Tensor, key_pool: Tensor, value_pool: Tensor, top_k: int) -> Tensor:
    """Replace frames with kNN.

    Args:
        query_series :: (Frame=frm, Feat=f) - kNN query series which is replaced by value pool elements based on query-key distance
        key_pool     :: (Choice=c,  Feat=f) - kNN key   pool (distance references)
        value_pool   :: (Choice=c,  Feat=f) - kNN value pool (will be selected based on query-key distance)
        top_k                               - 'k' in the kNN, the number of nearest neighbors to average over.
    Returns:
                     :: (Frame=frm, Feat=f) - Averaged top-K nearest neighbor
    """

    # Q-K Distance / TopK / index :: (Frame, Feat)(Pool, Feat) -> (Frame, Pool) -> (Frame, Topk=topk)
    topk_idx_series = pairwise_cosine_similarity(query_series, key_pool).topk(k=top_k).indices # type:ignore

    # Q-V replace / Average :: (Pool, Feat)(Frame, Topk=topk) -> (Frame, Topk=topk, Feat) -> (Frame, Feat)
    replaced_series = value_pool[topk_idx_series].mean(dim=1)                                 # type:ignore

    return replaced_series

out_feats3 = knn_alt2(query_seq, matching_set, synth_set, topk)


# Test
assert torch.allclose(out_feats,  out_feats2)
assert torch.allclose(out_feats2, out_feats3)
assert torch.allclose(out_feats3, out_feats)
print("passed")
