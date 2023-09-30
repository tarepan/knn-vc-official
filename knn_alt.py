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


# Test
assert torch.allclose(out_feats, out_feats2)
print("passed")
