import numpy as np
import torch

def find_features(valid_coords, h_coords, h_feats):
    assert valid_coords.shape[-1] == h_coords.shape[-1]
    device = valid_coords.device
    N = valid_coords.size(0)
    M = h_coords.size(0)
    C = h_feats.size(1)

    both = torch.cat([h_coords, valid_coords], dim=0)
    uniq, inv = torch.unique(both, dim=0, return_inverse=True)

    inv_h = inv[:M]
    inv_valid = inv[M:]

    h_raw_ids = torch.arange(M, device=device, dtype=torch.long)
    map_id2h_row = torch.full((uniq.size(0),), -1, device=device, dtype=torch.long)
    map_id2h_row.scatter_(0, inv_h, h_raw_ids)

    idx_in_h = map_id2h_row[inv_valid]
    mask = idx_in_h != -1
    n_matched = mask.sum().item()

    if n_matched > 0:
        new_coords = valid_coords[mask]
        new_feats = h_feats[idx_in_h[mask]]
    else:
        new_coords = valid_coords
        new_feats = torch.zeros((N, C), device=device, dtype=h_feats.dtype)

    return new_coords, new_feats