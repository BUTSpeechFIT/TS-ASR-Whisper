import itertools as it

import torch


def ctc_greedy_decode(logits: torch.Tensor, blank, pad_token_id) -> torch.Tensor:
    idxs = torch.argmax(logits, dim=-1)
    for i, prediction in enumerate(idxs):
        deduplicated = [k for k, g in it.groupby(prediction) if k != blank]
        idxs[i, : len(deduplicated)] = torch.tensor(deduplicated)
        idxs[i, len(deduplicated):] = pad_token_id
    return idxs
