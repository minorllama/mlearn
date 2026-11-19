import torch
import torch.nn as nn

import nnutils

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1) # same as tensor.shape[-1], extracts d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k) # .transpose(-2, -1) swaps the last two dimensions of a tensor
    if mask is not None:
        # applies a binary mask where $0$ indicates positions to hide. Replaces those positions with a large negative number ($-10^9$) so they become zero after the softmax operation.
        scores = scores.masked_fill(mask == 0, -1e9) 
    p_attn = scores.softmax(dim=-1) # take softmax along last dim
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn # attn output and attn weights


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nnutils.clone_layer(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1) # adds another dimension at position 1
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) # -1 in .view says to complute the dimension to be whatever it should be
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous() .contiguous() # re-allocates tensor memory in contiguous blocks for view
            .view(nbatches, -1, self.h * self.d_k) # final wrighed average of all head outputs
        )
        del query # free all the GPU memory allocated
        del key
        del value
        return self.linears[-1](x)