import copy
import torch.nn as nn

def clone_layer(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])