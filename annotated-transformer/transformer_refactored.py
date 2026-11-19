import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from transformer_layer import LayerNorm

import nnutils

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = nnutils.clone_layer(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = nnutils.clone_layer(layer, N)
        self.norm = LayerNorm(layer.size)
    # compare this with Encoder forward: the new piece is the memory
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class EncoderDecoder(nn.Module):
    """ A standard Encoder-Decoder architecture. Base for this and many other models."""
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()        
        self.encoder = encoder
        self.decoder = decoder
        # This transformer is specialized for translation, so target and source embedding are different as languages are different. 
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator # The generator is final projection from the decoder output onto the token space

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        # src_mask is padding for unequal sequence lengths
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        # target mark is causal mask to keep decoder from looking at future tokens
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        # self.proj has shape (batch_size, sequence_length, vocab_size), so the softmax should be the last dim, dim=-1
        return log_softmax(self.proj(x), dim=-1) 

