import math

import torch.nn as nn

from components.positional_encoding import PositionalEncoding
from layers.decoder_layer import TransformerDecoderLayer
from layers.encoder_layer import TransformerEncoderLayer


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        self.num_layers = num_layers

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, tgt_mask, memory_mask)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, target_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.encoder = TransformerEncoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.decoder = TransformerDecoder(num_layers, embed_dim, num_heads, ff_dim, dropout)
        self.src_embedding = nn.Embedding(input_dim, embed_dim)
        self.tgt_embedding = nn.Embedding(target_dim, embed_dim)
        self.positional_encoding = PositionalEncoding(embed_dim)
        self.fc_out = nn.Linear(embed_dim, target_dim)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src = self.positional_encoding(src)
        enc_output = self.encoder(src, src_mask)

        tgt = self.tgt_embedding(tgt) * math.sqrt(self.tgt_embedding.embedding_dim)
        tgt = self.positional_encoding(tgt)
        dec_output = self.decoder(tgt, enc_output, tgt_mask, memory_mask)

        output = self.fc_out(dec_output)
        output = self.softmax(output)
        return output
