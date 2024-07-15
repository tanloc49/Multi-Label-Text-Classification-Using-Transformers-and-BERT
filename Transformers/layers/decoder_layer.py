import torch.nn as nn

from Transformers.components.feed_forward_neural_network import FeedForwardNetwork
from Transformers.components.multi_head_attention import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.enc_dec_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.norm3 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, memory_mask=None):
        # Masked Self-Attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)

        # Encoder-Decoder Attention
        enc_dec_attn_output = self.enc_dec_attn(x, enc_output, enc_output, memory_mask)
        x = x + self.dropout(enc_dec_attn_output)
        x = self.norm2(x)

        # Feed Forward Network
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)

        return x
