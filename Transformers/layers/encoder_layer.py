import torch.nn as nn

from Transformers.components.feed_forward_neural_network import FeedForwardNetwork
from Transformers.components.multi_head_attention import MultiHeadAttention


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FeedForwardNetwork(embed_dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-Head Attention
        attn_output = self.self_attn(x, x, x, mask)  # [batch_size, seq_len, embed_dim]
        x = x + self.dropout(attn_output)  # Residual connection + Dropout
        x = self.norm1(x)  # Layer normalization

        # Feed Forward Network
        ffn_output = self.ffn(x)  # [batch_size, seq_len, embed_dim]
        x = x + self.dropout(ffn_output)  # Residual connection + Dropout
        x = self.norm2(x)  # Layer normalization

        return x
