import torch.nn as nn
import torch.nn.functional as F


class FeedForwardNetwork(nn.Module):
    def __init__(self, embed_dim, ff_dim, dropout=0.1):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.fc1(x)  # [batch_size, seq_len, ff_dim]
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, seq_len, embed_dim]

        return x
