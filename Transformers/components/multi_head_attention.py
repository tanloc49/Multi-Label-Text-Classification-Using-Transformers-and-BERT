import torch
import torch.nn as nn


def masked_fill_with_small_value(tensor, mask, small_value=-1e9):
    return tensor.masked_fill(mask == 0, small_value)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, mask=None):
        """
        x: Tensor of shape [batch_size, seq_len, embed_dim]
        mask: Tensor of shape [batch_size, 1, 1, seq_len] or [batch_size, 1, seq_len, seq_len] (optional)
        """

        batch_size = query.size(0)  # batch size

        # 1. Tạo Q, K, V thông qua các lớp Linear   
        Q = self.query(query)  # [batch_size, seq_len, embed_dim]
        K = self.key(key)  # [batch_size, seq_len, embed_dim]
        V = self.value(value)  # [batch_size, seq_len, embed_dim]

        # 2. Thay đổi kích thước của Q, K, V để phù hợp với số heads
        # View: [batch_size, seq_len, embed_dim] -> [batch_size, seq_len, num_heads, head_dim]
        # Transpose: [batch_size, seq_len, num_heads, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # [batch_size, num_heads, seq_len, head_dim]

        # 3. Tính toán attention scores
        scores = torch.matmul(Q,
                              K.transpose(-2, -1)) / self.head_dim ** 0.5  # [batch_size, num_heads, seq_len, seq_len]

        if mask is not None:
            scores = masked_fill_with_small_value(scores, mask)  # Che các giá trị không mong muốn

        # 4. Áp dụng softmax để lấy attention weights
        attention = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 5. Nhân attention weights với V để lấy các giá trị weighted sum
        out = torch.matmul(attention, V)  # [batch_size, num_heads, seq_len, head_dim]

        # 6. Hoán đổi lại các chiều và gộp các heads lại
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)  # [batch_size, seq_len, embed_dim]

        # 7. Đưa qua lớp Linear cuối cùng
        out = self.out(out)  # [batch_size, seq_len, embed_dim]

        return out
