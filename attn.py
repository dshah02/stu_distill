import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()
        assert torch.cuda.is_available(), "CUDA is required."
        assert config.n_embd % config.n_heads == 0
        self.n_heads = config.n_heads

        self.device = torch.device("cuda")
        self.bsz = config.bsz
        self.c_attn = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias, dtype=config.torch_dtype
        )
        self.c_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias, dtype=config.torch_dtype
        )
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        self.window_size = config.window_size

    def forward(self, x):
        bsz, seq_len, d_in = x.size()

        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)

        q = q.view(bsz, seq_len, self.n_heads, d_in // self.n_heads).transpose(1, 2)
        k = k.view(bsz, seq_len, self.n_heads, d_in // self.n_heads).transpose(1, 2)
        v = v.view(bsz, seq_len, self.n_heads, d_in // self.n_heads).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0
        )

        y = y.transpose(1, 2).contiguous().view(bsz, seq_len, d_in)

        y = self.resid_dropout(self.c_proj(y))
        return y

