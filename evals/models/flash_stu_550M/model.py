import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
import sys
import os
sys.path.append(os.path.abspath("../src"))

# --- Optional External Modules ---
try:
    from flashfftconv import FlashFFTConv
    flash_fft_available = True
except ImportError as e:
    print(f"Unable to import FlashFFTConv: {e}. Falling back to PyTorch implementation.")
    flash_fft_available = False

try:
    from flash_attn import flash_attn_func
except ImportError as e:
    print(f"Unable to import Triton-based flash attention: {e}. No alternative currently available.")
    flash_attn_func = None

# Assume these modules exist for caching
from kv_cache import KVCache
from cache import Cache

# --- Helper Functions ---

def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return 1 << (math.ceil(math.log2(x)) if round_up else math.floor(math.log2(x)))

def get_hankel(seq_len: int, use_hankel_L: bool = False) -> torch.Tensor:
    entries = torch.arange(1, seq_len + 1, dtype=torch.float64)
    i_plus_j = entries[:, None] + entries[None, :]
    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    else:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    return Z

def get_spectral_filters(seq_len: int, K: int, use_hankel_L: bool = False,
                         device: torch.device = None, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    Z = get_hankel(seq_len, use_hankel_L).to(device)
    sigma, phi = torch.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    return phi_k.to(dtype=dtype)

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float = 10000.0):
    freq_seq = torch.arange(0, head_dim, 2).float() / head_dim
    freqs = 1.0 / (theta ** freq_seq)
    t = torch.arange(max_seq_len, dtype=torch.float32)
    angles = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(angles), angles)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    seq_len = x.shape[2]
    return freqs_cis[:seq_len].view(1, 1, seq_len, -1)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # Convert last dim into complex pairs
    xq_complex = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_complex = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_complex)
    xq_complex = xq_complex * freqs_cis
    xk_complex = xk_complex * freqs_cis
    xq_out = torch.view_as_real(xq_complex).reshape(*xq.shape)
    xk_out = torch.view_as_real(xk_complex).reshape(*xk.shape)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1
    if use_approx:
        _, d_out = v.shape
        v = v.view(1, -1, d_out, 1).to(torch.float32).contiguous()
    else:
        _, K = v.shape
        sgn = sgn.unsqueeze(-1)
        v = v.view(1, -1, K, 1, 1).to(torch.float32).contiguous()
        u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)
    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32).contiguous()
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn
    return U_plus, U_minus

def flash_convolve(u: torch.Tensor, v: torch.Tensor, flash_fft: FlashFFTConv, use_approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, seq_len, d_in = u.shape
    _, K = v.shape
    padded_len = nearest_power_of_two(seq_len, round_up=True)
    pad_len = padded_len - seq_len
    sgn = torch.full((1, 1, padded_len), 1, device=u.device)
    sgn[:, :, 1::2] = -1
    if use_approx:
        u_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).contiguous()
        u_conv = torch.stack([u_padded, u_padded * sgn], dim=0).reshape(2 * bsz, d_in, padded_len)
    else:
        u_k_padded = F.pad(u.transpose(1, 2), (0, pad_len)).to(torch.bfloat16).repeat_interleave(K, dim=1).contiguous()
        v_padded = F.pad(v.transpose(0, 1), (0, pad_len)).to(torch.float32).repeat(d_in, 1).contiguous()
        u_conv = torch.stack([u_k_padded, u_k_padded * sgn], dim=0).reshape(2 * bsz, K * d_in, padded_len)
    U_conv = flash_fft(u_conv, v_padded)
    U_conv = U_conv[..., :seq_len]
    u_plus, u_minus = torch.chunk(U_conv, 2, dim=0)
    if use_approx:
        u_minus = u_minus * sgn[:, :, :seq_len]
        U_plus, U_minus = u_plus.transpose(1, 2), u_minus.transpose(1, 2)
    else:
        sgn_ = sgn[:, :, :seq_len].unsqueeze(-1).transpose(1, 2)
        U_plus = u_plus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous()
        U_minus = u_minus.view(bsz, d_in, K, seq_len).permute(0, 3, 2, 1).contiguous() * sgn_
    return U_plus, U_minus

# --- Model Modules ---

class STU(nn.Module):
    def __init__(self, config, filters) -> None:
        super().__init__()
        self.config = config
        self.stu_filters = filters
        self.n = nearest_power_of_two(config.seq_len * 2 - 1, round_up=True)
        self.K = config.num_eigh
        self.d_in = config.dim
        self.d_out = config.dim
        self.use_hankel_L = config.use_hankel_L
        self.use_approx = config.use_approx
        self.cache = None
        self.flash_fft = (
            FlashFFTConv(self.n, dtype=torch.bfloat16)
            if config.use_flash_fft and flash_fft_available
            else None
        )
        if self.use_approx:
            self.M_inputs = nn.Parameter(torch.empty(self.d_in, self.d_out, dtype=config.torch_dtype))
            self.M_filters = nn.Parameter(torch.empty(self.K, self.d_in, dtype=config.torch_dtype))
        else:
            self.M_phi_plus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype))
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(torch.empty(self.K, self.d_in, self.d_out, dtype=config.torch_dtype))

    def forward(self, x: torch.Tensor, input_pos) -> torch.Tensor:
        if self.use_approx:
            x_proj = x @ self.M_inputs
            phi_proj = self.stu_filters @ self.M_filters
            if self.cache is not None:
                _ = self.cache.update(x_proj, input_pos)
            if self.cache is not None and input_pos.shape[0] == 1:
                x_proj = self.cache.update(x_proj.squeeze(dim=0), input_pos)
                pos = input_pos.item()
                subset_seq = x_proj[:, :pos+1, :]
                flipped_seq = torch.flip(subset_seq, dims=[1])
                sign = torch.ones(phi_proj.size(0), device=phi_proj.device)
                sign[1::2] = -1
                alt_phi_proj = phi_proj * sign.unsqueeze(-1)
                common_length = flipped_seq.size(1)
                flipped_seq_clipped = flipped_seq[:, :common_length, :]
                phi_proj_clipped = phi_proj[:common_length, :]
                alt_phi_proj_clipped = alt_phi_proj[:common_length, :]
                spectral_plus = torch.sum(flipped_seq_clipped * phi_proj_clipped.unsqueeze(0), dim=1, keepdim=True)
                spectral_minus = torch.sum(flipped_seq_clipped * alt_phi_proj_clipped.unsqueeze(0), dim=1, keepdim=True)
                return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
            if self.flash_fft:
                spectral_plus, spectral_minus = flash_convolve(x_proj, phi_proj, self.flash_fft, self.use_approx)
            else:
                spectral_plus, spectral_minus = convolve(x_proj, phi_proj, self.n, self.use_approx)
        else:
            if self.flash_fft:
                U_plus, U_minus = flash_convolve(x, self.stu_filters, self.flash_fft, self.use_approx)
            else:
                U_plus, U_minus = convolve(x, self.stu_filters, self.n, self.use_approx)
            spectral_plus = torch.tensordot(U_plus, self.M_phi_plus, dims=([2, 3], [0, 1]))
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(U_minus, self.M_phi_minus, dims=([2, 3], [0, 1]))
        return spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus

class STULayer(nn.Module):
    def __init__(self, config, stu_filters):
        super().__init__()
        self.stu_norm = nn.RMSNorm(config.dim)
        self.stu = STU(config, stu_filters)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, input_pos) -> torch.Tensor:
        x = x + self.stu(self.stu_norm(x), input_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.dim
        self.intermediate_size = config.dim * config.mlp_scale
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = F.gelu(gate, approximate="tanh")
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        outputs = self.dropout(outputs)
        return outputs

class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim, self.num_heads = config.dim, config.num_heads
        assert config.dim % config.num_heads == 0, f"dim ({config.dim}) must be divisible by num_heads ({config.num_heads})"
        self.head_dim = config.dim // config.num_heads
        self.c_attn = nn.Linear(self.dim, 3 * self.dim, bias=config.bias)
        self.c_proj = nn.Linear(config.dim, config.dim, bias=config.bias)
        self.c_proj.SCALE_INIT = 1
        from torchtune.modules import RotaryPositionalEmbeddings as RoPE
        self.rope = RoPE(self.head_dim, config.seq_len, config.theta)
        self.alibi_slopes = self._get_alibi_slopes(self.num_heads) if config.use_alibi else None
        self.window_size = config.window_size
        self.softcap = config.softcap
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(self.dropout)
        self.kv_cache = None

    def _generate_slopes(self, n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        return [start * (start**i) for i in range(n)]

    def _get_alibi_slopes(self, num_heads: int, interpolation_factor: float = 0.25):
        if math.log2(num_heads).is_integer():
            slopes = self._generate_slopes(num_heads)
        else:
            n = nearest_power_of_two(num_heads, round_up=False)
            slopes_power_of_two = self._generate_slopes(n)
            extra_slopes = self._generate_slopes(2 * n)
            extra_slopes_trunc = extra_slopes[0::2][: num_heads - n]
            slopes = slopes_power_of_two + extra_slopes_trunc
        slopes = torch.tensor(slopes, device=torch.device("cuda"))
        slopes = slopes * interpolation_factor
        return slopes

    def forward(self, x: torch.Tensor = None, q: torch.Tensor = None, k: torch.Tensor = None,
                v: torch.Tensor = None, freqs_cis: torch.Tensor = None, input_pos: torch.Tensor = None) -> torch.Tensor:
        if x is not None:
            q = k = v = x
        if any(t is None for t in [q, k, v]):
            raise ValueError("Must provide either x for self-attention or q/k/v for cross-attention.")
        bsz, q_len, _ = q.shape
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=2)
        q = q.view(bsz, q_len, self.num_heads, self.head_dim)
        k = k.view(bsz, q_len, self.num_heads, self.head_dim)
        v = v.view(bsz, q_len, self.num_heads, self.head_dim)
        q, k = self.rope(q, k)
        if self.kv_cache is not None:
            k, v = self.kv_cache.update(input_pos, k, v)
            if len(input_pos) == 1:
                k = k[:, :input_pos+1]
                v = v[:, :input_pos+1]
            else:
                k = k[:, :max(input_pos)+1]
                v = v[:, :max(input_pos)+1]
        y = flash_attn_func(
            q=q, k=k, v=v,
            dropout_p=self.dropout if self.training else 0.0,
            causal=True,
            window_size=(self.window_size, 0),
            alibi_slopes=self.alibi_slopes,
            softcap=self.softcap,
        )
        y = y.contiguous().view(bsz, q_len, -1)
        y = self.resid_dropout(self.c_proj(y))
        return y

class AttentionLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(config.dim)
        self.attn = Attention(config)
        self.mlp_norm = nn.RMSNorm(config.dim)
        self.mlp = MLP(config)

    def reset(self):
        self.attn.kv_cache = None

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None, input_pos: torch.Tensor = None) -> torch.Tensor:
        x = x + self.attn(x=self.attn_norm(x), freqs_cis=freqs_cis, input_pos=input_pos)
        x = x + self.mlp(self.mlp_norm(x))
        return x

# --- Config and Model Definitions ---

class FlashSTUConfig(PretrainedConfig):
    model_type = "FlashSTU"

    def __init__(self,
                 bsz: int = 8,
                 dim: int = 896,
                 num_heads: int = 8,
                 num_layers: int = 12,
                 seq_len: int = 8192,
                 weight_tying: bool = True,
                 window_size: int = 1024,
                 vocab_size: int = 200064,
                 mlp_scale: int = 12,
                 bias: bool = False,
                 dropout: float = 0.1,
                 num_eigh: int = 24,
                 use_hankel_L: bool = False,
                 use_flash_fft: bool = True,
                 use_approx: bool = True,
                 use_attn: bool = True,
                 softcap: float = 50.0,
                 theta: float = 10000.0,
                 use_alibi: bool = False,
                 dilation: int = 2,
                 torch_dtype: torch.dtype = torch.bfloat16,
                 device: torch.device = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.bsz = bsz
        self.dim = dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.seq_len = seq_len
        self.weight_tying = weight_tying
        self.window_size = window_size
        self.vocab_size = vocab_size
        self.hidden_size = dim
        self.mlp_scale = mlp_scale
        self.dropout = dropout
        self.num_eigh = num_eigh
        self.use_hankel_L = use_hankel_L
        self.use_flash_fft = use_flash_fft
        self.use_approx = use_approx
        self.use_attn = use_attn
        self.softcap = softcap
        self.theta = theta
        self.use_alibi = use_alibi
        self.dilation = dilation
        self.torch_dtype = torch_dtype
        self.device = device

class FlashSTU(PreTrainedModel):
    config_class = FlashSTUConfig

    def __init__(self, config, filters) -> None:
        super().__init__(config)
        self.num_layers = config.num_layers
        self.head_dim = config.dim // config.num_heads
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(self.head_dim, config.seq_len, theta=config.theta),
            persistent=True,
        )
        self.use_approx = config.use_approx
        self.use_hankel_L = config.use_hankel_L
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim, dtype=config.torch_dtype)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList()
        for layer_idx in range(config.num_layers):
            if layer_idx % 2 == 0:
                self.layers.append(STULayer(config, filters))
            else:
                self.layers.append(AttentionLayer(config) if config.use_attn else STULayer(config, filters))
        self.norm = nn.RMSNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=config.bias)
        if config.weight_tying:
            self.tok_emb.weight = self.lm_head.weight
        self.std = config.dim ** -0.5
        self.apply(self._init_weights)
        print("Model Parameter Count: %.2fM" % (self._get_num_params() / 1e6))

    def setup_caches(self, batch_size: int, dtype: torch.dtype = None) -> None:
        if dtype is None:
            dtype = self.config.torch_dtype
        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.attn.kv_cache = KVCache(
                    batch_size=batch_size,
                    max_seq_len=self.config.seq_len,
                    num_heads=self.config.num_heads,
                    head_dim=self.head_dim,
                    dtype=dtype,
                ).to(self.device)
            if hasattr(layer, "stu"):
                layer.stu.cache = Cache(
                    batch_size=batch_size,
                    max_seq_len=self.config.seq_len,
                    dim=self.config.dim,
                    dtype=dtype,
                ).to(self.device)

    def caches_are_enabled(self) -> bool:
        for layer in self.layers:
            if hasattr(layer, "attn") and layer.attn.kv_cache is not None:
                return True
        return False

    def reset_caches(self):
        if not self.caches_are_enabled():
            raise RuntimeError("Key value caches are not setup. Call setup_caches() first.")
        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.attn.kv_cache = None
            elif hasattr(layer, "stu"):
                layer.stu.cache = None

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor = None, cache: bool = False) -> torch.Tensor:
        tok_emb = self.tok_emb(x)
        x = self.dropout(tok_emb)
        for layer in self.layers:
            if hasattr(layer, "attn"):
                x = layer(x, freqs_cis=self.freqs_cis, input_pos=input_pos)
            else:
                x = layer(x, input_pos=input_pos)
        y_hat = self.lm_head(self.norm(x))
        return y_hat

    def reset(self):
        for layer in self.layers:
            if hasattr(layer, "attn"):
                layer.reset()

    def _get_num_params(self):
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            if hasattr(module, "SCALE_INIT"):
                self.std *= (2 * self.num_layers) ** -0.5
            nn.init.normal_(module.weight, mean=0.0, std=self.std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.std)
        elif isinstance(module, Attention):
            nn.init.xavier_normal_(module.c_attn.weight)
            nn.init.xavier_normal_(module.c_proj.weight)
            if module.c_attn.bias is not None:
                nn.init.zeros_(module.c_attn.bias)
            if module.c_proj.bias is not None:
                nn.init.zeros_(module.c_proj.bias)
        elif isinstance(module, STU):
            if self.use_approx:
                nn.init.xavier_normal_(module.M_inputs)
                nn.init.xavier_normal_(module.M_filters)
            else:
                nn.init.xavier_normal_(module.M_phi_plus)
                if not self.use_hankel_L:
                    nn.init.xavier_normal_(module.M_phi_minus)

if __name__ == '__main__':
    config_path = "config.json"
    with open(config_path, "r") as f:
        config_data = json.load(f)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Device:", device)
    torch_dtype = getattr(torch, config_data["torch_dtype"])
    config = FlashSTUConfig(
        bsz=config_data["bsz"],
        dim=config_data["dim"],
        num_heads=config_data["num_heads"],
        num_layers=config_data["num_layers"],
        seq_len=config_data["seq_len"],
        weight_tying=config_data["weight_tying"],
        window_size=config_data["window_size"],
        vocab_size=config_data["vocab_size"],
        mlp_scale=config_data["mlp_scale"],
        bias=config_data["bias"],
        dropout=config_data["dropout"],
        num_eigh=config_data["num_eigh"],
        use_hankel_L=config_data["use_hankel_L"],
        use_flash_fft=config_data["use_flash_fft"],
        use_approx=config_data["use_approx"],
        use_attn=config_data["use_attn"],
        softcap=config_data["softcap"],
        theta=config_data["theta"],
        use_alibi=config_data["use_alibi"],
        dilation=config_data["dilation"],
        torch_dtype=torch_dtype,
        device=device
    )
    filters = get_spectral_filters(
        seq_len=config_data["seq_len"],
        K=config_data["num_eigh"],
        use_hankel_L=config_data["use_hankel_L"],
        device=device
    )
    print("Configs:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    model = FlashSTU(config, filters).to(device=device, dtype=torch_dtype)
    x = torch.randint(0, config.vocab_size, (config.bsz, config.seq_len), dtype=torch.long).to(device)
    outputs = model(x)
    print("Output shape:", outputs.shape)
    print("Sample output:", outputs[0, 0, :10])
