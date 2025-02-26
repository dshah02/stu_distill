import sys
import os
sys.path.append(os.path.abspath("../src"))

import torch
import torch.nn as nn
import torch.nn.functional  as  F
import math
import numpy as np

device = torch.device("cuda"  if torch.cuda.is_available() else "cpu")

def nearest_power_of_two(x: int, round_up: bool = False) -> int:
    return (
        1 << math.floor(math.log2(x)) if not round_up else 1 << math.ceil(math.log2(x))
    )

def convolve(u: torch.Tensor, v: torch.Tensor, n: int, use_approx: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    # print(u.shape, v.shape)
    bsz, seq_len, d_in = u.shape

    sgn = torch.full((1, seq_len, 1), 1, device=u.device)
    sgn[:, 1::2] *= -1

    _, K = v.shape
    sgn = sgn.unsqueeze(-1)
    v = v.view(1, -1, K, 1, 1).to(torch.float32) # (bsz, seq_len, K, d_in, stack)
    u = u.view(bsz, -1, 1, d_in).expand(bsz, -1, K, d_in)

    v = torch.fft.rfft(v, n=n, dim=1)
    U = torch.stack([u, u * sgn], dim=-1).to(torch.float32)
    U = torch.fft.rfft(U, n=n, dim=1)
    U_conv = torch.fft.irfft(v * U, n=n, dim=1)[:, :seq_len]
    U_plus, U_minus = torch.unbind(U_conv, dim=-1)
    U_minus = U_minus * sgn

    return U_plus, U_minus


def get_hankel(seq_len: int, use_hankel_L: bool = False) -> np.ndarray:
    entries = np.arange(1, seq_len + 1, dtype=np.float64)
    i_plus_j = entries[:, None] + entries[None, :]

    if use_hankel_L:
        sgn = (-1.0) ** (i_plus_j - 2.0) + 1.0
        denom = (i_plus_j + 3.0) * (i_plus_j - 1.0) * (i_plus_j + 1.0)
        Z = sgn * (8.0 / denom)
    elif not use_hankel_L:
        Z = 2.0 / (i_plus_j**3 - i_plus_j)
    else:
        raise ValueError("use_hankel_L must be a boolean")

    return Z

def get_spectral_filters(
    seq_len: int,
    K: int,
    use_hankel_L: bool = False,
    device: torch.device = None,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    # assert torch.cuda.is_available(), "CUDA is required."
    Z = get_hankel(seq_len, use_hankel_L)
    sigma, phi = np.linalg.eigh(Z)
    sigma_k, phi_k = sigma[-K:], phi[:, -K:]
    phi_k *= sigma_k ** 0.25
    filters = torch.from_numpy(phi_k)
    return filters.to(device=device, dtype=dtype)

class STU(nn.Module):
    def __init__(self, config, phi) -> None:
        super(STU, self).__init__()
        self.config = config
        self.phi = phi
        self.n = nearest_power_of_two(config['seq_len'] * 2 - 1, round_up=True)
        self.K = config['num_eigh']
        self.d_in = config['d_in']
        self.d_out = config['d_out']
        self.use_hankel_L = config['use_hankel_L']
        self.use_approx = False
        self.k_u = config['k_u']

        self.M = nn.Parameter(torch.randn(self.d_out, self.d_in, self.k_u, dtype=config['torch_dtype']) / self.d_in)
        
        self.M_phi_plus = nn.Parameter(
            torch.randn(self.K, self.d_in, self.d_out, dtype=config['torch_dtype']) / 10
        )

        self.M_phi_minus = nn.Parameter(
            torch.randn(self.K, self.d_in, self.d_out, dtype=config['torch_dtype']) / 10
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Convolve inputs and filters,
        U_plus, U_minus = convolve(x, self.phi, self.n, False)
        # Then, contract over the K and d_in dimensions

        # print(U_plus.shape, U_minus.shape)
        spectral_plus = torch.tensordot(
            U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
        )

        spectral_minus = torch.tensordot(
            U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
        )
            
        output = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
        ar = compute_ar_x_preds(self.M, x)
        return output + ar
        
    def loss(self, inputs, targets):
        pred = self.forward(inputs)
        # print(pred, targets)
        loss = F.mse_loss(pred, targets)
        return  loss


class LDS:
  def __init__(self, A: torch.tensor, B: torch.tensor, C: torch.tensor, D: torch.tensor, h0: torch.tensor):
    self.d_h = A.shape
    _, self.d_u = B.shape
    self.d_o, _ = C.shape
    self.A = A  # hidden state dynamics
    self.B = B  # hidden state dynamics
    self.C = C  # observation projection
    self.D = D  # observation projection
    self.h0 = h0  # initial hidden state
    self.h = h0  # current hidden state
    self.dtype = float

  def step(self, u: torch.tensor) -> torch.tensor:
    assert u.shape == (self.d_u,)
    h_next = self.A * self.h + self.B @ u
    obs = self.C @ h_next + self.D @ u
    self.h = h_next
    assert obs.shape == (self.d_o,)
    return obs

  def reset(self):
    self.h = self.h0
    return self

  def generate_trajectory(self, us: torch.tensor, h0: torch.tensor = None) -> torch.tensor:
    if h0 is not None:
      self.h = h0
    _, d_u = us.shape
    assert d_u == self.d_u, (d_u, self.d_u)
    obs = []
    for u in us:
      obs.append(self.step(u))
    return torch.stack(obs, dim=0)

def random_LDS(d_h: int, d_o: int, d_u: int, lower_bound: float):
  """
  makes a random LDS with hidden state dimension d_h, observation dimension d_o, and control dimension d_u.
  `lower_bound` is a float in [0, 1] specifying the minimum absolute value for entries in A.
  Each entry in A will be in [lower_bound, 1] multiplied by +/-1 with equal probability.
  """
  # Generate random values in [lower_bound, 1]
  A = torch.rand(d_h) * (1 - lower_bound) + lower_bound
  signs = torch.randint(0, 2, (d_h,)) * 2 - 1
  A = A * signs.float()
  
  B = torch.randn(d_h, d_u).to(device)
  C = torch.randn(d_o, d_h).to(device)
  D = torch.zeros(d_o, d_u).to(device)
  h0 = torch.zeros(d_h).to(device)
  return LDS(A.to(device), B, C, D, h0)

seq_len = 8192
num_eigh = 40
use_hankel_L  = True
phi= get_spectral_filters(seq_len = seq_len, K = num_eigh,  use_hankel_L= use_hankel_L,
                                device  = device,  dtype = torch.float32)

stu_config = {
    "num_eigh": num_eigh,
    "use_hankel_L": True,
    "torch_dtype": torch.float32,
    "d_in": 1,
    "d_out": 1,
    "seq_len": seq_len,
    "k_u": 10
}



def train_stu(lds, verbose = True):
    model = STU(stu_config, phi).to(device)
    lr = 1
    optimizer = torch.optim.Adagrad(model.parameters(), lr=lr)

    steps = 2000 # @param
    model.train()

    bsz = 1

    for step in range(steps):
        inputs = torch.randn(bsz * seq_len, d_in).to(device)
        targets = lds.reset().generate_trajectory(inputs)

        inputs = inputs.reshape(bsz, seq_len, d_in).to(device)
        targets = targets.reshape(bsz, seq_len, d_out).to(device)
        loss = model.loss(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0 and verbose:
            print(f"Step {step}: Loss = {loss.item()}")

    model.eval()
    return model, loss

import numpy as np
import tqdm

ldss = []
pos_weights_matrix = []
neg_weights_matrix = []
autoreg_weights_matrix = []
losses = []
store = []


for i in tqdm.tqdm(range(200)):
 
    new_lds = random_LDS(d_h = d_h, d_o = d_out, d_u = d_in, lower_bound= 0.95)  # Replace LDS with the actual class constructor if different
    
    # Fit STU to the LDS
    stu, loss = train_stu(new_lds, verbose = False)
    
    # Extract positive and negative weights
    pos_weights = stu.M_phi_plus[:, 0, 0].detach().cpu().numpy()  # Shape: [num_filters]
    neg_weights = stu.M_phi_minus[:, 0, 0].detach().cpu().numpy()  # Shape: [num_filters]
    auto_reg = stu.M.detach().cpu().numpy()
    autoreg_weights_matrix.append(auto_reg)
    
    pos_weights_matrix.append(pos_weights)
    neg_weights_matrix.append(neg_weights)
    losses.append(loss)
    store.append([new_lds, stu])

    ldss.append(new_lds)


import pickle

lds_nps = []
for lds in ldss:
    lds_np = {} 
    lds_np['A'] = lds.A.cpu().numpy()
    lds_np['B'] = lds.B.cpu().numpy()
    lds_np['C'] = lds.C.cpu().numpy()
    lds_np['D'] = lds.D.cpu().numpy()
    lds_np['h'] = lds.h.cpu().numpy()
    lds_np['h0'] = lds.h0.cpu().numpy()
    lds_nps.append(lds_np)

data = {
    "ldss": lds_nps,
    "pos_weights_matrix": pos_weights_matrix,
    "neg_weights_matrix": neg_weights_matrix,
    "autoreg_weights_matrix": autoreg_weights_matrix,
    "losses": [l.item() for l in losses],
    
    # "store": [[lds, stu.to('cpu')] for lds, stu in store]
}

with open("saved_data_5_kx10.pkl", "wb") as f:
    pickle.dump(data, f)

print("Data saved successfully!")
