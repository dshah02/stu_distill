import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import sys
import os
sys.path.append(os.path.abspath("../../src"))

# from lds import LDS
# from inference_lds import LDS
from nlds import NLDS as LDS

class FullFastSTU(nn.Module):
    def __init__(self, stu, name=None) -> None:
        super(FullFastSTU, self).__init__()
        
        stu = copy.deepcopy(stu)

        self.config = stu.config
        self.K = stu.config.num_eigh
        self.d_in = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.d_out = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.use_hankel_L = stu.config.use_hankel_L
        self.use_approx = stu.config.use_approx
        
        if name is not None:
            self.lds = self.get_lds(name)
        else:
            self.lds = self.get_lds()
        
        if self.use_approx:
            self.M_inputs = nn.Parameter(
                stu.M_inputs.data.to(self.lds.dtype)
            )
            # self.M_filters has shape [K, d_in]. Note: Do not slice here.
            self.M_filters = stu.M_filters.data.to(self.lds.dtype)
        else:
            self.M_phi_plus = nn.Parameter(
                stu.M_phi_plus.data.to(self.lds.dtype)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    stu.M_phi_minus.data.to(self.lds.dtype)
                )
    
    @torch.compile
    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        if self.use_approx:
            x = x.to(self.lds.dtype) @ self.M_inputs
            
            bsz = x.shape[0]
            # Reshape to [B*d_in, L, 1]
            x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1], 1)
            U_reshaped = self.lds(x_reshaped)
            # Reshape U to [B, L, K_total, d_in]
            U = U_reshaped.reshape(bsz, x.shape[2], x.shape[1], -1).permute(0, 2, 3, 1)
            
            # Option 1: Use original einsum (unmodified)
            # spectral_plus = torch.einsum('blkd,kd->bld', U[:, :, self.K:, :], self.M_filters)
            # spectral_minus = torch.einsum('blkd,kd->bld', U[:, :, :self.K, :], self.M_filters)
            
            # Option 2: Use broadcast & sum instead (should be equivalent)
            spectral_plus = (U[:, :, self.K:, :] * self.M_filters[None, None, :, :]).sum(dim=2)
            spectral_minus = (U[:, :, :self.K, :] * self.M_filters[None, None, :, :]).sum(dim=2)
            
            ret = spectral_plus if self.use_hankel_L else (spectral_plus + spectral_minus)
            # Return in bfloat16 (as originally)
            return ret.to(torch.bfloat16)
        else:
            x = x.double()
            bsz = x.shape[0]
            x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1], 1)
            U_reshaped = self.lds(x_reshaped)
            U = U_reshaped.reshape(bsz, x.shape[2], x.shape[1], -1).permute(0, 2, 3, 1)
            U_plus, U_minus = U[:, :, :self.K, :], U[:, :, self.K:, :]
            
            spectral_plus = torch.tensordot(U_plus, self.M_phi_plus, dims=([2, 3], [0, 1]))
            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(U_minus, self.M_phi_minus, dims=([2, 3], [0, 1]))
            ret = spectral_plus if self.use_hankel_L else (spectral_plus + spectral_minus)
            return ret.to(torch.bfloat16)

    def loss(self, inputs, targets):
        pred = self.forward(inputs, input_pos=None)
        return F.mse_loss(pred, targets)

    def get_lds(self, checkpoint_path='./experiments/convex_hull/best_phi_lds.pt'):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        lds_phi = LDS(
            state_dim=checkpoint['state_dim'],
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            kx=checkpoint['kx'],
            dtype=torch.float32 if checkpoint['dtype'] == 'torch.float32' else torch.float64,
        )
        lds_phi.load_state_dict(checkpoint['model_state_dict'], strict=False)
        return lds_phi.cuda()

