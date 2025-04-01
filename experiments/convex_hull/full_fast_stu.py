import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

import sys
import os
sys.path.append(os.path.abspath("../../src"))

# from lds import LDS
from inference_lds import LDS

class FullFastSTU(nn.Module):
    def __init__(self, stu, name = None) -> None:
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
                stu.M_inputs.data.to(torch.float64)
            )

            # Stack the weights along the first axis to match dimensions
            M_filters = stu.M_filters.data.to(torch.float64)
            # Split the LDS.C data into two parts (for plus and minus)
            C_plus = self.lds.C.data[:, :24]  # First 24 coordinates
            C_minus = self.lds.C.data[:, 24:] # Second 24 coordinates

            self.M_filters = M_filters
            
            # # Apply M_filters to each part separately
            C_plus_transformed = C_plus @ M_filters
            C_minus_transformed = C_minus @ M_filters
            
            # # Concatenate the results to get shape [1149, 2*d_in]
            self.lds.C.data = torch.cat([C_plus_transformed, C_minus_transformed], dim=1)
            
            # Similarly for M data
            M_plus = self.lds.M.data[:24]     # First 24 coordinates
            M_minus = self.lds.M.data[24:]    # Second 24 coordinates
            
            # Apply M_filters to each part
            M_plus_transformed = torch.einsum('nik,nd->dik', M_plus, M_filters)
            M_minus_transformed = torch.einsum('nik,nd->dik', M_minus, M_filters)
            
            # Concatenate the results
            self.lds.M.data = torch.cat([M_plus_transformed, M_minus_transformed], dim=0)
        else:
            self.M_phi_plus = nn.Parameter(
                stu.M_phi_plus.data.to(torch.float64)
            )
            if not self.use_hankel_L:
                self.M_phi_minus = nn.Parameter(
                    stu.M_phi_minus.data.to(torch.float64)
                )

    def forward(self, x: torch.Tensor, input_pos: torch.Tensor) -> torch.Tensor:
        x = x.double()
        if self.use_approx:
            # Contract inputs and filters over the K and d_in dimensions, then convolve
            x = x @ self.M_inputs
        
        # Convolve inputs and filters,
        bsz = x.shape[0]
        x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1], 1)  # [B*d_in, L, 1]

        #NEW IDEA and we don't change the LDS
        # if self.use_approx:
        #     # print(self.lds(x_reshaped)).shape
        #     U_reshaped = self.lds(x_reshaped)
        #     U_reshaped = torch.stack([U_reshaped[:, :, :24] @ self.M_filters, U_reshaped[:, :, 24:] @ self.M_filters], dim = -1)
        #     print("FIRST", U_reshaped.shape)# [B*d_in, L, 2 * d_in]
            
        # else:    
        #     U_reshaped = self.lds(x_reshaped)  # [B*d_in, L, K]

        U_reshaped = self.lds(x_reshaped)  # [B*d_in, L, K]
        
        U = U_reshaped.reshape(bsz, x.shape[2], x.shape[1], -1).permute(0, 2, 3, 1)
        print("SECOND", U.shape)
        if self.use_approx:
            
            spectral_plus, spectral_minus = U[:,:,:self.d_out,:], U[:,:,self.d_out:,:]
            # Extract diagonal terms to convert from [bsz, s_len, n_eigh, n_eigh] to [bsz, s_len, n_eigh]
            spectral_plus = torch.diagonal(spectral_plus, dim1=2, dim2=3)
            spectral_minus = torch.diagonal(spectral_minus, dim1=2, dim2=3)
        
        else:
            U_plus, U_minus = U[:,:,:self.K,:], U[:,:,self.K:,:]

            # Then, contract over the K and d_in dimensions
            spectral_plus = torch.tensordot(
                U_plus, self.M_phi_plus, dims=([2, 3], [0, 1])
            )

            if not self.use_hankel_L:
                spectral_minus = torch.tensordot(
                    U_minus, self.M_phi_minus, dims=([2, 3], [0, 1])
                )
        
        ret = spectral_plus if self.use_hankel_L else spectral_plus + spectral_minus
        return ret.to(torch.bfloat16)

    def loss(self, inputs, targets):
        pred = self.forward(inputs)
        loss = F.mse_loss(pred, targets)
        return loss

    def get_lds(self,checkpoint_path = './experiments/convex_hull/best_phi_lds.pt'):
        # Load the LDS model from the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        
        
        # Create the LDS model
        lds_phi = LDS(
            state_dim=checkpoint['state_dim'],
            input_dim=checkpoint['input_dim'],
            output_dim=checkpoint['output_dim'],
            kx=checkpoint['kx'],
            dtype=torch.float32 if checkpoint['dtype'] == 'torch.float32' else torch.float64,
            
        )
        
        
        # Load the weights from checkpoint
        lds_phi.load_state_dict(checkpoint['model_state_dict'], strict = False)
        return lds_phi.cuda()
