import torch
import sys
import os
sys.path.append(os.path.abspath("./src"))
sys.path.append(os.path.abspath("./experiments/convex_hull"))

from model_550m import get_spectral_filters, STU
from full_fast_stu import FullFastSTU

class STUConfig():
    def __init__(
        self,
        d_in=768,
        d_out=768,
        num_eigh=24,
        seq_len=8192,
        use_hankel_L=False,
        use_approx=True,
        use_flash_fft = True,
        torch_dtype = torch.bfloat16
    ):
        super().__init__()
        self.n_embd = d_in  # Used by some parts of the code
        self.dim = d_in     # Used by other parts of the code
        self.d_out = d_out
        self.num_eigh = num_eigh
        self.seq_len = seq_len
        self.use_hankel_L = use_hankel_L
        self.use_approx = use_approx
        self.use_flash_fft = use_flash_fft
        self.torch_dtype = torch_dtype

def create_random_stu(d_in=1, d_out=1, num_eigh=24, use_hankel_L=False, use_approx=True):
    # Create a random config
 
    filters = get_spectral_filters(
        seq_len=8192,
        K=num_eigh,
        use_hankel_L=use_hankel_L,
        device=torch.device('cuda'),
        dtype=torch.bfloat16
    )
    
    # Create random STU
    config = STUConfig()
    stu = STU(config, filters).cuda()
    return stu

def test_full_fast_stu():
    # Create random STU
    stu = create_random_stu()
   
    seq_len = 8192    
    # Generate random input
    batch_size = 4
    d_in = stu.config.n_embd
    x = torch.randn(batch_size, seq_len, d_in, dtype=torch.bfloat16).cuda()
    input_pos = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).cuda()
    
    full_fast_stu = FullFastSTU(stu)
    # Get outputs from both models
    with torch.no_grad():
        stu_output = stu(x, input_pos)
        full_fast_output = full_fast_stu(x, input_pos)
    
    # Calculate differences
    abs_diff = torch.abs(stu_output - full_fast_output)
    mean_diff = abs_diff.mean().item()
    max_diff = abs_diff.max().item()
    

    print(f"Mean absolute difference: {mean_diff:.6f}")
    print(f"Max absolute difference: {max_diff:.6f}")
    
    # Check if outputs are close
    is_close = torch.allclose(stu_output, full_fast_output, rtol=1e-5, atol=1e-5)
    print(f"Outputs are close (within rtol=1e-5, atol=1e-5): {is_close}")

if __name__ == "__main__":
    test_full_fast_stu() 