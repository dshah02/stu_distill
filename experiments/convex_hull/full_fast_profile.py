import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import time
from typing import Optional, Tuple, Union, Dict, List
from contextlib import contextmanager
from collections import defaultdict


class ProfilingTimer:
    """Utility class for function profiling with context management."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all timing data."""
        self.timings = defaultdict(list)
        self.call_counts = defaultdict(int)
        self.currently_running = set()
    
    @contextmanager
    def timed(self, name):
        """Context manager to time a code block."""
        # Check if this timer is already running (for nested calls)
        nested = name in self.currently_running
        
        # Track that we're running this timer
        self.currently_running.add(name)
        
        # Record start time
        start = time.time()
        try:
            # Execute the context block
            yield
        finally:
            # Only record the timing if this wasn't a nested call
            if not nested:
                # Calculate duration and record it
                duration = time.time() - start
                self.timings[name].append(duration)
                self.call_counts[name] += 1
            
            # Remove from currently running set
            self.currently_running.remove(name)
    
    def summary(self, sort_by="total"):
        """
        Return a formatted summary of timing data.
        
        Args:
            sort_by: How to sort results - "total", "average", or "calls"
        """
        results = []
        
        for name, times in self.timings.items():
            total_time = sum(times)
            avg_time = total_time / len(times)
            call_count = self.call_counts[name]
            
            results.append({
                "name": name,
                "total": total_time,
                "average": avg_time,
                "calls": call_count,
                "percent": 0.0  # Will be calculated after sorting
            })
        
        # Sort results based on specified criteria
        if sort_by == "total":
            results.sort(key=lambda x: x["total"], reverse=True)
        elif sort_by == "average":
            results.sort(key=lambda x: x["average"], reverse=True)
        elif sort_by == "calls":
            results.sort(key=lambda x: x["calls"], reverse=True)
        
        # Calculate percentages based on total time of the first entry (assumed to be the full process)
        if results:
            total_process_time = results[0]["total"] if sort_by == "total" else sum(r["total"] for r in results)
            for result in results:
                result["percent"] = (result["total"] / total_process_time) * 100
        
        return results


class NLDS(nn.Module):
    def __init__(
        self,
        state_dim: int,
        input_dim: int,
        output_dim: int,
        kx: int = 5,
        dtype: torch.dtype = torch.float64,
        bsz_dim: int = 896,
    ):
        """
        Non-Autoregressive Linear Dynamical System optimized for STU inference.
        """
        super().__init__()
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kx = kx
        self.dtype = dtype
        self.bsz_dim = bsz_dim
        self.cache = False
        
        # Parameters
        self.A = nn.Parameter(torch.randn(state_dim).to(dtype))
        self.B = nn.Parameter((torch.randn(input_dim, state_dim) / input_dim).to(dtype))
        self.C = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))
        self.h0 = nn.Parameter(torch.zeros(state_dim, dtype=dtype))
        
        self.BC = nn.Parameter((torch.randn(state_dim, output_dim) / state_dim).to(dtype))
        # Pre-allocate hidden state buffer
        self.h = self.h0.unsqueeze(0).expand(bsz_dim, -1).clone().cuda()
        
        # Profiling timer
        self.profiler = ProfilingTimer()

    def setup(self):
        self.BC.data = self.C.data * self.B.data.reshape(-1, 1)

    def reset_state(self, batch_size=None):
        """Reset the hidden state for a new sequence."""
        with self.profiler.timed("reset_state"):
            self.cache = False
            if batch_size is None:
                batch_size = self.bsz_dim
                
            device = self.A.device
            self.h = self.h0.unsqueeze(0).expand(batch_size, -1).clone().to(device)
    
    def next_step(self, x_t):
        """Single-step update for inference."""
        with self.profiler.timed("next_step_overall"):
            with self.profiler.timed("next_step_update_h"):
                self.h = self.h * self.A + x_t
                
            with self.profiler.timed("next_step_compute_output"):
                return self.h.matmul(self.BC)
    
    def forward(self, inputs):
        """Forward pass for sequence processing."""
        with self.profiler.timed("lds_forward"):
            batch_size, seq_len, _ = inputs.size()
            
            # Reset state at beginning of sequence if not cached
            if not self.cache:
                self.reset_state(batch_size)
            
            # Process sequence
            with self.profiler.timed("lds_sequence_processing"):
                # Pre-allocate output tensor
                outputs = torch.empty((batch_size, seq_len, self.output_dim), 
                                   dtype=inputs.dtype, 
                                   device=inputs.device)
                for t in range(seq_len):
                    outputs[:, t] = self.next_step(inputs[:, t])
                
            with self.profiler.timed("lds_output_concat"):
                return outputs


class FullFastSTU(nn.Module):
    """
    Heavily optimized FullFastSTU for inference with use_approx=True.
    Includes profiling capabilities for performance analysis.
    """
    def __init__(self, stu, checkpoint_path=None):
        super(FullFastSTU, self).__init__()
        
        # Create profiler - make sure this is initialized first
        self.profiler = ProfilingTimer()
        
        stu = copy.deepcopy(stu)
        self.config = stu.config
        self.K = stu.config.num_eigh
        self.d_in = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.d_out = stu.config.n_embd if hasattr(stu.config, "n_embd") else stu.config.dim
        self.use_hankel_L = stu.config.use_hankel_L
        
        # We know use_approx is True in practice
        self.use_approx = True
        
        # Load LDS
        self.lds = self._get_lds(checkpoint_path)
        
        # Store weights
        self.M_inputs = nn.Parameter(stu.M_inputs.data.to(torch.float64))
        self.register_buffer('M_filters', stu.M_filters.data.to(torch.float64))
        
        # Precompute some values for efficiency
        self.half_size = 24  # Hardcoded value from the original code
        
        # Store profiling results
        self.profile_results = None
        self.profile_runs = 0

    def forward(self, x, input_pos=None):
        """Forward pass with profiling."""
        with self.profiler.timed("forward_total"):
            # Convert to double precision
            with self.profiler.timed("convert_input"):
                x = x.to(torch.float64)
            
            # Matrix multiplication with inputs
            with self.profiler.timed("M_inputs_matmul"):
                x = x @ self.M_inputs
            
            # Reshape for LDS processing
            with self.profiler.timed("reshape_for_lds"):
                bsz = x.shape[0]
                x_reshaped = x.permute(0, 2, 1).reshape(-1, x.shape[1], 1)
            
            # Process through LDS
            with self.profiler.timed("lds_processing"):
                U_reshaped = self.lds(x_reshaped)
            
            # Apply filters and concatenate
            with self.profiler.timed("apply_filters"):
                U_plus = U_reshaped[:, :, :self.half_size] @ self.M_filters
                U_minus = U_reshaped[:, :, self.half_size:] @ self.M_filters
                U_combined = torch.cat([U_plus, U_minus], dim=-1)
            
            # Reshape to original dimensions
            with self.profiler.timed("reshape_to_original"):
                U = U_combined.reshape(bsz, x.shape[2], x.shape[1], -1).permute(0, 2, 3, 1)
            
            # Extract spectral components
            with self.profiler.timed("extract_spectral"):
                spectral_plus = U[:,:,:self.d_out,:]
                spectral_minus = U[:,:,self.d_out:,:]
                
                # Extract diagonal terms
                spectral_plus_diag = torch.diagonal(spectral_plus, dim1=2, dim2=3)
                spectral_minus_diag = torch.diagonal(spectral_minus, dim1=2, dim2=3)
            
            # Combine based on configuration
            with self.profiler.timed("combine_results"):
                if self.use_hankel_L:
                    result = spectral_plus_diag
                else:
                    result = spectral_plus_diag + spectral_minus_diag
                    
                # Convert to bfloat16 for memory efficiency
                result = result.to(torch.bfloat16)
                
            return result

    def _get_lds(self, checkpoint_path=None):
        """Initialize the LDS component."""
        with self.profiler.timed("get_lds"):
            if checkpoint_path is None:
                checkpoint_path = './experiments/convex_hull/best_phi_lds.pt'
                
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            
            lds = NLDS(
                state_dim=checkpoint['state_dim'],
                input_dim=checkpoint['input_dim'],
                output_dim=checkpoint['output_dim'],
                kx=checkpoint['kx'],
                dtype=torch.float64,
            )
            
            lds.load_state_dict(checkpoint['model_state_dict'], strict=False)
            lds.setup()
            
            return lds.cuda()

    def profile(self, input_tensor=None, runs=10, warmup=3, input_shape=(2, 128, 768)):
        """
        Profile the model to identify bottlenecks.
        
        Args:
            input_tensor: Optional tensor to use for profiling. If None, a random tensor is created.
            runs: Number of profiling runs to perform
            warmup: Number of warmup runs before profiling
            input_shape: Shape of random input tensor if input_tensor is None
            
        Returns:
            Profile results summary
        """
        # Reset profilers
        self.profiler.reset()
        self.lds.profiler.reset()
        
        # Create input tensor if not provided
        if input_tensor is None:
            input_tensor = torch.randn(*input_shape, device="cuda")
        
        # Perform warmup runs
        print(f"Performing {warmup} warmup runs...")
        with torch.no_grad():
            for _ in range(warmup):
                _ = self(input_tensor)
        
        # Perform profiling runs
        print(f"Performing {runs} profiling runs...")
        with torch.no_grad():
            torch.cuda.synchronize()  # Ensure GPU operations are complete
            for _ in range(runs):
                self.lds.reset_state(input_tensor.shape[0] * 896)
                self.lds.cache = True
                for t in range(input_tensor.shape[1]):
                    _ = self(input_tensor[:, t].unsqueeze(1))
                torch.cuda.synchronize()  # Ensure GPU operations are complete
        
        # Store results
        self.profile_results = {
            "model": self.profiler.summary(),
            "lds": self.lds.profiler.summary()
        }
        self.profile_runs = runs
        
        # Print summary
        self._print_profile_summary()
        
        return self.profile_results
    
    def _print_profile_summary(self):
        """Print a formatted summary of profiling results."""
        if not self.profile_results:
            print("No profiling results available. Run profile() first.")
            return
        
        def print_section(title, results):
            print(f"\n{title}")
            print("-" * 80)
            print(f"{'Operation':<30} {'Time (ms)':<12} {'% of Total':<12} {'Calls':<8} {'Avg (ms)':<12}")
            print("-" * 80)
            
            for r in results:
                name = r["name"]
                total_ms = r["total"] * 1000
                percent = r["percent"]
                calls = r["calls"]
                avg_ms = r["average"] * 1000
                
                print(f"{name:<30} {total_ms:<12.3f} {percent:<12.2f} {calls:<8} {avg_ms:<12.3f}")
        
        print(f"\n{'='*40} PROFILING RESULTS {'='*40}")
        print(f"Runs: {self.profile_runs}")
        
        print_section("FULL MODEL OPERATIONS", self.profile_results["model"])
        print_section("LDS OPERATIONS", self.profile_results["lds"])
        
        # Calculate percentages for complete model
        model_times = {r["name"]: r["total"] for r in self.profile_results["model"]}
        total_time = model_times["forward_total"]
        
        # Show LDS as percentage of total
        if "lds_processing" in model_times:
            lds_percent = (model_times["lds_processing"] / total_time) * 100
            print(f"\nLDS Processing: {lds_percent:.2f}% of total forward pass time")
        
        print(f"\n{'='*90}")


def process_batch(model, inputs, batch_size=32, profile=False):
    """
    Process inputs in batches with optional profiling.
    
    Args:
        model: The FullFastSTU model
        inputs: Input tensor
        batch_size: Batch size for processing
        profile: Whether to profile the first batch
        
    Returns:
        Processed output tensor
    """
    total_size = inputs.size(0)
    outputs = []
    
    with torch.no_grad():
        for i, start_idx in enumerate(range(0, total_size, batch_size)):
            end_idx = min(start_idx + batch_size, total_size)
            batch = inputs[start_idx:end_idx]
            
            # Profile the first batch if requested
            if profile and i == 0:
                # Run profiling
                model.profile(input_tensor=batch, runs=5, warmup=2)
                
            # Process batch
            output = model(batch)
            outputs.append(output)
    
    # Concatenate all batch outputs
    return torch.cat(outputs, dim=0)


def create_optimized_stu(stu, checkpoint_path=None):
    """Create an optimized FullFastSTU model from an existing STU model."""
    return FullFastSTU(stu, checkpoint_path).cuda()