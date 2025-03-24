# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#THIS IS FROM TORCHTUNE, Cite appropriately
from typing import Tuple

import torch
from torch import nn, Tensor


class KVCache(nn.Module):
    """
    Standalone ``nn.Module`` containing a kv-cache to cache past key and values during inference.

    Args:
        batch_size (int): batch size model will be run with
        max_seq_len (int): maximum sequence length model will be run with
        num_heads (int): number of heads. We take num_heads instead of num_kv_heads because
            the cache is created after we've expanded the key and value tensors to have the
            same shape as the query tensor. See attention.py for more details
        head_dim (int): per-attention head embedding dimension
        dtype (torch.dtype): dtype for the caches
    """

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, max_seq_len, num_heads, head_dim)
        self.register_buffer(
            "k_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        self.register_buffer(
            "v_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
        # self.register_buffer(
        #     "y_cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        # )
        self.batch_size = batch_size

    def reset(self) -> None:
            """Reset the cache to zero."""
            self.k_cache.zero_()
            self.v_cache.zero_()

    def update(
            self, input_pos: Tensor, k_val: Tensor, v_val: Tensor
        ) -> Tuple[Tensor, Tensor]:
            """Update KV cache with the new k_val, v_val and return the updated cache.

            Args:
                input_pos (Tensor): Current position tensor with shape [S]
                k_val (Tensor): Current key tensor with shape [B, S, H, D]
                v_val (Tensor): Current value tensor with shape [B, S, H, D]

            Returns:
                Tuple[Tensor, Tensor]: Updated KV cache with key first
            """
                
            # Update the cache
            # print(input_pos, k_val.shape, self.k_cache.shape)
            self.k_cache[:, input_pos] = k_val
            self.v_cache[:, input_pos] = v_val
            
            
            return self.k_cache, self.v_cache
    
    # def update_y(
    #             self, y_val, input_pos
    # ):
        
    #       self.y_cache[:, input_pos] = y_val
    #       return self.y_cache