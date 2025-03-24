# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#THIS IS FROM TORCHTUNE, Cite appropriately
from typing import Tuple

import torch
from torch import nn, Tensor


class Cache(nn.Module):
   

    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        dim: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        cache_shape = (batch_size, max_seq_len, dim)
        self.register_buffer(
            "cache", torch.zeros(cache_shape, dtype=dtype), persistent=False
        )
  
        self.batch_size = batch_size

    def reset(self) -> None:
            """Reset the cache to zero."""
            self.cache.zero_()


    def update(
            self, x: Tensor, input_pos: Tensor, 
        ) -> Tuple[Tensor, Tensor]:    
            # print(input_pos, x.shape, self.cache.shape)
            self.cache[:, input_pos] = x
        
            return self.cache