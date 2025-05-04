# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import time
import pytest
import torch
from qaintum_qe.layers.scaled_dot_product import ScaledDotProduct

@pytest.fixture
def scaled_dot_product_model(embed_len=64):
    """Fixture to create an instance of ScaledDotProduct."""
    return ScaledDotProduct(embed_len)

@pytest.fixture
def dummy_inputs(batch_size=32, seq_len=10, embed_len=64):
    """Fixture to create dummy input tensors."""
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)
    return queries, keys, values

def test_scaled_dot_product(scaled_dot_product_model, dummy_inputs):
    """
    Test the ScaledDotProduct model with standard inputs.
    """
    queries, keys, values = dummy_inputs

    # Forward pass
    start_time = time.time()
    output = scaled_dot_product_model(queries, keys, values)
    elapsed_time = time.time() - start_time

    # Check the output shape
    expected_shape = queries.shape
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {output.shape}"
    )

    # Check the output type
    assert isinstance(output, torch.Tensor), (
        f"Expected output type torch.Tensor, but got {type(output)}"
    )

    # Check performance: Assert the forward pass is reasonably fast
    assert elapsed_time < 1.0, (
        f"Forward pass took too long: {elapsed_time:.4f} seconds"
    )

    print("Test passed!")

@pytest.mark.parametrize(
    "batch_size, seq_len, embed_len",
    [
        (1, 1, 1),  # Edge case: Small tensors
        (64, 1000, 512),  # Edge case: Large tensors
    ]
)
def test_edge_cases(batch_size, seq_len, embed_len):
    """
    Test edge cases for small and large tensors.
    """
    # Create an instance of ScaledDotProduct
    model = ScaledDotProduct(embed_len)

    # Create dummy input tensors
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    # Forward pass
    output = model(queries, keys, values)

    # Check the output shape
    expected_shape = (batch_size, seq_len, embed_len)
    assert output.shape == expected_shape, (
        f"Expected output shape {expected_shape}, but got {output.shape}"
    )

    print(f"Edge case for batch_size={batch_size}, seq_len={seq_len}, embed_len={embed_len} passed!")

