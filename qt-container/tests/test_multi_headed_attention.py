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

import pytest
import torch
from qaintum_qt.layers.multi_headed_attention import MultiHeadedAttention


@pytest.fixture
def multi_head_attention():
    """
    Fixture to create a MultiHeadedAttention instance and sample inputs.
    """
    num_heads = 8
    embed_len = 64
    seq_len = 10
    batch_size = 32
    mask = None

    # Create an instance of MultiHeadedAttention
    attention_layer = MultiHeadedAttention(
        num_heads=num_heads, embed_len=embed_len, mask=mask
    )

    # Create sample inputs for queries, keys, and values
    queries = torch.rand(batch_size, seq_len, embed_len)
    keys = torch.rand(batch_size, seq_len, embed_len)
    values = torch.rand(batch_size, seq_len, embed_len)

    return {
        "attention_layer": attention_layer,
        "queries": queries,
        "keys": keys,
        "values": values,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_len": embed_len,
    }


def test_output_shape(multi_head_attention):
    """
    Test that the output shape of the MultiHeadedAttention layer is as expected.
    """
    output = multi_head_attention["attention_layer"](
        multi_head_attention["queries"],
        multi_head_attention["keys"],
        multi_head_attention["values"],
    )
    expected_shape = (
        multi_head_attention["batch_size"],
        multi_head_attention["seq_len"],
        multi_head_attention["embed_len"],
    )
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"


def test_attention_with_different_seq_len(multi_head_attention):
    """
    Test that the MultiHeadedAttention layer can handle different sequence lengths.
    """
    new_seq_len = 20
    queries = torch.rand(
        multi_head_attention["batch_size"], new_seq_len, multi_head_attention["embed_len"]
    )
    keys = torch.rand(
        multi_head_attention["batch_size"], new_seq_len, multi_head_attention["embed_len"]
    )
    values = torch.rand(
        multi_head_attention["batch_size"], new_seq_len, multi_head_attention["embed_len"]
    )

    output = multi_head_attention["attention_layer"](queries, keys, values)
    expected_shape = (
        multi_head_attention["batch_size"],
        new_seq_len,
        multi_head_attention["embed_len"],
    )
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"


def test_attention_masking(multi_head_attention):
    """
    Test that masking functionality works as expected (if implemented).
    """
    # Initialize with masking enabled (assuming masking can be handled in your implementation)
    masked_attention = MultiHeadedAttention(
        num_heads=multi_head_attention["attention_layer"].num_heads,
        embed_len=multi_head_attention["embed_len"],
        mask=True,
    )

    output = masked_attention(
        multi_head_attention["queries"],
        multi_head_attention["keys"],
        multi_head_attention["values"],
    )
    expected_shape = (
        multi_head_attention["batch_size"],
        multi_head_attention["seq_len"],
        multi_head_attention["embed_len"],
    )
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"


def test_forward_invalid_input(multi_head_attention):
    """
    Test that the MultiHeadedAttention layer raises an error when given invalid inputs.
    """
    # Provide mismatched dimensions for queries, keys, values
    queries = torch.randn(
        multi_head_attention["batch_size"],
        multi_head_attention["seq_len"] + 2,
        multi_head_attention["embed_len"],
    )
    keys = torch.randn(
        multi_head_attention["batch_size"],
        multi_head_attention["seq_len"],
        multi_head_attention["embed_len"],
    )
    values = torch.randn(
        multi_head_attention["batch_size"],
        multi_head_attention["seq_len"],
        multi_head_attention["embed_len"],
    )

    with pytest.raises(RuntimeError):
        multi_head_attention["attention_layer"](queries, keys, values)