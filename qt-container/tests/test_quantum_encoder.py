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
from qt.models import QuantumDecoder
from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit

@pytest.fixture
def decoder_block_setup():
    """
    Fixture to set up parameters and create a QuantumDecoder instance.
    """
    embed_len = 64
    num_heads = 8
    num_layers = 2
    num_wires = 6
    quantum_nn = qnn_circuit
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    mask = None

    # Create an instance of QuantumDecoder
    model = QuantumDecoder(
        embed_len=embed_len,
        num_heads=num_heads,
        num_layers=num_layers,
        num_wires=num_wires,
        quantum_nn=quantum_nn,
        batch_size=batch_size,
        dropout=dropout,
        mask=mask,
    )

    # Create dummy input tensors
    target = torch.rand(batch_size, seq_len, embed_len)
    encoder_output = torch.rand(batch_size, seq_len, embed_len)

    return {
        "model": model,
        "target": target,
        "encoder_output": encoder_output,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "embed_len": embed_len,
    }


def test_decoder_block_output_shape(decoder_block_setup):
    """
    Test that the output shape of the QuantumDecoder is as expected.
    """
    model = decoder_block_setup["model"]
    target = decoder_block_setup["target"]
    encoder_output = decoder_block_setup["encoder_output"]

    # Forward pass
    output = model(target, encoder_output)

    # Check the output shape
    expected_shape = (
        decoder_block_setup["batch_size"],
        decoder_block_setup["seq_len"],
        decoder_block_setup["embed_len"],
    )
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"


def test_decoder_block_output_type(decoder_block_setup):
    """
    Test that the output type of the QuantumDecoder is torch.Tensor.
    """
    model = decoder_block_setup["model"]
    target = decoder_block_setup["target"]
    encoder_output = decoder_block_setup["encoder_output"]

    # Forward pass
    output = model(target, encoder_output)

    # Check the output type
    assert isinstance(output, torch.Tensor), f"Expected torch.Tensor, but got {type(output)}"


def test_decoder_block_integration(decoder_block_setup):
    """
    Integration test to ensure the QuantumDecoder works as a whole.
    """
    model = decoder_block_setup["model"]
    target = decoder_block_setup["target"]
    encoder_output = decoder_block_setup["encoder_output"]

    # Forward pass
    output = model(target, encoder_output)

    # Validate both shape and type
    expected_shape = (
        decoder_block_setup["batch_size"],
        decoder_block_setup["seq_len"],
        decoder_block_setup["embed_len"],
    )
    assert output.shape == expected_shape, f"Expected {expected_shape}, but got {output.shape}"
    assert isinstance(output, torch.Tensor), f"Expected torch.Tensor, but got {type(output)}"