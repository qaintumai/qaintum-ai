# Copyright 2025 The qAIntum.ai Authors. All Rights Reserved.
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
from qnn.models.quantum_neural_network import QuantumNeuralNetwork

@pytest.fixture
def qnn_model():
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "single"
    model = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size
    )
    return model, num_wires, cutoff_dim, num_layers, output_size

def test_initialization(qnn_model):
    model, num_wires, cutoff_dim, num_layers, output_size = qnn_model
    assert model.num_wires == num_wires
    assert model.cutoff_dim == cutoff_dim
    assert model.num_layers == num_layers
    assert model.output_size == output_size

def test_trainable_parameters(qnn_model):
    model, num_wires, _, num_layers, _ = qnn_model
    trainable_params = list(model.parameters())

    assert len(trainable_params) > 0
    expected_shape = (num_layers, 9 * num_wires - 4)
    assert trainable_params[0].shape == expected_shape

def test_forward_pass(qnn_model):
    model, num_wires, cutoff_dim, _, output_size = qnn_model
    batch_size = 3
    inputs = torch.randn(batch_size, num_wires)
    outputs = model(inputs)

    if output_size == "single":
        expected_shape = (batch_size,)
    elif output_size == "multi":
        expected_shape = (batch_size, num_wires)
    else:  # probabilities
        expected_shape = (batch_size, cutoff_dim ** num_wires)

    assert outputs.shape == expected_shape

@pytest.mark.parametrize("invalid_input", [
    [1, 2, 3, 4],
    "invalid",
    None
])
def test_invalid_input_type(qnn_model, invalid_input):
    model, *_ = qnn_model
    with pytest.raises(TypeError):
        model(invalid_input)

def test_gradient_computation(qnn_model):
    model, num_wires, _, _, _ = qnn_model
    batch_size = 3
    inputs = torch.randn(batch_size, num_wires)
    outputs = model(inputs)

    loss = outputs.mean()
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None
