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

# tests/test_advanced_features.py

import torch
import pytest
from qaintum_qnn.models.quantum_neural_network import QuantumNeuralNetwork
from qaintum_qnn.utils.normalization import ZScoreNormalization, MinMaxScaling, NormalizeToRange, NormalizeToRadians

# Fixtures for reusable components
@pytest.fixture
def inputs():
    return torch.tensor([[3.14, 6.28, 9.42, 12.56]])  # Example input tensor

@pytest.fixture
def quantum_neural_network():
    return QuantumNeuralNetwork(
        num_wires=4,
        cutoff_dim=5,
        num_layers=2,
        output_size="multi",
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=False
    )

@pytest.fixture
def hybrid_model():
    class HybridModel(torch.nn.Module):
        def __init__(self, num_wires, cutoff_dim, num_layers, output_size):
            super(HybridModel, self).__init__()
            self.classical_layer = torch.nn.Linear(4, 8)
            self.normalizer = NormalizeToRadians()
            self.quantum_layer = QuantumNeuralNetwork(
                num_wires=num_wires,
                cutoff_dim=cutoff_dim,
                num_layers=num_layers,
                output_size=output_size,
                init_method="xavier",
                active_sd=0.001,
                passive_sd=0.2,
                gain=1.0,
                normalize_inputs=False
            )

        def forward(self, x):
            x = torch.relu(self.classical_layer(x))
            x = self.normalizer(x)
            x = self.quantum_layer(x)
            return x

    return HybridModel(num_wires=4, cutoff_dim=5, num_layers=2, output_size="multi")

# Test Custom Weight Initialization
def test_custom_weight_initialization():
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "multi"

    # Initialize QNN with Xavier initialization
    model_xavier = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True
    )

    # Initialize QNN with Kaiming initialization
    model_kaiming = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="kaiming",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True
    )

    assert model_xavier.init_method == "xavier"
    assert model_kaiming.init_method == "kaiming"

# Test Input Normalization Methods
def test_input_normalization(inputs):
    # Z-Score Normalization
    z_score_normalizer = ZScoreNormalization()
    z_score_normalized = z_score_normalizer(inputs)
    assert z_score_normalized.shape == inputs.shape
    assert torch.allclose(z_score_normalized.mean(dim=1), torch.zeros(1), atol=1e-6)

    # Min-Max Scaling
    min_max_scaler = MinMaxScaling(min_value=0.0, max_value=1.0)
    min_max_scaled = min_max_scaler(inputs)
    assert torch.all(min_max_scaled >= 0.0) and torch.all(min_max_scaled <= 1.0)

    # Normalize to Range [-1, 1]
    normalize_to_range = NormalizeToRange(target_min=-1.0, target_max=1.0)
    range_normalized = normalize_to_range(inputs)
    assert torch.all(range_normalized >= -1.0) and torch.all(range_normalized <= 1.0)

    # Normalize to Radians [0, 2π]
    normalize_to_radians = NormalizeToRadians()
    radians_normalized = normalize_to_radians(inputs)
    assert torch.all(radians_normalized >= 0.0) and torch.all(radians_normalized <= 2 * torch.pi)

# Test Forward Pass with Normalized Inputs
def test_forward_pass_with_normalization(quantum_neural_network, inputs):
    normalize_to_radians = NormalizeToRadians()
    radians_normalized = normalize_to_radians(inputs)
    outputs = quantum_neural_network(radians_normalized)
    assert outputs.shape == (1, 4)  # Assuming "multi" output mode

# Test Hybrid Quantum-Classical Model
def test_hybrid_model(hybrid_model, inputs):
    hybrid_outputs = hybrid_model(inputs)
    assert hybrid_outputs.shape == (1, 4)  # Assuming "multi" output mode

# Test Post-Processing Outputs
def test_post_processing_outputs(quantum_neural_network, inputs):
    outputs = quantum_neural_network(inputs)
    predicted_class = torch.argmax(outputs, dim=-1)
    assert predicted_class.item() in range(4)  # Ensure predicted class is within valid range