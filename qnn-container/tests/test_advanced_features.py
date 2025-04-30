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
import torch.nn as nn
from qnn.models.quantum_neural_network import QuantumNeuralNetwork

@pytest.fixture
def qnn_params():
    return {
        "num_wires": 4,
        "cutoff_dim": 5,
        "num_layers": 2,
        "output_size": "probabilities",
        "active_sd": 0.001,
        "passive_sd": 0.2,
        "gain": 1.0
    }

def test_custom_weight_initialization(qnn_params):
    model_xavier = QuantumNeuralNetwork(init_method="xavier", normalize_inputs=True, **qnn_params)
    model_kaiming = QuantumNeuralNetwork(init_method="kaiming", normalize_inputs=True, **qnn_params)

    assert model_xavier is not None
    assert model_kaiming is not None

def test_hybrid_model_forward(qnn_params):
    class HybridModel(nn.Module):
        def __init__(self):
            super(HybridModel, self).__init__()
            self.classical_layer = nn.Linear(4, 8)
            self.quantum_layer = QuantumNeuralNetwork(
                **qnn_params, normalize_inputs=True, init_method="xavier"
            )

        def forward(self, x):
            x = torch.relu(self.classical_layer(x))
            return self.quantum_layer(x)

    hybrid_model = HybridModel()
    inputs = torch.tensor([[3.14, 6.28, 9.42, 12.56]])
    outputs = hybrid_model(inputs)

    assert outputs is not None
    assert outputs.shape[0] == inputs.shape[0]

def test_post_processing_output(qnn_params):
    inputs = torch.tensor([[3.14, 6.28, 9.42, 12.56]])
    model = QuantumNeuralNetwork(init_method="xavier", normalize_inputs=True, **qnn_params)
    outputs = model(inputs)

    predicted_class = torch.argmax(outputs, dim=-1)
    assert predicted_class.item() >= 0
    assert predicted_class.shape == torch.Size([1])
