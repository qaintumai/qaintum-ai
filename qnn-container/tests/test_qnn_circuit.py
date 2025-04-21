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
import pennylane as qml

from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit
from qnn.utils.quantum_weight_init import QuantumWeightInitializer

@pytest.fixture
def qnn_params():
    return {
        "num_wires": 4,
        "cutoff_dim": 2,
        "num_layers": 2
    }

def generate_inputs(num_wires, num_layers):
    inputs = torch.randn(8 * num_wires - 2).tolist()
    weights = QuantumWeightInitializer()
    var = weights.init_weights(num_layers, num_wires)
    return inputs, var

def test_single_output(qnn_params):
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="single"
    )
    circuit = qnn_circuit.build_circuit()
    inputs, var = generate_inputs(qnn_params["num_wires"], qnn_params["num_layers"])
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    assert output.dim() == 0  # Scalar tensor

def test_multi_output(qnn_params):
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="multi"
    )
    circuit = qnn_circuit.build_circuit()
    inputs, var = generate_inputs(qnn_params["num_wires"], qnn_params["num_layers"])
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    assert len(output) == qnn_params["num_wires"]

def test_probabilities_output(qnn_params):
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="probabilities"
    )
    circuit = qnn_circuit.build_circuit()
    inputs, var = generate_inputs(qnn_params["num_wires"], qnn_params["num_layers"])
    output = circuit(inputs, var)

    expected_len = qnn_params["cutoff_dim"] ** qnn_params["num_wires"]
    assert len(output) == expected_len

def test_invalid_output_size(qnn_params):
    with pytest.raises(ValueError):
        QuantumNeuralNetworkCircuit(
            num_wires=qnn_params["num_wires"],
            cutoff_dim=qnn_params["cutoff_dim"],
            num_layers=qnn_params["num_layers"],
            output_size="invalid"
        )
