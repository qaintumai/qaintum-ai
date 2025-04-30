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

# tests/test_qnn_circuit.py

import pytest
import torch
import numpy as np
import pennylane as qml

from qnn.layers.qnn_data_encoder import QuantumDataEncoder  # Import QuantumDataEncoder
from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit
from qnn.utils.qnn_weight_init import QuantumWeightInitializer

@pytest.fixture
def qnn_params():
    return {
        "num_wires": 4,
        "cutoff_dim": 2,
        "num_layers": 2
    }


def generate_inputs_and_weights(qnn_params):
    """
    Helper function to generate random inputs and weights for testing.

    Parameters:
        qnn_params (dict): Dictionary containing QNN parameters.

    Returns:
        tuple: Inputs (list) and weights (torch.Tensor).
    """
    inputs = np.random.randn(2 * qnn_params["num_wires"]).tolist()
    weight_initializer = QuantumWeightInitializer()
    var = weight_initializer.init_weights(qnn_params["num_layers"], qnn_params["num_wires"])
    return inputs, var

def test_single_output(qnn_params):
    # Generate inputs
    inputs, var = generate_inputs_and_weights(qnn_params)

    # Initialize QuantumNeuralNetworkCircuit
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="single",
    )
    circuit = qnn_circuit.build_circuit()

    # Pass encoded inputs to the circuit
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    assert output.dim() == 0  # Scalar tensor

def test_multi_output(qnn_params):
     # Generate inputs and weights
    inputs, var = generate_inputs_and_weights(qnn_params)

    # Initialize QuantumNeuralNetworkCircuit
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="multi",
    )
    circuit = qnn_circuit.build_circuit()

    # Pass encoded inputs to the circuit
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    assert output.dim() == 0  # Scalar tensor

def test_multi_output(qnn_params):
    # Generate inputs and weights
    inputs, var = generate_inputs_and_weights(qnn_params)

    # Initialize QuantumNeuralNetworkCircuit
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="multi",
    )
    circuit = qnn_circuit.build_circuit()

    # Pass encoded inputs to the circuit
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    assert len(output) == qnn_params["num_wires"]

def test_probabilities_output(qnn_params):
    # Generate inputs
    inputs = np.random.randn(2*qnn_params["num_wires"]).tolist()

    # Generate weights
    weights = QuantumWeightInitializer()
    var = weights.init_weights(qnn_params["num_layers"], qnn_params["num_wires"])

    # Initialize QuantumNeuralNetworkCircuit
    qnn_circuit = QuantumNeuralNetworkCircuit(
        num_wires=qnn_params["num_wires"],
        cutoff_dim=qnn_params["cutoff_dim"],
        num_layers=qnn_params["num_layers"],
        output_size="probabilities",
    )
    circuit = qnn_circuit.build_circuit()

    # Pass encoded inputs to the circuit
    output = circuit(inputs, var)

    assert isinstance(output, torch.Tensor)
    expected_len = qnn_params["cutoff_dim"] ** qnn_params["num_wires"]
    assert len(output) == expected_len

def test_invalid_output_size(qnn_params):
    with pytest.raises(ValueError):
        QuantumNeuralNetworkCircuit(
            num_wires=qnn_params["num_wires"],
            cutoff_dim=qnn_params["cutoff_dim"],
            num_layers=qnn_params["num_layers"],
            output_size="invalid",
            encoder=QuantumDataEncoder(num_wires=qnn_params["num_wires"])
        )


