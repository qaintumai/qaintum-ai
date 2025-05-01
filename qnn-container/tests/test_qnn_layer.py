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

# tests/test_qnn_layer.py

import pytest
import pennylane as qml
import torch

from qaintum_qnn.layers.qnn_layer import QuantumNeuralNetworkLayer

@pytest.fixture
def setup_qnn_layer():
    num_wires = 4
    required_params = 9 * num_wires - 4
    qnn_layer = QuantumNeuralNetworkLayer(num_wires=num_wires)
    dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=2)
    return num_wires, required_params, qnn_layer, dev

def test_layer_applies_correct_operations(setup_qnn_layer):
    num_wires, required_params, qnn_layer, dev = setup_qnn_layer
    params = torch.tensor([0.1] * required_params)

    @qml.qnode(dev)
    def circuit(params):
        qnn_layer.apply(params)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(params)
    assert len(output) == num_wires

def test_circuit_with_shorter_params(setup_qnn_layer):
    num_wires, required_params, qnn_layer, dev = setup_qnn_layer
    short_params = torch.tensor([0.1] * (required_params - 5))

    @qml.qnode(dev)
    def circuit(params):
        qnn_layer.apply(params)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(short_params)
    assert len(output) == num_wires

def test_circuit_with_longer_params(setup_qnn_layer):
    num_wires, required_params, qnn_layer, dev = setup_qnn_layer
    long_params = torch.tensor([0.1] * (required_params + 5))

    @qml.qnode(dev)
    def circuit(params):
        qnn_layer.apply(params)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(long_params)
    assert len(output) == num_wires

def test_apply_edge_case_params(setup_qnn_layer):
    num_wires, required_params, qnn_layer, dev = setup_qnn_layer
    zero_params = torch.zeros(required_params)
    extreme_params = torch.tensor([100.0] * required_params)

    @qml.qnode(dev)
    def circuit(params):
        qnn_layer.apply(params)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output_zero = circuit(zero_params)
    output_extreme = circuit(extreme_params)

    assert len(output_zero) == num_wires
    assert len(output_extreme) == num_wires
