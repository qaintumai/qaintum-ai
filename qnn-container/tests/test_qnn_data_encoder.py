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

from qnn.layers.qnn_data_encoder import QuantumDataEncoder

@pytest.fixture
def setup_encoder():
    num_wires = 4
    encoder = QuantumDataEncoder(num_wires=num_wires)
    dev = qml.device("strawberryfields.fock", wires=num_wires, cutoff_dim=2)
    return encoder, dev, num_wires

def test_encoding_applies_gates(setup_encoder):
    encoder, dev, num_wires = setup_encoder
    num_params = 8 * num_wires - 2
    input_data = torch.randn(num_params).tolist()

    @qml.qnode(dev)
    def circuit(input_data):
        encoder.encode(input_data)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(input_data)
    assert len(output) == num_wires

def test_encoder_with_insufficient_data(setup_encoder):
    encoder, dev, num_wires = setup_encoder
    insufficient_data = torch.randn(2 * num_wires - 1).tolist()

    @qml.qnode(dev)
    def circuit(input_data):
        encoder.encode(input_data)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(insufficient_data)
    assert len(output) == num_wires

def test_encoder_with_exact_data(setup_encoder):
    encoder, dev, num_wires = setup_encoder
    exact_data = torch.randn(8 * num_wires - 2).tolist()

    @qml.qnode(dev)
    def circuit(input_data):
        encoder.encode(input_data)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(exact_data)
    assert len(output) == num_wires

def test_encoder_with_multiple_rounds(setup_encoder):
    encoder, dev, num_wires = setup_encoder
    multiple_rounds_data = torch.randn((8 * num_wires - 2) * 2).tolist()

    @qml.qnode(dev)
    def circuit(input_data):
        encoder.encode(input_data)
        return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

    output = circuit(multiple_rounds_data)
    assert len(output) == num_wires

def test_invalid_data_type(setup_encoder):
    encoder, dev, num_wires = setup_encoder
    invalid_data = "invalid input data"

    with pytest.raises(TypeError):
        @qml.qnode(dev)
        def circuit(input_data):
            encoder.encode(input_data)
            return [qml.expval(qml.X(wire)) for wire in range(num_wires)]

        circuit(invalid_data)
