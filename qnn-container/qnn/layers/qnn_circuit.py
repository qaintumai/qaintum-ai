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

# qnn/layers/qnn_circuit.py

import torch
import pennylane as qml
from qnn.layers.qnn_data_encoder import QuantumDataEncoder
from qnn.layers.qnn_layer import QuantumNeuralNetworkLayer
from qnn.layers.qnn_weight_init import QuantumWeightInitializer  # Import the QuantumWeightInitializer

class QuantumNeuralNetworkCircuit:
    """
    Builds a quantum neural network circuit using Strawberry Fields' Fock backend.

    Attributes:
        num_wires (int): Number of quantum wires (qumodes).
        cutoff_dim (int): Cutoff dimension for the Fock space.
        dev (qml.device): Quantum device for simulating the circuit.
        output_size (str): Defines the type of output ("single", "multi", or "probabilities").
        num_layers (int): Number of quantum layers in the circuit.
        weights (torch.Tensor): Pre-initialized weights for the quantum layers.

    Methods:
        build_circuit: Constructs and returns a QNode representing the quantum circuit.
        initialize_weights: Initializes the weights for the quantum layers.
    """

    def __init__(self, num_wires, cutoff_dim, num_layers, output_size="single", init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0):
        """
        Initializes the QuantumNeuralNetworkCircuit.

        Parameters:
            num_wires (int): Number of quantum wires (qumodes).
            cutoff_dim (int): Cutoff dimension for the Fock space.
            num_layers (int): Number of quantum layers in the circuit.
            output_size (str): Type of output ("single", "multi", or "probabilities").
                               Defaults to "single".
            init_method (str): Initialization method ('normal', 'uniform', 'xavier', or 'kaiming').
                               Defaults to 'normal'.
            active_sd (float): Standard deviation for active gate weights. Used only for 'normal'. Default is 0.0001.
            passive_sd (float): Standard deviation for passive gate weights. Used only for 'normal'. Default is 0.1.
            gain (float): Scaling factor for Xavier/Kaiming initialization. Default is 1.0.
        """
        self.num_wires = num_wires
        self.cutoff_dim = cutoff_dim
        self.num_layers = num_layers
        self.output_size = output_size.lower()  # Normalize to lowercase

        # Validate output_size
        if self.output_size not in ["single", "multi", "probabilities"]:
            raise ValueError("output_size must be one of 'single', 'multi', or 'probabilities'.")

        # Initialize the quantum device
        try:
            self.dev = qml.device("strawberryfields.fock", wires=self.num_wires, cutoff_dim=self.cutoff_dim)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize the quantum device: {e}")

        # Initialize weights using QuantumWeightInitializer
        self.initialize_weights(init_method, active_sd, passive_sd, gain)

    def initialize_weights(self, init_method, active_sd, passive_sd, gain):
        """
        Initializes the weights for the quantum layers using QuantumWeightInitializer.

        Parameters:
            init_method (str): Initialization method ('normal', 'uniform', 'xavier', or 'kaiming').
            active_sd (float): Standard deviation for active gate weights. Used only for 'normal'.
            passive_sd (float): Standard deviation for passive gate weights. Used only for 'normal'.
            gain (float): Scaling factor for Xavier/Kaiming initialization.
        """
        initializer = QuantumWeightInitializer(
            method=init_method,
            active_sd=active_sd,
            passive_sd=passive_sd,
            gain=gain
        )

        # Generate weights for the specified number of layers and wires
        weights = initializer.init_weights(layers=self.num_layers, num_wires=self.num_wires)

    def build_circuit(self):
        """
        Constructs and returns a QNode representing the quantum circuit.

        Returns:
            qml.QNode: A QNode representing the quantum circuit.
        """
        def circuit(inputs, var):
            """
            Quantum circuit for encoding data and applying quantum neural network layers.

            Parameters:
                inputs (torch.Tensor or list): Input data to encode into the quantum state.
                var (iterable): Iterable of parameter vectors for the quantum layers.

            Returns:
                array-like: Expectation values or probabilities based on output_size.
            """
            # Ensure inputs is a Python list for compatibility with the encoder
            if isinstance(inputs, torch.Tensor):
                inputs = inputs.tolist()

            # Encode input data
            q_encoder = QuantumDataEncoder(self.num_wires)
            q_encoder.encode(inputs)

            # Apply quantum layers
            q_layer = QuantumNeuralNetworkLayer(self.num_wires)
            for v in var:
                q_layer.apply(v)

            # Return outputs based on output_size
            if self.output_size == "single":
                return qml.expval(qml.X(0))
            elif self.output_size == "multi":
                return [qml.expval(qml.X(wire)) for wire in range(self.num_wires)]
            elif self.output_size == "probabilities":
                wires = list(range(self.num_wires))
                return qml.probs(wires=wires)

        # Create the QNode with the quantum circuit
        return qml.QNode(circuit, self.dev, interface="torch")