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

# qnn/models/quantum_neural_network.py

import pennylane as qml
import torch
import numpy as np

from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit

class QuantumNeuralNetwork(torch.nn.Module):
    def __init__(self, num_wires=4, cutoff_dim=5, num_layers=2, output_size="single", init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0, normalize_inputs=True, dropout_rate=0.0):
        """
        Initializes the quantum neural network model.

        Parameters:
        - num_wires (int): Number of quantum wires (qumodes) in the circuit.
        - cutoff_dim (int): Cutoff dimension for the Fock space.
        - num_layers (int): Number of quantum layers in the circuit.
        - output_size (str): Type of output ("single", "multi", or "probabilities").
                             Defaults to "single".
        - init_method (str): Initialization method ('normal', 'uniform', 'xavier', or 'kaiming').
                             Defaults to 'normal'.
        - active_sd (float): Standard deviation for active gate weights. Used only for 'normal'. Default is 0.0001.
        - passive_sd (float): Standard deviation for passive gate weights. Used only for 'normal'. Default is 0.1.
        - gain (float): Scaling factor for Xavier/Kaiming initialization. Default is 1.0.
        - normalize_inputs (bool): Whether to normalize inputs to [0, 2π]. Defaults to True.
        - dropout_rate (float): Probability of an element being zeroed in dropout. Default is 0.0 (no dropout).
        """
        super(QuantumNeuralNetwork, self).__init__()
        self.num_wires = num_wires
        self.cutoff_dim = cutoff_dim
        self.num_layers = num_layers
        self.output_size = output_size.lower()
        self.normalize_inputs = normalize_inputs
        self.dropout_rate = dropout_rate

        # Validate output_size
        if self.output_size not in ["single", "multi", "probabilities"]:
            raise ValueError("output_size must be one of 'single', 'multi', or 'probabilities'.")

        # Initialize the quantum circuit
        self.qnn_circuit = QuantumNeuralNetworkCircuit(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            num_layers=self.num_layers,
            output_size=self.output_size,
            init_method=init_method,
            active_sd=active_sd,
            passive_sd=passive_sd,
            gain=gain
        ).build_circuit()

        # Build the Torch-compatible quantum layer
        self.qlayers = self._build_quantum_layers()

        # Add dropout layer
        self.dropout = torch.nn.Dropout(p=self.dropout_rate) if self.dropout_rate > 0 else None

    def _build_quantum_layers(self):
        """
        Converts the quantum neural network to a Torch layer.

        Returns:
            qml.qnn.TorchLayer: A Torch-compatible quantum layer.
        """
        # Define the shape of the weights
        weight_shapes = {"var": (self.num_layers, 9 * self.num_wires - 4)}

        # Create a TorchLayer from the quantum circuit
        return qml.qnn.TorchLayer(self.qnn_circuit, weight_shapes)

    def forward(self, inputs):
        """
        Performs a forward pass through the quantum neural network.

        Parameters:
            inputs (torch.Tensor): Input data to encode into the quantum state.

        Returns:
            torch.Tensor: Output of the quantum circuit.
        """
        # Ensure inputs are a PyTorch tensor
        if not isinstance(inputs, torch.Tensor):
            raise TypeError("Inputs must be a PyTorch tensor.")

        # Optionally normalize inputs to the range [0, 2π]
        if self.normalize_inputs:
            inputs = torch.remainder(inputs, 2 * np.pi)

        # Pass the inputs through the quantum layers
        outputs = self.qlayers(inputs)

        # Apply dropout if enabled
        if self.dropout is not None:
            outputs = self.dropout(outputs)

        # Post-process outputs based on the task
        # if self.output_size == "probabilities":
        #    return torch.softmax(outputs, dim=-1)  # Convert logits to probabilities
        return outputs