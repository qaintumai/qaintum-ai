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

# qt/models/quantum_feed_forward.py

from torch import nn
from qnn.layers.qnn_circuit import QuantumNeuralNetworkCircuit

class QuantumFeedForward(nn.Module):
    """
    A class used to define a feedforward block for a quantum neural network.

    Usage:
    To use the QuantumFeedForward class, import it as follows:
    from layers.quantum_feed_forward import QuantumFeedForward

    Example:
    model = QuantumFeedForward(num_layers=2, num_wires=4, cutoff_dim=10, embed_len=64)
    output = model(input_tensor)
    """

    def __init__(self, num_layers, num_wires, cutoff_dim, embed_len, dropout=0.1, output_size="probabilities"):
        """
        Initializes the QuantumFeedForward class with the given parameters.

        Parameters:
        - num_layers (int): Number of layers in the quantum neural network.
        - num_wires (int): Number of wires (qubits/qumodes) in the quantum circuit.
        - cutoff_dim (int): Cutoff dimension for the Fock space representation.
        - embed_len (int): Length of the embedding vector.
        - dropout (float, optional): Dropout rate for regularization. Default is 0.1.
        - output_size (str, optional): Output type of the quantum circuit ("single", "multi", or "probabilities").
        """
        super(QuantumFeedForward, self).__init__()
        self.num_layers = num_layers
        self.num_wires = num_wires
        self.cutoff_dim = cutoff_dim
        self.embed_len = embed_len

        # Initialize the quantum neural network (QNN)
        self.quantum_nn = QuantumNeuralNetworkCircuit(
            num_wires=num_wires,
            cutoff_dim=cutoff_dim,
            num_layers=num_layers,
            output_size=output_size
        )

        # Define the quantum feedforward layers
        self.qnn_model = self.quantum_nn.build_circuit()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embed_len)

    def forward(self, x):
        """
        Applies the feedforward block to the input tensor.

        Parameters:
        - x (torch.Tensor): Input tensor.

        Returns:
        - torch.Tensor: Output tensor after applying feedforward, dropout, and layer normalization.
        """
        # Ensure the input tensor matches the number of wires
        if x.shape[-1] != self.num_wires:
            raise ValueError(f"Input tensor must have {self.num_wires} features to match the number of wires.")

        # Apply the quantum circuit
        ff_output = self.qnn_model(x)

        # Apply dropout and layer normalization
        ff_output = self.dropout_layer(ff_output)
        return self.layer_norm(ff_output + x)