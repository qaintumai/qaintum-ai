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

# qaintum_qt/layers/quantum_feed_forward.py

import torch
import torch.nn as nn
from qaintum_qt.utils.qff_config import determine_qnn_parameters
from qaintum_qnn.models.quantum_neural_network import QuantumNeuralNetwork

class QuantumFeedForward(nn.Module):
    """
    Quantum Feed-Forward Layer built on top of QuantumNeuralNetwork.

    Parameters:
    - task_type (str): Type of task ("sequence", "classification", "regression", "generation").
    - vocab_size (int, optional): Vocabulary size for sequence-to-sequence or generation tasks.
    - num_classes (int, optional): Number of classes for classification tasks.
    - sequence_length (int, optional): Sequence length for sequence-to-sequence or generation tasks.
    - init_method (str, optional): Weight initialization method ("normal", "uniform", "xavier", "kaiming").
    - active_sd (float, optional): Std. dev. for active gate weights.
    - passive_sd (float, optional): Std. dev. for passive gate weights.
    - gain (float, optional): Gain factor applied to weights.
    - normalize_inputs (bool, optional): Whether to normalize inputs to the circuit.
    - dropout_rate (float, optional): Dropout rate on the output during training.
    """
    def __init__(self, task_type, vocab_size=None, num_classes=None, sequence_length=None,
                 init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0,
                 normalize_inputs=True, dropout_rate=0.0, **kwargs):
        super(QuantumFeedForward, self).__init__()
        self.task_type = task_type
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.sequence_length = sequence_length

        # Validate task type
        if task_type not in ["sequence", "classification", "regression", "generation"]:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Validate required parameters
        if task_type in ["sequence", "generation"]:
            if vocab_size is None or sequence_length is None:
                raise ValueError("vocab_size and sequence_length must be provided for sequence or generation tasks.")
        elif task_type == "classification":
            if num_classes is None:
                raise ValueError("num_classes must be provided for classification tasks.")

        # Determine QNN parameters based on task type
        params = determine_qnn_parameters(
            task_type=task_type,
            vocab_size=vocab_size,
            num_classes=num_classes,
            sequence_length=sequence_length
        )
        self.cutoff_dim = params["cutoff_dim"]
        self.num_wires = params["num_wires"]
        self.output_size = params["output_size"]

        # Initialize QuantumNeuralNetwork
        self.quantum_network = QuantumNeuralNetwork(
            num_wires=self.num_wires,
            cutoff_dim=self.cutoff_dim,
            num_layers=2,
            output_size=self.output_size,
            init_method=init_method,
            active_sd=active_sd,
            passive_sd=passive_sd,
            gain=gain,
            normalize_inputs=normalize_inputs,
            dropout_rate=dropout_rate
        )

    def forward(self, inputs):
        """
        Performs a forward pass through the quantum feed-forward layer and reshapes the output
        based on the task type.
        """
        quantum_output = self.quantum_network(inputs)
        print(f"Quantum output shape: {quantum_output.shape}")

        batch_size = inputs.size(0)

        if self.task_type == "sequence":
            embedding_size = inputs.size(-1)
            required_total_size = batch_size * self.sequence_length * embedding_size

            if quantum_output.numel() < required_total_size:
                raise ValueError(
                    f"Quantum output size ({quantum_output.numel()}) is smaller than "
                    f"the required size ({required_total_size})."
                )

            # Truncate if needed
            quantum_output = quantum_output.flatten()[:required_total_size]
            reshaped_output = quantum_output.view(batch_size, self.sequence_length, embedding_size)

        elif self.task_type == "generation":
            required_total_size = batch_size * self.sequence_length * self.vocab_size

            if quantum_output.numel() < required_total_size:
                raise ValueError(
                    f"Quantum output size ({quantum_output.numel()}) is smaller than "
                    f"the required size ({required_total_size})."
                )

            # Truncate if needed
            quantum_output = quantum_output.flatten()[:required_total_size]
            reshaped_output = quantum_output.view(batch_size, self.sequence_length, self.vocab_size)

            # Apply softmax to get token probabilities
            reshaped_output = torch.softmax(reshaped_output, dim=-1)

        elif self.task_type == "classification":
            if self.num_classes <= 10:
                # Expect direct multi-class output shape
                expected_shape = (batch_size, self.num_classes)
                if quantum_output.numel() < batch_size * self.num_classes:
                    raise ValueError(
                        f"Quantum output size ({quantum_output.numel()}) is smaller than "
                        f"the required size ({batch_size * self.num_classes}) for multi-class output."
                    )
                reshaped_output = quantum_output.view(expected_shape)
            else:
                # Expect flattened probabilities, truncate to num_classes
                required_total_size = batch_size * self.num_classes
                if quantum_output.numel() < required_total_size:
                    raise ValueError(
                        f"Quantum output size ({quantum_output.numel()}) is smaller than "
                        f"the required size ({required_total_size}) for probability output."
                    )
                quantum_output = quantum_output.flatten()[:required_total_size]
                reshaped_output = quantum_output.view(batch_size, self.num_classes)

        elif self.task_type == "regression":
            reshaped_output = quantum_output

        else:
            raise ValueError(f"Unsupported task type: {self.task_type}")

        return reshaped_output

