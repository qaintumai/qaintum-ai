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

# qaintum_qt/layers/quantum_feed_forward.py

import torch
import torch.nn as nn
from qaintum_qe.utils.qff_config import determine_qnn_parameters
from qaintum_qnn.models.quantum_neural_network import QuantumNeuralNetwork

class QuantumFeedForward(nn.Module):
    """
    Quantum Feed-Forward Layer for classification and regression tasks.
    """
    def __init__(self, task_type, num_classes=None,
                 init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0,
                 normalize_inputs=True, dropout_rate=0.0, **kwargs):
        super(QuantumFeedForward, self).__init__()

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Unsupported task type: {task_type}")

        if task_type == "classification" and num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        self.task_type = task_type
        self.num_classes = num_classes

        params = determine_qnn_parameters(
            task_type=task_type,
            num_classes=num_classes
        )
        self.cutoff_dim = params["cutoff_dim"]
        self.num_wires = params["num_wires"]
        self.output_size = params["output_size"]

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
        quantum_output = self.quantum_network(inputs)
        batch_size = inputs.size(0)

        if self.task_type == "classification":
            required_size = batch_size * self.num_classes
            if quantum_output.numel() < required_size:
                raise ValueError(
                    f"Quantum output size ({quantum_output.numel()}) is smaller than "
                    f"the required size ({required_size})."
                )
            output = quantum_output.flatten()[:required_size].view(batch_size, self.num_classes)

        elif self.task_type == "regression":
            output = quantum_output

        return output
