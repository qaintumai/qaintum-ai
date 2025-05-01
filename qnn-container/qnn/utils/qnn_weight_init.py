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

# qnn/utils/qnn_weight_init.py

import torch
import numpy as np

class QuantumWeightInitializer:
    """
    A class used to initialize the weights for a quantum neural network layer.
    """

    def __init__(self, method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0):
        """
        Initialize the weight initializer.
        Args:
            method (str, optional): Initialization method ('normal', 'uniform', 'xavier', or 'kaiming'). Default is 'normal'.
            active_sd (float, optional): Standard deviation for active gate weights. Used only for 'normal'. Default is 0.0001.
            passive_sd (float, optional): Standard deviation for passive gate weights. Used only for 'normal'. Default is 0.1.
            gain (float, optional): Scaling factor for Xavier/Kaiming initialization. Default is 1.0.
        """
        self.method = method
        self.active_sd = active_sd
        self.passive_sd = passive_sd
        self.gain = gain

    def init_weights(self, layers, num_wires):
        """
        Initializes the weights for the quantum neural network layer.

        Parameters:
        - layers (int): Number of layers in the quantum neural network.
        - num_wires (int): Number of wires (qumodes) in the quantum circuit.

        Returns:
        - np.ndarray: A numpy array containing the initialized weights for the quantum neural network.
        """
        M = (num_wires - 1) * 2 + num_wires  # Number of interferometer parameters

        if self.method == 'normal':
            int1_weights = np.random.normal(size=[layers, M], scale=self.passive_sd)  # Beamsplitters and rotations
            s_weights = np.random.normal(size=[layers, num_wires], scale=self.active_sd)  # Squeezers
            int2_weights = np.random.normal(size=[layers, M], scale=self.passive_sd)  # Beamsplitters and rotations
            dr_weights = np.random.normal(size=[layers, num_wires], scale=self.active_sd)  # Displacement
            k_weights = np.random.normal(size=[layers, num_wires], scale=self.active_sd)  # Kerr

        elif self.method == 'uniform':
            int1_weights = np.random.uniform(low=-self.passive_sd, high=self.passive_sd, size=[layers, M])
            s_weights = np.random.uniform(low=-self.active_sd, high=self.active_sd, size=[layers, num_wires])
            int2_weights = np.random.uniform(low=-self.passive_sd, high=self.passive_sd, size=[layers, M])
            dr_weights = np.random.uniform(low=-self.active_sd, high=self.active_sd, size=[layers, num_wires])
            k_weights = np.random.uniform(low=-self.active_sd, high=self.active_sd, size=[layers, num_wires])

        elif self.method == 'xavier':
            fan_in = num_wires  # Approximation for quantum gates
            fan_out = num_wires
            std = self.gain * np.sqrt(2.0 / (fan_in + fan_out))

            int1_weights = np.random.normal(size=[layers, M], scale=std)
            s_weights = np.random.normal(size=[layers, num_wires], scale=std)
            int2_weights = np.random.normal(size=[layers, M], scale=std)
            dr_weights = np.random.normal(size=[layers, num_wires], scale=std)
            k_weights = np.random.normal(size=[layers, num_wires], scale=std)

        elif self.method == 'kaiming':
            fan_in = num_wires  # Approximation for quantum gates
            std = self.gain * np.sqrt(1.0 / fan_in)

            int1_weights = np.random.normal(size=[layers, M], scale=std)
            s_weights = np.random.normal(size=[layers, num_wires], scale=std)
            int2_weights = np.random.normal(size=[layers, M], scale=std)
            dr_weights = np.random.normal(size=[layers, num_wires], scale=std)
            k_weights = np.random.normal(size=[layers, num_wires], scale=std)

        else:
            raise ValueError(f"Unsupported initialization method: {self.method}")

        weights = np.concatenate(
            [int1_weights, s_weights, int2_weights, dr_weights, k_weights], axis=1)

        # Convert to a PyTorch tensor before returning
        return torch.tensor(weights, dtype=torch.float32)