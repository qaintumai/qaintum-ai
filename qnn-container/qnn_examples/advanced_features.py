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

# qnn_examples/advanced_features.py

import torch
import torch.nn as nn
from qaintum_qnn.models.quantum_neural_network import QuantumNeuralNetwork
from qaintum_qnn.utils.normalization import ZScoreNormalization, MinMaxScaling, NormalizeToRange, NormalizeToRadians

def main():
    print("Demonstrating Advanced Features of the Quantum Neural Network (QNN)")

    # === Feature 1: Custom Weight Initialization ===
    print("\nFeature 1: Custom Weight Initialization")
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "multi"

    # Initialize QNN with Xavier initialization
    model_xavier = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",  # Use Xavier initialization
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True
    )

    # Initialize QNN with Kaiming initialization
    model_kaiming = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="kaiming",  # Use Kaiming initialization
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True
    )

    print("QNN initialized with Xavier and Kaiming methods.")

    # === Feature 2: Input Normalization ===
    print("\nFeature 2: Input Normalization")
    inputs = torch.tensor([[3.14, 6.28, 9.42, 12.56]])  # Raw inputs (not normalized)

    # Define normalization methods
    z_score_normalizer = ZScoreNormalization()
    min_max_scaler = MinMaxScaling(min_value=0.0, max_value=1.0)
    normalize_to_range = NormalizeToRange(target_min=-1.0, target_max=1.0)
    normalize_to_radians = NormalizeToRadians()

    # Apply Z-Score Normalization
    z_score_normalized = z_score_normalizer(inputs)
    print(f"Z-Score Normalized Inputs: {z_score_normalized}")

    # Apply Min-Max Scaling
    min_max_scaled = min_max_scaler(inputs)
    print(f"Min-Max Scaled Inputs: {min_max_scaled}")

    # Apply Normalization to a Specific Range [-1, 1]
    range_normalized = normalize_to_range(inputs)
    print(f"Normalized to Range [-1, 1]: {range_normalized}")

    # Apply Normalization to [0, 2π] for Quantum Circuits
    radians_normalized = normalize_to_radians(inputs)
    print(f"Normalized to Radians [0, 2π]: {radians_normalized}")

    # Pass normalized inputs to the QNN
    model_with_normalization = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=False  # Disable internal normalization since we're handling it explicitly
    )

    # Forward pass with normalized inputs
    normalized_outputs = model_with_normalization(radians_normalized)  # Using radians-normalized inputs
    print(f"Outputs with normalization: {normalized_outputs}")

    # === Feature 3: Hybrid Quantum-Classical Model ===
    print("\nFeature 3: Hybrid Quantum-Classical Model")
    class HybridModel(nn.Module):
        def __init__(self, num_wires, cutoff_dim, num_layers, output_size):
            super(HybridModel, self).__init__()
            self.classical_layer = nn.Linear(4, 8)  # Classical fully connected layer
            self.normalizer = NormalizeToRadians()  # Normalize inputs to [0, 2π]
            self.quantum_layer = QuantumNeuralNetwork(
                num_wires=num_wires,
                cutoff_dim=cutoff_dim,
                num_layers=num_layers,
                output_size=output_size,
                init_method="xavier",
                active_sd=0.001,
                passive_sd=0.2,
                gain=1.0,
                normalize_inputs=False  # Disable internal normalization
            )

        def forward(self, x):
            x = torch.relu(self.classical_layer(x))  # Classical preprocessing
            x = self.normalizer(x)                  # Normalize inputs for quantum processing
            x = self.quantum_layer(x)               # Quantum processing
            return x

    hybrid_model = HybridModel(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size
    )

    # Forward pass through the hybrid model
    hybrid_outputs = hybrid_model(inputs)
    print(f"Hybrid model outputs: {hybrid_outputs}")

    # === Feature 4: Post-Processing Outputs ===
    print("\nFeature 4: Post-Processing Outputs")
    # Simulate a classification task
    outputs = model_xavier(inputs)
    predicted_class = torch.argmax(outputs, dim=-1)
    print(f"Predicted class: {predicted_class.item()}")

if __name__ == "__main__":
    main()