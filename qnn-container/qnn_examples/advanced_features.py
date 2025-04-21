# examples/qnn_examples/advanced_features.py

import torch
import torch.nn as nn
from qnn.models.quantum_neural_network import QuantumNeuralNetwork

def main():
    print("Demonstrating Advanced Features of the Quantum Neural Network (QNN)")

    # === Feature 1: Custom Weight Initialization ===
    print("\nFeature 1: Custom Weight Initialization")
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "probabilities"

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

    # Normalize inputs to [0, 2π]
    model_with_normalization = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True  # Enable normalization
    )
    normalized_outputs = model_with_normalization(inputs)
    print(f"Outputs with normalization: {normalized_outputs}")

    # Disable normalization
    model_without_normalization = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=False  # Disable normalization
    )
    non_normalized_outputs = model_without_normalization(inputs)
    print(f"Outputs without normalization: {non_normalized_outputs}")

    # === Feature 3: Hybrid Quantum-Classical Model ===
    print("\nFeature 3: Hybrid Quantum-Classical Model")
    class HybridModel(nn.Module):
        def __init__(self, num_wires, cutoff_dim, num_layers, output_size):
            super(HybridModel, self).__init__()
            self.classical_layer = nn.Linear(4, 8)  # Classical fully connected layer
            self.quantum_layer = QuantumNeuralNetwork(
                num_wires=num_wires,
                cutoff_dim=cutoff_dim,
                num_layers=num_layers,
                output_size=output_size,
                init_method="xavier",
                active_sd=0.001,
                passive_sd=0.2,
                gain=1.0,
                normalize_inputs=True
            )

        def forward(self, x):
            x = torch.relu(self.classical_layer(x))  # Classical preprocessing
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