# qnn_examples/basic_usage.py

import torch
from qaintum_qnn.models.quantum_neural_network import QuantumNeuralNetwork

def main():
    # Initialize the quantum neural network
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "probabilities"

    model = QuantumNeuralNetwork(
        num_wires=num_wires,
        cutoff_dim=cutoff_dim,
        num_layers=num_layers,
        output_size=output_size,
        init_method="xavier",
        active_sd=0.001,
        passive_sd=0.2,
        gain=1.0,
        normalize_inputs=True,
        dropout_rate=0.2
    )

    # Example input data
    inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])

    # Perform a forward pass
    outputs = model(inputs)
    print("Outputs:", outputs)

if __name__ == "__main__":
    main()