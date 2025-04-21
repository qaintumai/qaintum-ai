import torch
import pytest
from qnn.models.quantum_neural_network import QuantumNeuralNetwork

def test_basic_usage_forward_pass():
    num_wires = 4
    cutoff_dim = 5
    num_layers = 2
    output_size = "probabilities"
    batch_size = 1

    model = QuantumNeuralNetwork(
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

    inputs = torch.tensor([[0.1, 0.2, 0.3, 0.4]])  # shape: (1, 4)

    outputs = model(inputs)

    # Check type
    assert isinstance(outputs, torch.Tensor), "Output should be a torch.Tensor"

    # Check shape
    expected_output_size = cutoff_dim ** num_wires
    assert outputs.shape == (batch_size, expected_output_size), (
        f"Expected output shape ({batch_size}, {expected_output_size}), got {outputs.shape}"
    )

    # Optional checks
    assert torch.all(outputs >= 0), "All probabilities must be non-negative"

