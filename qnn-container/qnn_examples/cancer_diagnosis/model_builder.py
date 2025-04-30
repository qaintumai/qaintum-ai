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

# qnn_examples/cancer_diagnosis/model_builder.py

def get_model(
    quantum_nn,
    num_wires,
    num_layers,
    weights=None,
    include_classical_layer=False,
    hidden_units=16,
    output_units=1
):
    """
    Builds and returns a PyTorch model containing a custom quantum layer (TorchLayer) and optionally a classical layer.

    Arguments:
    - quantum_nn: Quantum neural network (qnode) passed into the TorchLayer.
    - num_wires: Number of quantum wires (modes).
    - num_layers: Number of layers in the quantum circuit.
    - weights: Optional initial weights to assign to the quantum layer (default: None).
    - include_classical_layer: Whether to include a classical layer after the quantum layer (default: False).
    - hidden_units: Number of units in the classical hidden layer (if included).
    - output_units: Number of output units in the final layer (default: 1).

    Returns:
    - A PyTorch Sequential model containing the quantum layer and potentially a classical layer.
    """
    # Initialize the TorchLayer with the quantum neural network, number of layers, and number of wires
    qlayer = TorchLayer(
        quantum_nn=quantum_nn,
        num_layers=num_layers,
        num_wires=num_wires,
        active_sd=0.01,
        passive_sd=0.02,
        method='uniform'
    )

    # Build a PyTorch Sequential model
    if include_classical_layer:
        # Add a classical layer after the quantum layer
        clayer = nn.Linear(hidden_units, output_units)
        activation = nn.ReLU()  # Example activation function
        model = nn.Sequential(
            qlayer,       # The quantum layer
            clayer,       # The classical layer
            activation    # Activation function
        )
    else:
        # Use only the quantum layer
        model = nn.Sequential(qlayer)

    # If weights are provided, manually initialize the quantum layer's parameters
    if weights is not None:
        with torch.no_grad():
            weights = torch.tensor(weights, dtype=torch.float32)  # Ensure weights are torch tensors
            for name, param in model.named_parameters():
                if "weights" in name:  # Adjust this check based on actual parameter naming
                    if param.shape == weights.shape:  # Check if the shapes match
                        param.copy_(weights)  # Assign the provided weights to the quantum layer
                        print(f"Weights for {name} have been initialized with shape {param.shape}.")
                    else:
                        raise ValueError(f"Shape mismatch: {param.shape} != {weights.shape}")

    return model