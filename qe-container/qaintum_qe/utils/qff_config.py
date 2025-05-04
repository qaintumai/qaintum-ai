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

# qaintum_qe/utils/qff_config.py

def find_optimal_cutoff_and_wires(target_output_size, max_cutoff_dim=10, max_num_wires=10):
    """
    Finds the optimal cutoff_dim and num_wires such that cutoff_dim ** num_wires >= target_output_size,
    while minimizing cutoff_dim ** num_wires.

    Parameters:
    - target_output_size (int): Required output size.
    - max_cutoff_dim (int): Maximum allowed cutoff dimension.
    - max_num_wires (int): Maximum allowed number of wires.

    Returns:
    - tuple: (optimal_cutoff_dim, optimal_num_wires)
    """
    optimal_cutoff_dim = None
    optimal_num_wires = None
    min_hilbert_space_size = float('inf')

    for cutoff_dim in range(2, max_cutoff_dim + 1):  # Start from 2 to avoid trivial cases
        for num_wires in range(1, max_num_wires + 1):
            hilbert_space_size = cutoff_dim ** num_wires
            if hilbert_space_size >= target_output_size < min_hilbert_space_size:
                min_hilbert_space_size = hilbert_space_size
                optimal_cutoff_dim = cutoff_dim
                optimal_num_wires = num_wires

    if optimal_cutoff_dim is None or optimal_num_wires is None:
        raise ValueError("No valid cutoff_dim and num_wires found within the given constraints.")

    return optimal_cutoff_dim, optimal_num_wires

def determine_qnn_parameters(task_type, num_classes=None):
    """
    Determines the optimal cutoff_dim, num_wires, and output_size based on the task type.

    Parameters:
    - task_type (str): Type of task ("classification", "regression").
    - num_classes (int, optional): Number of classes for classification tasks.

    Returns:
    - dict: Dictionary containing cutoff_dim, num_wires, and output_size.
    """
    if task_type == "classification":
        if num_classes is None:
            raise ValueError("num_classes must be provided for classification tasks.")

        # Set num_wires equal to num_classes for small num_classes
        if num_classes <= 10:
            cutoff_dim = 3
            num_wires = num_classes
            output_size = "multi"
        else:
            cutoff_dim, num_wires = find_optimal_cutoff_and_wires(num_classes)
            output_size = "probabilities"

    elif task_type == "regression":
        # Default parameters for regression tasks
        cutoff_dim = 3
        num_wires = 4
        output_size = "single"

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return {
        "cutoff_dim": cutoff_dim,
        "num_wires": num_wires,
        "output_size": output_size,
    }
