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

# qaintum_qt/utils/qff_config.py

def determine_qnn_parameters(task_type, vocab_size=None, num_classes=None, sequence_length=None):
    """
    Determines the optimal cutoff_dim, num_wires, and output_size based on the task type.

    Parameters:
    - task_type (str): Type of task ("sequence", "classification", "regression", "generation").
    - vocab_size (int, optional): Vocabulary size for sequence-to-sequence or generation tasks.
    - num_classes (int, optional): Number of classes for classification tasks.
    - sequence_length (int, optional): Sequence length for sequence-to-sequence or generation tasks.

    Returns:
    - dict: Dictionary containing cutoff_dim, num_wires, and output_size.
    """
    if task_type == "sequence" or task_type == "generation":
        required_output_size = vocab_size * sequence_length
        output_size = "probabilities"
    elif task_type == "classification":
        required_output_size = num_classes
        output_size = "multi" if num_classes <= 20 else "probabilities"
    elif task_type == "regression":
        required_output_size = 1
        output_size = "single"
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # Find optimal cutoff_dim and num_wires
    cutoff_dim, num_wires = find_optimal_cutoff_and_wires(required_output_size)

    return {
        "cutoff_dim": cutoff_dim,
        "num_wires": num_wires,
        "output_size": output_size,
    }

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
            if hilbert_space_size >= target_output_size and hilbert_space_size < min_hilbert_space_size:
                min_hilbert_space_size = hilbert_space_size
                optimal_cutoff_dim = cutoff_dim
                optimal_num_wires = num_wires

    if optimal_cutoff_dim is None or optimal_num_wires is None:
        raise ValueError("No valid cutoff_dim and num_wires found within the given constraints.")

    return optimal_cutoff_dim, optimal_num_wires