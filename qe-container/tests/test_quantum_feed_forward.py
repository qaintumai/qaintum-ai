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

import pytest
import torch
from qaintum_qe.layers.quantum_feed_forward import QuantumFeedForward

@pytest.fixture
def small_classification_task_params():
    return {
        "task_type": "classification",
        "num_classes": 5,
    }

@pytest.fixture
def large_classification_task_params():
    return {
        "task_type": "classification",
        "num_classes": 30,
    }

@pytest.fixture
def regression_task_params():
    return {
        "task_type": "regression",
    }

@pytest.fixture
def sample_input():
    return torch.randn(8, 16)  # Batch size = 8, Input features = 16

def test_quantum_feed_forward_initialization_small_classification(small_classification_task_params):
    """Test that QuantumFeedForward initializes correctly for small classification tasks."""
    quantum_ff = QuantumFeedForward(**small_classification_task_params)
    assert quantum_ff.task_type == "classification"
    assert quantum_ff.num_classes == 5

def test_quantum_feed_forward_initialization_large_classification(large_classification_task_params):
    """Test that QuantumFeedForward initializes correctly for large classification tasks."""
    quantum_ff = QuantumFeedForward(**large_classification_task_params)
    assert quantum_ff.task_type == "classification"
    assert quantum_ff.num_classes == 30

def test_quantum_feed_forward_initialization_regression(regression_task_params):
    """Test that QuantumFeedForward initializes correctly for regression tasks."""
    quantum_ff = QuantumFeedForward(**regression_task_params)
    assert quantum_ff.task_type == "regression"

def test_forward_pass_small_classification(sample_input, small_classification_task_params):
    """Test the forward pass for small classification tasks."""
    quantum_ff = QuantumFeedForward(**small_classification_task_params)
    output = quantum_ff(sample_input)
    batch_size, num_classes = output.shape
    assert batch_size == sample_input.size(0)
    assert num_classes == small_classification_task_params["num_classes"]

def test_forward_pass_large_classification(sample_input, large_classification_task_params):
    """Test the forward pass for large classification tasks."""
    quantum_ff = QuantumFeedForward(**large_classification_task_params)
    output = quantum_ff(sample_input)
    batch_size, num_classes = output.shape
    assert batch_size == sample_input.size(0)
    assert num_classes == large_classification_task_params["num_classes"]

def test_forward_pass_regression(sample_input, regression_task_params):
    """Test the forward pass for regression tasks."""
    quantum_ff = QuantumFeedForward(**regression_task_params)
    output = quantum_ff(sample_input)
    assert output.shape == sample_input.shape