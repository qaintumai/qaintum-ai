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

import pytest
import torch
from qaintum_qt.layers.quantum_feed_forward import QuantumFeedForward


@pytest.fixture
def sequence_task_params():
    return {
        "task_type": "sequence",
        "vocab_size": 100,
        "sequence_length": 10,
        "num_classes": None,
    }


@pytest.fixture
def small_classification_task_params():
    return {
        "task_type": "classification",
        "num_classes": 5,
        "vocab_size": None,
        "sequence_length": None,
    }


@pytest.fixture
def large_classification_task_params():
    return {
        "task_type": "classification",
        "num_classes": 30,
        "vocab_size": None,
        "sequence_length": None,
    }


@pytest.fixture
def regression_task_params():
    return {
        "task_type": "regression",
        "num_classes": None,
        "vocab_size": None,
        "sequence_length": None,
    }


@pytest.fixture
def generation_task_params():
    return {
        "task_type": "generation",
        "vocab_size": 500,
        "sequence_length": 20,
        "num_classes": None,
    }


@pytest.fixture
def sample_input():
    return torch.randn(8, 16)  # Batch size = 8, Input features = 16


def test_quantum_feed_forward_initialization_sequence(sequence_task_params):
    """Test that QuantumFeedForward initializes correctly for sequence tasks."""
    quantum_ff = QuantumFeedForward(**sequence_task_params)
    assert quantum_ff.task_type == "sequence", "Task type should be 'sequence'."
    assert quantum_ff.vocab_size == 100, "Vocabulary size should match input."
    assert quantum_ff.sequence_length == 10, "Sequence length should match input."


def test_quantum_feed_forward_initialization_classification(classification_task_params):
    """Test that QuantumFeedForward initializes correctly for classification tasks."""
    quantum_ff = QuantumFeedForward(**classification_task_params)
    assert quantum_ff.task_type == "classification", "Task type should be 'classification'."
    assert quantum_ff.num_classes == 10, "Number of classes should match input."


def test_quantum_feed_forward_initialization_regression(regression_task_params):
    """Test that QuantumFeedForward initializes correctly for regression tasks."""
    quantum_ff = QuantumFeedForward(**regression_task_params)
    assert quantum_ff.task_type == "regression", "Task type should be 'regression'."


def test_quantum_feed_forward_initialization_generation(generation_task_params):
    """Test that QuantumFeedForward initializes correctly for generation tasks."""
    quantum_ff = QuantumFeedForward(**generation_task_params)
    assert quantum_ff.task_type == "generation", "Task type should be 'generation'."
    assert quantum_ff.vocab_size == 500, "Vocabulary size should match input."
    assert quantum_ff.sequence_length == 20, "Sequence length should match input."

def test_forward_pass_sequence(sample_input, sequence_task_params):
    """Test the forward pass for sequence tasks."""
    quantum_ff = QuantumFeedForward(**sequence_task_params)
    output = quantum_ff(sample_input)
    batch_size, sequence_length, embedding_size = output.shape
    assert batch_size == sample_input.size(0), "Batch size should match input."
    assert sequence_length == sequence_task_params["sequence_length"], "Sequence length should match input."
    assert embedding_size == sample_input.size(-1), "Embedding size should match input features."

def test_forward_pass_generation(sample_input, generation_task_params):
    """Test the forward pass for generation tasks."""
    quantum_ff = QuantumFeedForward(**generation_task_params)
    output = quantum_ff(sample_input)
    batch_size, sequence_length, vocab_size = output.shape
    assert batch_size == sample_input.size(0), "Batch size should match input."
    assert sequence_length == generation_task_params["sequence_length"], "Sequence length should match input."
    assert vocab_size == generation_task_params["vocab_size"], "Vocabulary size should match input."

def test_forward_pass_small_classification(sample_input, small_classification_task_params):
    """Test the forward pass for classification tasks."""
    quantum_ff = QuantumFeedForward(**small_classification_task_params)
    output = quantum_ff(sample_input)
    batch_size, num_classes = output.shape
    assert batch_size == sample_input.size(0), "Batch size should match input."
    assert num_classes == small_classification_task_params["num_classes"], "Number of classes should match input."

def test_forward_pass_large_classification(sample_input, large_classification_task_params):
    """Test the forward pass for classification tasks."""
    quantum_ff = QuantumFeedForward(**large_classification_task_params)
    output = quantum_ff(sample_input)
    batch_size, num_classes = output.shape
    assert batch_size == sample_input.size(0), "Batch size should match input."
    assert num_classes == large_classification_task_params["num_classes"], "Number of classes should match input."

def test_forward_pass_regression(sample_input, regression_task_params):
    """Test the forward pass for regression tasks."""
    quantum_ff = QuantumFeedForward(**regression_task_params)
    output = quantum_ff(sample_input)
    assert output.shape == sample_input.shape, "Output shape should match input shape for regression tasks."
