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

# tests/test_normalization.py

import pytest
import torch
from qnn.utils.normalization import (
    ZScoreNormalization,
    MinMaxScaling,
    NormalizeToRange,
    NormalizeToRadians,
)

@pytest.fixture
def sample_data():
    """
    Fixture to provide sample input data as a PyTorch tensor.
    Shape: (batch_size=2, num_features=4)
    """
    return torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

def test_z_score_normalization(sample_data):
    """
    Test ZScoreNormalization class.
    Expected behavior: Data is standardized to have mean=0 and std=1 along the feature dimension.
    """
    normalizer = ZScoreNormalization()
    normalized_data = normalizer(sample_data)

    # Compute expected values
    mean = sample_data.mean(dim=1, keepdim=True)
    std = sample_data.std(dim=1, keepdim=True)
    expected_normalized_data = (sample_data - mean) / (std + 1e-8)

    # Validate output
    assert torch.allclose(normalized_data, expected_normalized_data, atol=1e-6), "Z-score normalization failed."

def test_min_max_scaling_default_range(sample_data):
    """
    Test MinMaxScaling class with default range [0.0, 1.0].
    Expected behavior: Data is scaled to the range [0.0, 1.0] along the feature dimension.
    """
    scaler = MinMaxScaling()
    scaled_data = scaler(sample_data)

    # Compute expected values
    x_min = sample_data.min(dim=1, keepdim=True)[0]
    x_max = sample_data.max(dim=1, keepdim=True)[0]
    expected_scaled_data = (sample_data - x_min) / (x_max - x_min + 1e-8)

    # Validate output
    assert torch.allclose(scaled_data, expected_scaled_data, atol=1e-6), "Min-max scaling failed."

def test_min_max_scaling_custom_range(sample_data):
    """
    Test MinMaxScaling class with custom range [-1.0, 1.0].
    Expected behavior: Data is scaled to the range [-1.0, 1.0] along the feature dimension.
    """
    min_value, max_value = -1.0, 1.0
    scaler = MinMaxScaling(min_value=min_value, max_value=max_value)
    scaled_data = scaler(sample_data)

    # Compute expected values
    x_min = sample_data.min(dim=1, keepdim=True)[0]
    x_max = sample_data.max(dim=1, keepdim=True)[0]
    x_scaled = (sample_data - x_min) / (x_max - x_min + 1e-8)
    expected_scaled_data = x_scaled * (max_value - min_value) + min_value

    # Validate output
    assert torch.allclose(scaled_data, expected_scaled_data, atol=1e-6), "Custom min-max scaling failed."

def test_normalize_to_range(sample_data):
    """
    Test NormalizeToRange class with target range [-5.0, 5.0].
    Expected behavior: Data is scaled to the range [-5.0, 5.0] along the feature dimension.
    """
    target_min, target_max = -5.0, 5.0
    normalizer = NormalizeToRange(target_min=target_min, target_max=target_max)
    normalized_data = normalizer(sample_data)

    # Compute expected values
    x_min = sample_data.min(dim=1, keepdim=True)[0]
    x_max = sample_data.max(dim=1, keepdim=True)[0]
    x_normalized = (sample_data - x_min) / (x_max - x_min + 1e-8)
    expected_normalized_data = x_normalized * (target_max - target_min) + target_min

    # Validate output
    assert torch.allclose(normalized_data, expected_normalized_data, atol=1e-6), "Normalize to range failed."

def test_normalize_to_radians(sample_data):
    """
    Test NormalizeToRadians class.
    Expected behavior: Data is scaled to the range [0, 2π] along the feature dimension.
    """
    normalizer = NormalizeToRadians()
    normalized_data = normalizer(sample_data)

    # Compute expected values
    x_min = sample_data.min(dim=1, keepdim=True)[0]
    x_max = sample_data.max(dim=1, keepdim=True)[0]
    x_normalized = (sample_data - x_min) / (x_max - x_min + 1e-8)
    expected_normalized_data = x_normalized * (2 * torch.pi)

    # Validate output
    assert torch.allclose(normalized_data, expected_normalized_data, atol=1e-6), "Normalize to radians failed."

def test_edge_cases():
    """
    Test edge cases such as empty tensors and constant-valued tensors.
    """
    # Empty tensor
    empty_tensor = torch.empty((0, 4))
    normalizer = ZScoreNormalization()
    with pytest.raises(ValueError):  # Expect a ValueError for empty input
        normalized_data = normalizer(empty_tensor)

    # Constant-valued tensor
    constant_tensor = torch.full((2, 4), 3.0)
    scaler = MinMaxScaling()
    scaled_data = scaler(constant_tensor)

    # For constant-valued tensors, min and max are equal, so scaling should result in zeros
    expected_scaled_data = torch.zeros_like(constant_tensor)
    assert torch.allclose(scaled_data, expected_scaled_data, atol=1e-6), "Constant-valued tensor scaling failed."