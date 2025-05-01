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

# qaintum_qnn/utils/normalization.py

import torch
import torch.nn as nn

class ZScoreNormalization(nn.Module):
    """
    Normalizes input data using Z-score normalization (standardization).
    Formula: z = (x - mean) / std
    """
    def forward(self, x):
        """
        Applies Z-score normalization to the input tensor.
        Parameters:
        - x: Input tensor of shape (batch_size, num_features)
        Returns:
        - Normalized tensor of the same shape
        Raises:
        - ValueError: If the input tensor is empty.
        """
        # Validate that the input tensor is not empty
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        mean = x.mean(dim=1, keepdim=True)  # Compute mean along the feature dimension
        std = x.std(dim=1, keepdim=True)    # Compute standard deviation along the feature dimension
        x_standardized = (x - mean) / (std + 1e-8)  # Add small value to prevent division by zero
        return x_standardized


class MinMaxScaling(nn.Module):
    """
    Normalizes input data using min-max scaling to a specified range [min_value, max_value].
    Formula: x_scaled = (x - x_min) / (x_max - x_min) * (max_value - min_value) + min_value
    """
    def __init__(self, min_value=0.0, max_value=1.0):
        """
        Initializes the MinMaxScaling layer.
        Parameters:
        - min_value: Minimum value of the scaled range (default: 0.0)
        - max_value: Maximum value of the scaled range (default: 1.0)
        """
        super(MinMaxScaling, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def forward(self, x):
        """
        Applies min-max scaling to the input tensor.
        Parameters:
        - x: Input tensor of shape (batch_size, num_features)
        Returns:
        - Scaled tensor of the same shape
        Raises:
        - ValueError: If the input tensor is empty.
        """
        # Validate that the input tensor is not empty
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        x_min = x.min(dim=1, keepdim=True)[0]  # Compute minimum along the feature dimension
        x_max = x.max(dim=1, keepdim=True)[0]  # Compute maximum along the feature dimension
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)  # Add small value to prevent division by zero
        x_scaled = x_scaled * (self.max_value - self.min_value) + self.min_value  # Scale to desired range
        return x_scaled


class NormalizeToRange(nn.Module):
    """
    Normalizes input data to a specific range [target_min, target_max].
    Formula: x_normalized = (x - x_min) / (x_max - x_min) * (target_max - target_min) + target_min
    """
    def __init__(self, target_min=0.0, target_max=1.0):
        """
        Initializes the NormalizeToRange layer.
        Parameters:
        - target_min: Minimum value of the target range (default: 0.0)
        - target_max: Maximum value of the target range (default: 1.0)
        """
        super(NormalizeToRange, self).__init__()
        self.target_min = target_min
        self.target_max = target_max

    def forward(self, x):
        """
        Applies normalization to the specified range.
        Parameters:
        - x: Input tensor of shape (batch_size, num_features)
        Returns:
        - Normalized tensor of the same shape
        Raises:
        - ValueError: If the input tensor is empty.
        """
        # Validate that the input tensor is not empty
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        x_min = x.min(dim=1, keepdim=True)[0]  # Compute minimum along the feature dimension
        x_max = x.max(dim=1, keepdim=True)[0]  # Compute maximum along the feature dimension
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)  # Add small value to prevent division by zero
        x_normalized = x_normalized * (self.target_max - self.target_min) + self.target_min  # Scale to target range
        return x_normalized


class NormalizeToRadians(nn.Module):
    """
    Normalizes input data to the range [0, 2π], suitable for quantum circuits or periodic functions.
    Formula: x_normalized = (x - x_min) / (x_max - x_min) * 2π
    """
    def forward(self, x):
        """
        Normalizes input data to the range [0, 2π].
        Parameters:
        - x: Input tensor of shape (batch_size, num_features)
        Returns:
        - Normalized tensor in the range [0, 2π]
        Raises:
        - ValueError: If the input tensor is empty.
        """
        # Validate that the input tensor is not empty
        if x.numel() == 0:
            raise ValueError("Input tensor cannot be empty.")

        x_min = x.min(dim=1, keepdim=True)[0]  # Compute minimum along the feature dimension
        x_max = x.max(dim=1, keepdim=True)[0]  # Compute maximum along the feature dimension
        x_normalized = (x - x_min) / (x_max - x_min + 1e-8)  # Add small value to prevent division by zero
        x_normalized = x_normalized * (2 * torch.pi)  # Scale to [0, 2π]
        return x_normalized