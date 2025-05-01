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

# tests/test_qnn_weight_init.py

import pytest
import torch
import numpy as np
from qaintum_qnn.utils.qnn_weight_init import QuantumWeightInitializer


@pytest.mark.parametrize("method", ["normal", "uniform", "xavier", "kaiming"])
def test_weight_shapes_and_types(method):
    layers = 3
    num_wires = 4
    initializer = QuantumWeightInitializer(method=method)

    weights = initializer.init_weights(layers=layers, num_wires=num_wires)

    # Expected shape calculation
    M = (num_wires - 1) * 2 + num_wires
    expected_features = M * 2 + num_wires * 3  # int1, int2, s, dr, k
    expected_shape = (layers, expected_features)

    assert isinstance(weights, torch.Tensor)
    assert weights.shape == expected_shape
    assert weights.dtype == torch.float32


def test_invalid_method_raises_error():
    with pytest.raises(ValueError, match="Unsupported initialization method"):
        initializer = QuantumWeightInitializer(method="invalid")
        initializer.init_weights(layers=2, num_wires=3)


def test_default_params_match_normal():
    layers = 2
    num_wires = 3
    default_initializer = QuantumWeightInitializer()
    normal_initializer = QuantumWeightInitializer(method="normal")

    default_weights = default_initializer.init_weights(layers, num_wires)
    normal_weights = normal_initializer.init_weights(layers, num_wires)

    assert default_weights.shape == normal_weights.shape
    assert torch.allclose(default_weights, default_weights, atol=1e-5)


def test_weight_variation_across_methods():
    layers = 2
    num_wires = 3
    methods = ["normal", "uniform", "xavier", "kaiming"]

    weight_sets = []
    for method in methods:
        initializer = QuantumWeightInitializer(method=method)
        weights = initializer.init_weights(layers, num_wires)
        weight_sets.append(weights)

    # Ensure all weight sets are different
    for i in range(len(weight_sets)):
        for j in range(i + 1, len(weight_sets)):
            assert not torch.allclose(weight_sets[i], weight_sets[j], atol=1e-4), \
                f"Weights for {methods[i]} and {methods[j]} should differ."


def test_weight_value_ranges_for_uniform():
    layers = 2
    num_wires = 3
    active_sd = 0.005
    passive_sd = 0.05
    initializer = QuantumWeightInitializer(method="uniform", active_sd=active_sd, passive_sd=passive_sd)
    weights = initializer.init_weights(layers, num_wires)

    assert torch.all(weights <= passive_sd)
    assert torch.all(weights >= -passive_sd)
