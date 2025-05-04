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

import torch
import pytest
from qaintum_qe.models.quantum_encoder import QuantumEncoder

@pytest.mark.parametrize("task_type,num_classes,input_shape,expected_output_shape", [
    ("classification", 5, (2, 10), (2, 5)),         # Small classification (multi-class one-hot or logits)
    ("classification", 20, (2, 10), (2, 20)),       # Large classification (flatten & truncate)
    ("regression", None, (2, 10), (2, 1)),          # Regression output
])
def test_quantum_encoder_forward(task_type, num_classes, input_shape, expected_output_shape):
    input_vocab_size = 100
    embed_len = 32
    num_heads = 4

    model = QuantumEncoder(
        input_vocab_size=input_vocab_size,
        embed_len=embed_len,
        num_heads=num_heads,
        task_type=task_type,
        num_classes=num_classes,
        sequence_length=input_shape[1],
        device="cpu"
    )

    dummy_input = torch.randint(0, input_vocab_size, input_shape)  # Simulate token IDs
    output = model(dummy_input)

    assert output.shape == expected_output_shape, f"Expected shape {expected_output_shape}, got {output.shape}"
    assert not torch.isnan(output).any(), "Output contains NaNs"

def test_invalid_task_type_raises():
    with pytest.raises(ValueError):
        _ = QuantumEncoder(
            input_vocab_size=100,
            embed_len=32,
            num_heads=4,
            task_type="invalid_task",
            device="cpu"
        )
