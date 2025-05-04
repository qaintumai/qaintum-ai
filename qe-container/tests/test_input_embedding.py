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

# qt-container/tests/test_input_embedding.py

import torch
import pytest
from qaintum_qe.layers.input_embedding import InputEmbedding
from qaintum_qe.utils.device import get_device  # Ensure this utility exists and is imported properly

@pytest.fixture
def setup_input_embedding():
    input_vocab_size = 100
    embed_len = 64
    seq_len = 10
    batch_size = 32
    dropout = 0.1
    device = get_device()

    model = InputEmbedding(input_vocab_size, embed_len, dropout, device)
    dummy_input = torch.randint(0, input_vocab_size, (batch_size, seq_len)).to(device)

    return model, dummy_input, batch_size, seq_len, embed_len

def test_input_embedding_output_shape(setup_input_embedding):
    model, dummy_input, batch_size, seq_len, embed_len = setup_input_embedding
    output = model(dummy_input)

    assert output.shape == (batch_size, seq_len, embed_len), \
        f"Expected output shape {(batch_size, seq_len, embed_len)}, but got {output.shape}"

def test_input_embedding_output_type(setup_input_embedding):
    model, dummy_input, _, _, _ = setup_input_embedding
    output = model(dummy_input)

    assert isinstance(output, torch.Tensor), \
        f"Expected output type torch.Tensor, but got {type(output)}"
