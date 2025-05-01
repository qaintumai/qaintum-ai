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
from qaintum_qt.models.quantum_decoder import QuantumDecoder

class TestQuantumDecoder:
    @pytest.fixture
    def decoder(self):
        """
        Fixture to create a QuantumDecoder instance for testing.
        """
        embed_len = 16
        num_heads = 4
        num_layers = 2
        num_wires = 4
        cutoff_dim = 5
        dropout = 0.1
        return QuantumDecoder(
            embed_len=embed_len,
            num_heads=num_heads,
            num_layers=num_layers,
            num_wires=num_wires,
            cutoff_dim=cutoff_dim,
            dropout=dropout
        )

    def test_initialization(self, decoder):
        """
        Test that the QuantumDecoder is initialized correctly.
        """
        assert isinstance(decoder, QuantumDecoder), "Decoder should be an instance of QuantumDecoder"
        assert decoder.embed_len == 16, "Embedding length should match initialization"
        assert isinstance(decoder.multihead_self_attention, torch.nn.Module), "Self-attention layer should be a PyTorch module"
        assert isinstance(decoder.multihead_enc_dec_attention, torch.nn.Module), "Encoder-decoder attention layer should be a PyTorch module"
        assert isinstance(decoder.first_norm, torch.nn.LayerNorm), "First normalization layer should be LayerNorm"
        assert isinstance(decoder.quantum_feed_forward, torch.nn.Module), "Quantum feed-forward layer should be a PyTorch module"

    def test_forward_pass(self, decoder):
        """
        Test the forward pass of the QuantumDecoder.
        """
        batch_size = 2
        sequence_length = 10
        embed_len = 16

        # Create dummy target and encoder output tensors
        target = torch.randn(batch_size, sequence_length, embed_len)
        encoder_output = torch.randn(batch_size, sequence_length, embed_len)

        # Forward pass
        output = decoder(target, encoder_output)

        # Validate output shape
        assert output.shape == (batch_size, sequence_length, embed_len), "Output shape should match input shape"
