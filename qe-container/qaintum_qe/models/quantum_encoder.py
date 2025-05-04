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

# qaintum_qe/models/quantum_encoder.py

from torch import nn
from qaintum_qe.layers.multi_headed_attention import MultiHeadedAttention
from qaintum_qe.layers.quantum_feed_forward import QuantumFeedForward
from qaintum_qe.layers.input_embedding import InputEmbedding

class QuantumEncoder(nn.Module):
    def __init__(self, input_vocab_size, embed_len, num_heads, task_type, num_classes=None, sequence_length=None, init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0,
                 normalize_inputs=True, dropout_rate=0.0, mask=None, device='cpu', **kwargs):
        super(QuantumEncoder, self).__init__()

        # Input embedding layer for text-based data (e.g., NLP)
        self.input_embedding = InputEmbedding(input_vocab_size=input_vocab_size, embed_len=embed_len, dropout=dropout_rate, device=device)

        # Quantum layers
        self.multihead = MultiHeadedAttention(num_heads, embed_len, mask)
        self.first_norm = nn.LayerNorm(embed_len)
        self.dropout_layer = nn.Dropout(p=dropout_rate)

        # Quantum Feed-Forward Layer for classification or regression tasks
        self.quantum_feed_forward = QuantumFeedForward(
            task_type=task_type,
            num_classes=num_classes,
            init_method=init_method,
            active_sd=active_sd,
            passive_sd=passive_sd,
            gain=gain,
            normalize_inputs=normalize_inputs,
            dropout_rate=dropout_rate
        )

    def forward(self, input):
        # Apply the embedding layer to the input (converting tokens to embeddings)
        embedded_input = self.input_embedding(input)

        # Attention mechanism (Multi-Headed Attention)
        attention_output = self.multihead(embedded_input, embedded_input, embedded_input)
        attention_output = self.dropout_layer(attention_output)

        # Layer normalization and adding residual connection
        first_sublayer_output = self.first_norm(attention_output + embedded_input)

        # Apply quantum feed-forward layer (for final output based on task type)
        quantum_output = self.quantum_feed_forward(first_sublayer_output)

        # Flatten and truncate output for classification tasks with many classes
        if hasattr(self.quantum_feed_forward, 'task_type') and self.quantum_feed_forward.task_type == "classification":
            if hasattr(self.quantum_feed_forward, 'num_classes') and self.quantum_feed_forward.num_classes and self.quantum_feed_forward.num_classes > 10:
                # Flatten and truncate
                quantum_output = quantum_output.view(quantum_output.size(0), -1)[:, :self.quantum_feed_forward.num_classes]

        return quantum_output
