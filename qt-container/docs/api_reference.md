## **API Reference**

### **Table of Contents**

1. [Introduction](#introduction)
2. [Modules](#modules)
   * [Layers](#layers)
   * [Models](#models)
3. [Layers API](#layers-api)
   * [input_embedding.py](#input_embeddingpy)
   * [multi_headed_attention.py](#multi_headed_attentionpy)
   * [quantum_feed_forward.py](#quantum_feed_forwardpy)
   * [scaled_dot_product.py](#scaled_dot_productpy)
4. [Models API](#models-api)
   * [quantum_decoder.py](#quantum_decoderpy)
   * [quantum_encoder.py](#quantum_encoderpy)
   * [quantum_neural_network.py](#quantum_neural_networkpy)
   * [quantum_transformer.py](#quantum_transformerpy)


### Introduction
This document provides a detailed reference for the APIs available in the Quantum Project. Each module and its respective components are described in detail, including the available functions, their parameters, return types, and usage examples.

### Modules

#### Layers

The **Layers** module provides the essential building blocks for constructing transformer architectures.
In quantum transformers, most components—such as input embedding, scaled dot-product attention, and multi-head attention—are identical to those in classical transformers.
The key distinction lies in the feed-forward block, which is replaced by a quantum neural network (QNN) to enable quantum-enhanced learning.

**Key Components:**

- `InputEmbedding`: Converts tokens into dense vector representations.
- `ScaledDotProductAttention`: Computes attention weights using scaled dot-product.
- `MultiHeadAttention`: Implements parallel attention heads.
- `QuantumFeedForward`: Replaces the classical feed-forward layer with a quantum neural network.


---

#### Models

The **Models** module implements quantum neural networks and quantum transformers.
It includes quantum encoder and decoder architectures, which are combined to form fully quantum transformer models designed for sequence learning and other advanced tasks.

**Key Components:**

- `QuantumEncoder`: Encodes input sequences using quantum-enhanced layers.
- `QuantumDecoder`: Decodes target sequences with cross-attention to encoder outputs.
- `QuantumTransformer`: End-to-end model that combines the encoder and decoder.


#### input_embedding.py

##### Class: InputEmbedding

* **Description**: A class representing the input embedding layer, which generates embeddings for input data and adds positional encodings. The layer consists of two embedding layers: one for the token embeddings and another for positional encodings. A dropout layer is applied to the combined embeddings for regularization.

* **Methods**:
    * **`__init__(self, input_vocab_size, embed_len, dropout=0.1, device='cpu')`**: Initializes the input embedding layer with the given configuration.
        * **Parameters**:
            - `input_vocab_size` (int): Size of the input vocabulary.
            - `embed_len` (int): Length of the embedding vector.
            - `dropout` (float, optional): Dropout rate for regularization. Default is 0.1.
            - `device` (str, optional): Device to run the model on (`'cpu'` or `'cuda'`). Default is `'cpu'`.

    * **`forward(self, input)`**: Computes the embeddings and positional encodings for the input data.
        * **Parameters**:
            - `input` (torch.Tensor): Input tensor containing the data to be embedded.
        * **Returns**:
            - `torch.Tensor`: Tensor containing the combined token embeddings and positional encodings with dropout applied.

* **Usage**:
    To use the `InputEmbedding` class, import it as follows:
    ```python
    from layers.input_embedding import InputEmbedding
    ```

    Example:
    ```python
    embedding_layer = InputEmbedding(input_vocab_size=10000, embed_len=128)
    output = embedding_layer(input_tensor)
    ```

#### multi_headed_attention.py

##### Class: MultiHeadedAttention

* **Description**: A class representing the multi-headed attention mechanism. This mechanism splits the input into multiple heads, applies scaled dot-product attention to each head, and then concatenates the results. It can optionally apply masking during the attention calculation.

* **Methods**:
    * **`__init__(self, num_heads, embed_len, mask=None)`**: Initializes the multi-headed attention layer with the given configuration.
        * **Parameters**:
            - `num_heads` (int): Number of attention heads.
            - `embed_len` (int): Length of the embedding vector.
            - `mask` (bool, optional): Whether to apply masking. Default is `None`.

    * **`forward(self, queries, keys, values)`**: Computes the multi-headed attention output.
        * **Parameters**:
            - `queries` (torch.Tensor): Tensor containing the queries (batch_size, seq_len, embed_len).
            - `keys` (torch.Tensor): Tensor containing the keys (batch_size, seq_len, embed_len).
            - `values` (torch.Tensor): Tensor containing the values (batch_size, seq_len, embed_len).
        * **Returns**:
            - `torch.Tensor`: Tensor containing the multi-headed attention output.

* **Usage**:
    To use the `MultiHeadedAttention` class, import it as follows:
    ```python
    from layers.multi_headed_attention import MultiHeadedAttention
    ```

    Example:
    ```python
    attention_layer = MultiHeadedAttention(num_heads=8, embed_len=128)
    output = attention_layer(queries, keys, values)
    ```

#### quantum_feed_forward.py

##### Class: QuantumFeedForward

* **Description**: A class that defines a feedforward block for a quantum neural network. The block utilizes a quantum neural network circuit and applies feedforward operations, including dropout and layer normalization.

* **Methods**:
    * **`__init__(self, num_layers, num_wires, cutoff_dim, embed_len, dropout=0.1, output_size="probabilities")`**: Initializes the QuantumFeedForward class with the provided configuration.
        * **Parameters**:
            - `num_layers` (int): Number of layers in the quantum neural network.
            - `num_wires` (int): Number of wires (qubits/qumodes) in the quantum circuit.
            - `cutoff_dim` (int): Cutoff dimension for the Fock space representation.
            - `embed_len` (int): Length of the embedding vector.
            - `dropout` (float, optional): Dropout rate for regularization. Default is 0.1.
            - `output_size` (str, optional): Output type of the quantum circuit ("single", "multi", or "probabilities").

    * **`forward(self, x)`**: Applies the feedforward block to the input tensor, including the quantum circuit, dropout, and layer normalization.
        * **Parameters**:
            - `x` (torch.Tensor): Input tensor.
        * **Returns**:
            - `torch.Tensor`: Output tensor after applying the quantum circuit, dropout, and layer normalization.

* **Usage**:
    To use the `QuantumFeedForward` class, import it as follows:
    ```python
    from layers.quantum_feed_forward import QuantumFeedForward
    ```

    Example:
    ```python
    model = QuantumFeedForward(num_layers=2, num_wires=4, cutoff_dim=10, embed_len=64)
    output = model(input_tensor)
    ```

#### `scaled_dot_product.py`

##### **Class: `ScaledDotProduct`**

**Description:**
Computes scaled dot-product attention, a core mechanism used in Transformer models for calculating the attention weights between query, key, and value vectors.

---

**Parameters:**

- **queries** (`array-like`):
  The query vectors. Typically a 2D array or tensor of shape `(batch_size, num_queries, embed_len)`.

- **keys** (`array-like`):
  The key vectors. Usually a 2D array or tensor of shape `(batch_size, num_keys, embed_len)`.

- **values** (`array-like`):
  The value vectors. Typically a 2D array or tensor of shape `(batch_size, num_values, embed_len)`.

---

**Returns:**
- **Output** (`array-like`):
  The result of the attention mechanism, which is the weighted sum of the value vectors based on the attention scores. The output is typically a tensor of shape `(batch_size, num_queries, embed_len)`.

---

#### **Example Usage:**

```python
from layers.scaled_dot_product import ScaledDotProduct

# Initialize the scaled dot product attention class
scaled_dot_product_attention = ScaledDotProduct(embed_len=128, mask=None)

# Example data for queries, keys, and values
queries = torch.randn(32, 10, 128)  # batch_size=32, num_queries=10, embed_len=128
keys = torch.randn(32, 10, 128)     # batch_size=32, num_keys=10, embed_len=128
values = torch.randn(32, 10, 128)   # batch_size=32, num_values=10, embed_len=128

# Apply attention mechanism
output = scaled_dot_product_attention(queries, keys, values)

### Models API
#### `quantum_decoder.py`

##### **Class: `QuantumDecoder`**

* **Description:**
A class representing a quantum decoder model used for sequence-to-sequence tasks, incorporating quantum feed-forward layers and multi-headed attention mechanisms.

---

* **Methods:**

- **`__init__(self, embed_len, num_heads, num_layers, num_wires, quantum_nn, dropout=0.1, mask=None)`**
  Initializes the QuantumDecoder model with the specified configuration.

  **Parameters:**
  - **`embed_len`** (`int`): Length of the embedding vector.
  - **`num_heads`** (`int`): Number of attention heads in multi-headed attention layers.
  - **`num_layers`** (`int`): Number of layers in the decoder (not used in this class directly but might be intended for future extensions).
  - **`num_wires`** (`int`): Number of quantum wires used for the quantum neural network component.
  - **`quantum_nn`** (`QuantumNeuralNetwork`): The quantum neural network used for the feed-forward layer.
  - **`dropout`** (`float`, optional): Dropout probability for regularization. Default is 0.1.
  - **`mask`** (`torch.Tensor`, optional): Mask tensor used for attention operations. Default is `None`.

- **`forward(self, target, encoder_output)`**
  Performs the forward pass through the QuantumDecoder, applying self-attention, encoder-decoder attention, and quantum feed-forward layers.

  **Parameters:**
  - **`target`** (`torch.Tensor`): Input tensor containing the target sequence (usually from the previous decoder step).
  - **`encoder_output`** (`torch.Tensor`): Output tensor from the encoder.

  **Returns:**
  - **`torch.Tensor`**: The output after applying the quantum feed-forward operation on the decoder's second sublayer output.

---

#### **Example Usage:**

```python
from qaintum_qt.models.quantum_decoder import QuantumDecoder
from layers.multi_headed_attention import MultiHeadedAttention
from layers.quantum_feed_forward import QuantumFeedForward

# Initialize the QuantumDecoder model
quantum_decoder = QuantumDecoder(embed_len=128,
                                  num_heads=8,
                                  num_layers=6,
                                  num_wires=4,
                                  quantum_nn=None,
                                  dropout=0.1)

# Example input tensors
target = torch.randn(32, 10, 128)  # batch_size=32, num_queries=10, embed_len=128
encoder_output = torch.randn(32, 10, 128)  # batch_size=32, num_keys=10, embed_len=128

# Apply the forward pass
output = quantum_decoder(target, encoder_output)


#### `quantum_encoder.py`

##### **Class: `QuantumEncoder`**

**Description:**
A class representing a quantum encoder model used for sequence-to-sequence tasks, incorporating multi-headed attention mechanisms and quantum feed-forward layers.

---

**Methods:**

- **`__init__(self, embed_len, num_heads, num_layers, num_wires, dropout=0.1, mask=None)`**
  Initializes the QuantumEncoder model with the specified configuration.

  **Parameters:**
  - **`embed_len`** (`int`): Length of the embedding vector.
  - **`num_heads`** (`int`): Number of attention heads in the multi-headed attention layer.
  - **`num_layers`** (`int`): Number of layers in the encoder (not directly used but can be extended for future layers).
  - **`num_wires`** (`int`): Number of quantum wires used for the quantum neural network component.
  - **`dropout`** (`float`, optional): Dropout probability for regularization. Default is 0.1.
  - **`mask`** (`torch.Tensor`, optional): Mask tensor used for attention operations. Default is `None`.

- **`forward(self, queries, keys, values)`**
  Performs the forward pass through the QuantumEncoder, applying multi-headed attention and quantum feed-forward layers.

  **Parameters:**
  - **`queries`** (`torch.Tensor`): Tensor containing the query vectors.
  - **`keys`** (`torch.Tensor`): Tensor containing the key vectors.
  - **`values`** (`torch.Tensor`): Tensor containing the value vectors.

  **Returns:**
  - **`torch.Tensor`**: The output after applying the quantum feed-forward operation on the encoder's first sublayer output.

---

#### **Example Usage:**

```python
from qaintum_qt.models.quantum_encoder import QuantumEncoder
from layers.multi_headed_attention import MultiHeadedAttention
from models.quantum_feed_forward import QuantumFeedForward

# Initialize the QuantumEncoder model
quantum_encoder = QuantumEncoder(embed_len=128,
                                  num_heads=8,
                                  num_layers=6,
                                  num_wires=4,
                                  dropout=0.1)

# Example input tensors
queries = torch.randn(32, 10, 128)  # batch_size=32, num_queries=10, embed_len=128
keys = torch.randn(32, 10, 128)     # batch_size=32, num_keys=10, embed_len=128
values = torch.randn(32, 10, 128)   # batch_size=32, num_values=10, embed_len=128

# Apply the forward pass
output = quantum_encoder(queries, keys, values)


#### `quantum_transformer.py`

##### **Class: `QuantumTransformer`**

**Description:**
A class representing a quantum transformer model that incorporates quantum neural network (QNN) layers in both the encoder and decoder components. It utilizes multi-head attention mechanisms, quantum feed-forward layers, and quantum neural networks for advanced sequence modeling tasks.

---

**Methods:**

- **`__init__(self, num_encoder_layers, num_decoder_layers, embed_len, num_heads, num_layers, num_wires, cutoff_dim, batch_size, vocab_size, output_size="probabilities", dropout=0.1, device='cpu')`**
  Initializes the QuantumTransformer model with the given configuration.

  **Parameters:**
  - **`num_encoder_layers`** (`int`): Number of layers in the encoder.
  - **`num_decoder_layers`** (`int`): Number of layers in the decoder.
  - **`embed_len`** (`int`): Length of the embedding vector.
  - **`num_heads`** (`int`): Number of attention heads in the multi-headed attention layer.
  - **`num_layers`** (`int`): Number of layers in the quantum neural network.
  - **`num_wires`** (`int`): Number of quantum wires used in the quantum neural network.
  - **`cutoff_dim`** (`int`): Cutoff dimension for the quantum neural network.
  - **`batch_size`** (`int`): Batch size for training.
  - **`vocab_size`** (`int`): Vocabulary size for the model's input and output.
  - **`output_size`** (`str`, optional): The output size format for the quantum neural network. Default is `"probabilities"`.
  - **`dropout`** (`float`, optional): Dropout probability for regularization. Default is 0.1.
  - **`device`** (`str`, optional): Device to run the model on (e.g., `"cpu"`, `"cuda"`). Default is `"cpu"`.

- **`encode(self, data)`**
  Encodes the input data through the quantum transformer model.

  **Parameters:**
  - **`data`** (`array-like`): The input data to be encoded.

  **Returns:**
  - **`encoded_data`** (`array-like`): The encoded representation of the input data.

- **`forward(self, src, tgt)`**
  Performs the forward pass through the QuantumTransformer, applying the encoder and decoder layers sequentially.

  **Parameters:**
  - **`src`** (`torch.Tensor`): The source input tensor.
  - **`tgt`** (`torch.Tensor`): The target input tensor.

  **Returns:**
  - **`torch.Tensor`**: The final output after passing through the transformer model's encoder-decoder stack and output linear layer.

---

#### **Example Usage:**

```python
from qaintum_qt.models.quantum_transformer import QuantumTransformer
import torch

# Initialize the QuantumTransformer model
quantum_transformer = QuantumTransformer(
    num_encoder_layers=6,
    num_decoder_layers=6,
    embed_len=128,
    num_heads=8,
    num_layers=2,
    num_wires=4,
    cutoff_dim=16,
    batch_size=32,
    vocab_size=10000,
    output_size="probabilities",
    dropout=0.1,
    device="cuda"
)

# Example input tensors
src = torch.randint(0, 10000, (32, 10))  # batch_size=32, sequence_len=10
tgt = torch.randint(0, 10000, (32, 10))  # batch_size=32, sequence_len=10

# Apply the forward pass
output = quantum_transformer(src, tgt)


