# **API Reference**

## **Table of Contents**

1. [Introduction](#introduction)
2. [Modules](#modules)
   - [Layers](#layers)
   - [Models](#models)
3. [Layers API](#layers-api)
   - [qnn_circuit.py](#qnn_circuitpy)
   - [qnn_layer.py](#qnn_layerpy)
   - [qnn_data_encoder.py](#qnn_data_encoderpy)
   - [qnn_weight_init.py](#qnn_weight_initpy)
4. [Models API](#models-api)
   - [quantum_neural_network.py](#quantum_neural_networkpy)
5. [Utils API](#utils-api)

---

## **Introduction**

This document provides a detailed reference for the APIs available in the Quantum Neural Network Project. Each module and its respective components are described, including the available classes, their methods, parameters, and usage.

---

## **Modules**

### **Layers**

The Layers module contains core components for building quantum neural networks (QNNs).

**Components:**

- **Quantum Data Encoder**: Converts classical data into quantum states.
- **QNN Layer**: A quantum analog of a classical layer, composed of a weight matrix, bias addition, and non-linear activation.
- **QNN Circuit**: Composes a full quantum circuit from the data encoder and QNN layers. Supports the following output modes:
  - `"single_output"`: Returns a scalar value via expectation measurement on the first wire.
  - `"multi"`: Returns a vector with measurements from all wires.
  - `"probabilities"`: Returns a probability distribution over all basis states.
- **QNN Weight Initializer**: Initializes weights for quantum layers based on optical quantum gate principles.

**Initialization Methods:**

- `normal`: Gaussian distribution with mean 0 and user-defined standard deviations.
- `uniform`: Uniform distribution over a defined range.
- `xavier`: Scales variance by average number of input/output units. Suited for symmetric activations (e.g., `tanh`).
- `kaiming`: Scales variance based on the number of input units. Ideal for ReLU-like activations.

### **Models**

Implements the full quantum neural network model, which wraps the circuit, training, and prediction logic.

---

## **Layers API**

### `qnn_circuit.py`

#### `class QuantumNeuralNetworkCircuit`

**Description:**
Builds a quantum circuit composed of a data encoder and a series of QNN layers.

**Methods:**

- `__init__(self, num_wires, cutoff_dim, num_layers, output_size, init_method, active_sd, passive_sd, gain)`
  Initializes the quantum circuit configuration.

- `initialize_weights(self, init_method, active_sd, passive_sd, gain)`
  Initializes weights using the selected method.

- `build_circuit(self)`
  Constructs the full circuit with layers and encoder.

---

### `qnn_data_encoder.py`

#### `class QuantumDataEncoder`

**Description:**
Encodes classical data into quantum states for further computation.

**Methods:**

- `__init__(self, num_wires)`
  Initializes the encoder with the specified number of wires.

---

### `qnn_layer.py`

#### `class QuantumNeuralNetworkLayer`

**Description:**
Represents a quantum layer using various quantum gates.

**Methods:**

- `__init__(self, num_wires)`
  Initializes a quantum layer using the given number of wires.

- `apply(self, v)`
  Applies a series of quantum gates (interferometers, squeezing, displacement, Kerr) using the given weights `v`.

---

### `qnn_weight_init.py`

#### `class QuantumWeightInitializer`

**Description:**
Initializes weights for QNN layers based on a selected strategy.

**Methods:**

- `__init__(self, method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0)`
  Sets up the initializer with a method and distribution parameters.

- `init_weights(self, layers, num_wires)`
  Generates weight arrays based on method, number of layers, and wires.

**Supported Initialization Methods:**

- **`normal`**: Uses Gaussian distribution centered at 0 with user-defined standard deviation (`active_sd`, `passive_sd`).
- **`uniform`**: Uses a flat distribution over a defined range, useful for non-biased initial states.
- **`xavier`**: Maintains activation variance; best for symmetric activations like `tanh`.
- **`kaiming`**: Preserves forward signal variance in deeper networks; optimized for ReLU-like nonlinearities.

---

## **Models API**

### `quantum_neural_network.py`

#### `class QuantumNeuralNetwork`

**Description:**
This class integrates the data encoder, QNN layers, and circuit into a trainable PyTorch model. It encapsulates initialization, circuit building, and forward-pass computation.

---

**Methods:**

- `__init__(self, num_wires=4, cutoff_dim=5, num_layers=2, output_size="single", init_method='normal', active_sd=0.0001, passive_sd=0.1, gain=1.0, normalize_inputs=True, dropout_rate=0.0)`
  Initializes the Quantum Neural Network model with the specified architecture and weight initialization parameters.

  **Parameters:**
  - `num_wires` (int): Number of quantum wires (modes).
  - `cutoff_dim` (int): Cutoff dimension for Fock space.
  - `num_layers` (int): Number of quantum layers.
  - `output_size` (str): Output type (`"single"`, `"multi"`, or `"probabilities"`).
  - `init_method` (str): Weight initialization method (`"normal"`, `"uniform"`, `"xavier"`, `"kaiming"`).
  - `active_sd` (float): Std. dev. for active gate weights.
  - `passive_sd` (float): Std. dev. for passive gate weights.
  - `gain` (float): Gain factor applied to weights.
  - `normalize_inputs` (bool): Whether to normalize inputs to the circuit.
  - 'dropout_rate' (float): Dropout rate on the output during training

- `_build_quantum_layers(self)`
  Builds the quantum circuit and wraps it as a `TorchLayer`, making it compatible with PyTorch’s training loop.

- `forward(self, inputs)`
  Performs a forward pass through the quantum model, returning predictions based on the selected output mode.

---

## **Utils API**

Additional utilities that support core quantum components (to be expanded as needed).

---
