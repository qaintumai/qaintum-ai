# User Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
    - [Basic Usage](#basic-usage)
    - [Advanced Features](#advanced-features)
5. [QNN Examples](#qnn-examples)
    - [Bio-oil Projection](#bio-oil-projection)
    - [Cancer Diagnosis](#cancer-diagnosis)
6. [Documentation](#documentation)
8. [Contributing](#contributing)
9. [Support](#license)

---

## Introduction

Welcome to the Quantum Neural Network Project! This guide provides an overview of the Continuous Variable Quantum Neural Networks (CV-QNN) project. It includes instructions for setting up the project, an explanation of the project structure, and examples of how to use the various components.

---

## Project Structure

The project is organized into the following main directories:

- **/docs**: Documentation files
- **/qnn**: QNN source code
  - **/layers**: Contains various layer implementations
    - `qnn_circuit.py`
    - `qnn_data_encoding.py`
    - `qnn_layer.py`
  - **/models**: Contains model definitions
    - `quantum_neural_network.py`
  - **/utils**: Utility functions
    - `normalization.py`
    - `qnn_weight_init.py`
- **/tests**: Test files

---

## Setup Instructions

### **Prerequisites**

Ensure you have the following installed:

- Python 3.10+
- Required Python packages (listed in `requirements.txt`)


### **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/qaintumai/qaintum-ai/qnn-container.git
    cd qnn-container
    ```
2. Create and activate a virtual environment:
    For MacOS
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    For Windows
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

There are basic_usage.py and advanced_features.py in the qnn_examples directory.

### Basic Usage
To initialize the Quantum Neural Network class as a model, run the `basic_usage.py` script:
```bash
python basic_usage.py
```

### Advanced Features
To initialize the Quantum Neural Network class with advanced features, run the `basic_usage.py` script:
```bash
python advanced_features.py

Key Advanced Features:

1. Custom Weight Initialization

The QNN supports multiple weight initialization methods to optimize training:

'normal' (default): Initializes weights using a Gaussian distribution.
'xavier': Scales weights to maintain activation variance, ideal for symmetric activations like tanh.
'kaiming': Optimizes weight scaling for ReLU-like nonlinearities, preserving signal variance in deeper networks.
 2. Input Normalization

The input data can be normalized using various techniques to ensure compatibility with quantum circuits:

Z-Score Normalization : Standardizes inputs to have zero mean and unit variance.
Min-Max Scaling : Scales inputs to a specified range (e.g., [0, 1]).
Normalize to Range : Adjusts inputs to a custom range (e.g., [-1, 1]).
Normalize to Radians : Maps inputs to the range [0, 2π], suitable for quantum gates and periodic functions.
 3. Hybrid Quantum-Classical Models

Combines classical preprocessing layers (e.g., fully connected neural networks) with quantum processing layers for enhanced flexibility and performance.

4. Output Modes

The QNN supports multiple output modes for different use cases:

"single": Returns a scalar value via expectation measurement on the first wire.
"multi": Provides a vector of measurements from all wires.
"probabilities": Outputs a probability distribution over all basis states.
 5. Dropout and Regularization

Includes optional dropout layers to prevent overfitting during training.

By leveraging these advanced features, you can fine-tune the QNN for specific tasks, such as regression, classification, or hybrid quantum-classical workflows. For detailed implementation and examples, refer to the advanced_features.py script. ```
---

## QNN Examples

1. Bio-oil Projection: regression problem for projecting the concentration ratios of bio-oil components.

2. Cancer Diagnosis: classification problem for diagnosis of cancers based on image data.

---

## Documentation

Detailed documentation for each component is available in the `/docs` directory. Key files include:

- `quantum_neural_network_overview.md`: Overview of Quantum Neural Networks
- `api_reference.md`: API reference for the project's modules and functions

---

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` file in the `/docs` directory for guidelines on how to contribute to the project.

---

## Support

If you encounter any issues or have questions, please open an issue on the project's GitHub repository.