# User Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Documentation](#documentation)
5. [Contributing](#contributing)
6. [Support](#license)

---

## Introduction

Welcome to the Quantum Neural Network Project! This guide provides an overview of the Continuous Variable Quantum Neural Networks (CV-QNN) project. The project is a **classical simulation** of CV-QNN using quantum optical gates that can be run on Xanadu's X8. It includes instructions for setting up the project and an explanation of the project structure. Examples of how to use the various components can be found on https://github.com/qaintumai/qaintum-ai/tree/main/qnn-container/qnn_examples.

---

## Project Structure

The project is organized into the following main directories:

- **/qnn**: QNN source code
  - **/docs**: Documentation files
    - `api_reference.md`
    - `quantum_neural_network.md`
    - `user_guide.md`
  - **/layers**: Contains various layer implementations
    - `qnn_circuit.py`
    - `qnn_data_encoding.py`
    - `qnn_layer.py`
  - **/models**: Contains model definitions
    - `quantum_neural_network.py`
  - **/utils**: Utility functions
    - `normalization.py`
    - `qnn_weight_init.py`

---

## Setup Instructions

### **Prerequisites**

Ensure you have the following installed:

- Python 3.10+
- Required Python packages (listed in `requirements.txt`)


### **Installation**

1. Clone the repository:
    ```bash
    git clone https://github.com/qaintumai/qaintum-ai.git
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

Detailed documentation is available in the `/docs` directory. Key files include:

- [API Reference](https://github.com/qaintumai/qaintum-ai/blob/main/qnn-container/docs/api_reference.md): Comprehensive descriptions of all classes and methods.
- [User Guide](https://github.com/qaintumai/qaintum-ai/blob/main/qnn-container/docs/user_guide.md): Step-by-step instructions for setting up and using the package.
- [Quantum Neural Networks Overview](hhttps://github.com/qaintumai/qaintum-ai/blob/main/qnn-container/docs/quantum_neural_network_overview.md): Theoretical background and design principles.
- [Examples](https://github.com/qaintumai/qaintum-ai/tree/main/qnn-container/qnn_examples): Practical usage examples.
- [Contributing Guidelines](https://github.com/qaintumai/qaintum-ai/blob/main/CONTRIBUTING.md): How to contribute to the project.
- [License](https://github.com/qaintumai/qaintum-ai/blob/main/LICENSE): Terms of use for the package.
---

## Contributing

Contributions are welcome! Please refer to the `CONTRIBUTING.md` [file](https://github.com/qaintumai/qaintum-ai/blob/main/CONTRIBUTING.md) in the `/docs` directory for guidelines on how to contribute to the project.

---

## Support

If you encounter any issues, have questions, or would like to request features, please visit our [GitHub repository](https://github.com/qaintumai/qaintum-ai/qnn-container). You can:

- **Open an Issue**: Use the [Issues tab](https://github.com/qaintumai/qaintum-ai/issues) to report bugs, request enhancements, or ask questions.
- **Check Existing Issues**: Before opening a new issue, search the existing ones to see if your question has already been addressed.
- **Contribute**: If you’d like to contribute to the project, please refer to our [Contributing Guidelines](#contributing).

For additional help, you can also explore the following resources:

- **Documentation**: Detailed documentation is available in the `/docs` directory ([API Reference](#documentation)).
- **Examples**: Practical usage examples are provided in the `qnn_examples` directory ([QNN Examples](https://github.com/qaintumai/qaintum-ai/tree/main/qnn-container/qnn_examples)).

We appreciate your feedback and contributions to improve the Quantum Neural Network Project!