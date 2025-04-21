# User Guide

---

## Table of Contents
1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
4. [Usage](#usage)
5. [Example Notebooks](#example-notebooks)
6. [Documentation](#examples)
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
    - `qnn_weight_init.py`
  - **/models**: Contains model definitions
    - `quantum_neural_network.py`
  - **/utils**: Utility functions to be added
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
    git clone https://github.com/qaintumai/quantum.git
    cd quantum
    ```
2. Create and activate a virtual environment:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

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