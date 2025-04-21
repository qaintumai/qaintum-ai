# QAI: Quantum AI Framework
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-%3E=3.8-blue)
![PennyLane](https://img.shields.io/badge/PennyLane-0.29.1-green)
![Torch](https://img.shields.io/badge/PyTorch-2.2.2-red)

## **Overview**
This folder contains the code for Quantum AI Quantum Neural Networks (QNN) based on the Continuous Variable (CV) model of Quantum Computing implemented using Quantum Optics. This framework provides modular components for building, training, and deploying Quantum Neural Networks (QNNs) for classification and regression tasks and Quantum Transformers (QTs) for sequential data generation tasks for Generative AI.

National Science Foundation (NSF) and Department of Energy (DOE) have been designing a blueprint for Quantum Internet on top of classical internet infrastructure as seen in these papers:
* [Development of Quantum InterConnects (QuICs) for Next-Generation Information Technologies, 2019](https://arxiv.org/pdf/1912.06642)
* [A Roadmap for Quantum Interconnects, 2022](https://publications.anl.gov/anlpubs/2022/12/179439.pdf)

Quantum InterConnects are composed of
* Quantum Communications
* Quantum Computing
* Quantum Memory
* Transducers
* Quantum Sensing.

Quantum computing is a method of computation that utilizes physical mediums operating based on the principles of quantum mechanics. There are two types of quantum computing:
* Discrete Variable (DV) Quantum Computing: quantizing digital computing based on the binary logic. The basis states are |0> and |1>.
* Continuous Variable Quantum Computing: using the continuous properties of nature. The basis states are |0>, |1>, ..., |n>

The difference between DV quantum computing and CV quantum computing is stated in this paper: [Quantum computing overview](https://arxiv.org/pdf/2206.07246).

The implementation of CV quantum computing was pioneered by Xanadu in 2020 using quantum optics. The architecture of the Photonic CV chip is detailed in [paper](https://arxiv.org/abs/2103.02109). The key benefits of CV QC are:
* Compatible with the existing communications infrastructure.
* Operates at room-temperature.
* Higher dimensional computation space.
* Easy to network and multiplex
* Low cost of mass production
* Mountable on smartphones, laptops, and edge devices.

## **Components**
- **Quantum Neural Network (QNN)**:
CV Quantum Neural Networks (QNNs) is a framework for quantum-enhanced machine learning, designed to integrate quantum computing with classical deep learning models. Photonic QNN faithfully implements classical neural networks due to optical gates capable of representing bias addition and nonlinear activation function. The optical quantum gates for implementing a neural network layer:
* weight matrix: interferometer + squeezing + interferometer
* bias addition: displacement
* nonlinear activation function: Kerr effect
is outline in [paper](https://arxiv.org/abs/1806.06871).
QNNs require a substantially reduced number of parameters to train due to its inherent parallelism from the superposition property of quantu mechanics.
- **Quantum Transformer (QT)**:
Quantum Transformers (QTs) extend transformer-based architectures by replacing the feedforward layers with QNNs. Key components of a transformer include:

 * Input Encoding: Input embeddings with positional encoding for contextual understanding.

 * Multi-Headed Attention: Query, Key, and Value mechanisms.

 * Feedforward Layer: Classical neural network, replaced by QNN for improved efficiency and parameter reduction.

- **Quantum Small Language Model (QSML) (Coming Soon)**:
QSML is designed as a compact, enterprise-grade language model leveraging quantum-enhanced transformers. Unlike traditional large language models, QSML utilizes optimized transformer blocks combined with quantum-enhanced learning techniques to provide efficient, domain-specific AI solutions.

## **Key Features**
- 🧠 **Quantum Models**: Leverages quantum circuits for feature encoding and processing.
- 🔥 **PyTorch & PennyLane Integration**: Seamlessly works with **PyTorch** for training and **PennyLane** for quantum circuit simulations.
- ⚡ **Custom Quantum Layers**: Provides pre-built layers like **Quantum Parametrized Circuits** with parameterized quantum gates whose parameters are learned through training.
- 📡 **Supports Multiple Quantum Devices**: Compatible with simulators and real quantum hardware (Xanadu's X8) via Pennylane.

## Getting Started

### Clone the Repository
```sh
git clone https://github.com/qaintumai/qaintum-ai.git
cd quantum
```

### Create a Branch
```sh
git checkout -b <new_branch_name>
```

### Virtual Environment
```shell
python3 -m venv venv
source venv/bin/activate
```

### Dependency Installation

#### Third-party Dependencies

- PyTorch
- Pennylane (**0.29.1**)
- Scikit Learn
- Pandas
- Numpy

```sh
pip install -r requirements.txt
```
* Caution: Sometimes, pip may default to a user installation outside the virtual environment instead of installing packages within the virtual environment's site-packages. To avoid this, you can run

```sh
/Users/<your_user_directory_name>/<path_where_you_stored_quantum>/quantum/venv/bin/pip install -r requirements.txt
```
or

```sh
/Users/<your_user_directory_name>/<path_where_you_stored_quantum>/quantum/venv/bin/pip install --no-user -r requirements.txt
```

### Running

```sh
pip install -e .
./examples/basic_usage.py
```

### Make Changes
Edit, add, or remove files as needed in your project. For example, you might edit a file called basic_usage.py

### Stage the Changes
Add the files you changed to the staging area.
```sh
git add <directory_where_the_changed_file_is_located>/<file_name_with_changes>
```

If you made changes to multiple files, to stage all changes you can use:
```sh
git add .
```

### Commit the Changes
Commit the staged changes with a descriptive commit message.
```sh
git commit -m "Add changes to <file_name_with_changes>"
```

### Push the New Branch to 'quantum'
Push your new branch with the changes to the 'quantum' repository.
```sh
git push origin <new_branch_name>

```

### Contributing and Best Practices

Follow [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)

#### Coming Soon

- PyPi
- Type checking with mypy
- Linting with flake8


