import numpy as np
import torch
from qnn.models.quantum_neural_network import QuantumNeuralNetwork, get_model
from data_processing import load_data, preprocess_labels, shuffle_data
from visualization import plotResults
from training_utils import train

def main() -> None:
    # Step 1: Load the data using the load_data function
    file_path = "data/OrData.npz"
    X_train, y_train, X_val, y_val, X_test, y_test, X_test2, y_test2 = load_data(file_path)

    # Print shapes of the datasets
    print('Shape of training data:', X_train.shape)
    print('Shape of training target:', y_train.shape)
    print('Shape of validation data:', X_val.shape)
    print('Shape of validation target:', y_val.shape)

    # Step 2: Shuffle the training and validation data using shuffle_data
    X_train, y_train = shuffle_data(X_train, y_train)
    X_val, y_val = shuffle_data(X_val, y_val)

    # Step 3: Preprocess the labels (one-hot encode with padding) using preprocess_labels
    y_train_padded = preprocess_labels(y_train, num_classes=2, padding_length=10)
    y_val_padded = preprocess_labels(y_val, num_classes=2, padding_length=10)

    # Step 4: Normalize the input data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_val_normalized = scaler.transform(X_val)

    # Step 5: Initialize the quantum neural network
    num_wires = 10
    num_layers = 2
    quantum_nn = QuantumNeuralNetwork(num_wires=num_wires, cutoff_dim=2, num_layers=num_layers)

    # Get the PyTorch model
    model = get_model(quantum_nn=quantum_nn, num_wires=num_wires, num_layers=num_layers)

    # Step 6: Train the model
    history = train(
        X_train=X_train_normalized[:200],  # Use a subset for demonstration purposes
        y_train=y_train_padded[:200],
        X_val=X_val_normalized,
        y_val=y_val_padded,
        model=model,
        batch_size=64,
        learning_rate=0.01,
        epochs=2
    )

    # Step 7: Plot the training results
    plotResults(history=history)

if __name__ == "__main__":
    main()