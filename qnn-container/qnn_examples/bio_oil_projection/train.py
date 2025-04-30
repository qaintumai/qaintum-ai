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

# qnn_examples/bio_oil_projection/train.py

import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from .custom_encoder import CustomFeatureEncoder
from qnn.models.quantum_neural_network import QuantumNeuralNetwork

# Original Google Drive shareable link
url = "https://drive.google.com/uc?id=10dUF-cSRUfE5SzMuCSjiVwvW8FytBpHl"

# Load the CSV directly
df = pd.read_csv(url)

# Preprocessing: Drop unnecessary columns and duplicates
df.drop(axis=1, columns=['Biomass', 'Ref.', 'Doi'], inplace=True)
df.drop_duplicates(inplace=True)

# Define features (inputs) and targets (multi-output)
X = df[['C (%)', 'H (%)', 'O (%)', 'S (%)', 'N (%)', 'HHV_bio (MJ/kg)', 'T (°C)', 'RT (min)']].values
y = df[['Yield_oil (%)', 'C_oil (%)', 'H_oil (%)', 'O_oil (%)', 'N_oil (%)', 'HHV_oil (MJ/kg)', 'ER_oil (%)']].values

# Zero-pad y to make it a vector of length 8
y_padded = np.hstack([y, np.zeros((y.shape[0], 1))])  # Append a column of zeros

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_padded, test_size=0.2, random_state=42)

# Normalize angle and magnitude features
def normalize_angle_magnitude(X_train, angle_dim=5):
    angle_scaler = MinMaxScaler(feature_range=(0, 2 * np.pi))
    magnitude_scaler = MinMaxScaler(feature_range=(0, 1))

    angle_features = X_train[:, :angle_dim]
    magnitude_features = X_train[:, angle_dim:]

    angle_features_normalized = angle_scaler.fit_transform(angle_features)
    magnitude_features_normalized = magnitude_scaler.fit_transform(magnitude_features)

    features_normalized = np.hstack([angle_features_normalized, magnitude_features_normalized])
    return features_normalized

# Normalize training and testing features
X_train_normalized = normalize_angle_magnitude(X_train)
X_test_normalized = normalize_angle_magnitude(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Initialize the custom encoder
num_wires = 8
custom_encoder = CustomFeatureEncoder(num_wires=num_wires)

# Initialize the quantum neural network with 8 wires and multi-output
model = QuantumNeuralNetwork(
    num_wires=num_wires,
    cutoff_dim=5,
    num_layers=2,
    output_size="multi",  # Multi-output regression
    init_method="xavier",
    active_sd=0.001,
    passive_sd=0.2,
    gain=1.0,
    normalize_inputs=False,  # Inputs are already normalized
    dropout_rate=0.1,
    encoder=custom_encoder
)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()  # Mean Squared Error Loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 50
batch_size = 32
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    # Mini-batch training
    for i in range(0, len(X_train_tensor), batch_size):
        X_batch = X_train_tensor[i:i + batch_size]
        y_batch = y_train_tensor[i:i + batch_size]

        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Print average loss for the epoch
    avg_loss = total_loss / (len(X_train_tensor) // batch_size)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Evaluation on test data
model.eval()
with torch.no_grad():
    test_outputs = model(X_test_tensor)
    test_loss = criterion(test_outputs, y_test_tensor)
    print(f"Test Loss: {test_loss.item():.4f}")