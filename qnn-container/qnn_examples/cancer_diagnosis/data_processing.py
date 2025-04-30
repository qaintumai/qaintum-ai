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

# qnn_examples/cancer_diagnosis/data_processing.py

def load_data(file_path):
    loaded_data = np.load(file_path)
    return (
        loaded_data['X_train'], loaded_data['y_train'],
        loaded_data['X_val'], loaded_data['y_val'],
        loaded_data['X_test'], loaded_data['y_test'],
        loaded_data['X_test2'], loaded_data['y_test2']
    )

def preprocess_labels(labels, num_classes=2, padding_length=10):
    """
    One-hot encodes the labels and pads them to a specified length.
    Parameters:
    - labels: Array-like, the target labels to encode.
    - num_classes: Number of classes for one-hot encoding (default is 2 for binary classification).
    - padding_length: Total length of the output vector (including padding).
    Returns:
    - Array of one-hot encoded vectors with padding.
    """
    encoder = OneHotEncoder(sparse_output=False, categories=[range(num_classes)])
    one_hot_encoded = encoder.fit_transform(labels.reshape(-1, 1))
    padding = np.zeros((one_hot_encoded.shape[0], padding_length - num_classes))
    return np.hstack([one_hot_encoded, padding])

def shuffle_data(X, y):
    return shuffle(X, y)