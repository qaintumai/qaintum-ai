# Copyright 2024 The qAIntum.ai Authors. All Rights Reserved.
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

# qnn/layers/qnn_data_encoder.py

import pennylane as qml
from itertools import zip_longest
import math
import numpy as np


class QuantumDataEncoder:
    """
    Encodes classical data into quantum states using squeezing, beamsplitter,
    rotation, displacement, and Kerr gates.

    Usage:
        encoder = QuantumDataEncoder(num_wires=8)
        encoder.encode(input_data)
    """

    def __init__(self, num_wires):
        """
        Initializes the QuantumDataEncoder.

        Parameters:
            num_wires (int): Number of quantum wires.
        """
        self.num_wires = num_wires
        self.params_per_round = 8 * num_wires - 2  # Parameters required per encoding cycle

    def encode(self, x):
        """
        Encodes input data into quantum states.

        Parameters:
            x (list or array-like): Input data.

        Raises:
            TypeError: If any element of the input data is not numeric.
        """
        # Validate that all elements of x are numeric
        if not all(isinstance(val, (int, float)) for val in x):
            raise TypeError("All elements of the input data must be numeric.")

        # Special case: If len(x) <= num_wires, apply Squeezing gates directly
        if len(x) <= self.num_wires:
            for i in range(len(x)):
                r = max(0, min(x[i], 5.0))  # Clip squeezing amplitude to [0, 5]
                qml.Squeezing(r, 0.0, wires=i)  # Use input value for r, set phi to 0.0
            return  # Exit early since no further encoding is needed

        rounds = math.ceil(len(x) / self.params_per_round)

        for j in range(rounds):
            start_idx = j * self.params_per_round
            params = x[start_idx:start_idx + self.params_per_round]

            # Pad the parameters with zeros if they are fewer than required
            params.extend([0] * (self.params_per_round - len(params)))

            # Flag to track if a zero parameter is encountered
            terminate_loops = False

            # Apply Squeezing gates
            for i, (r, phi) in zip(range(self.num_wires), zip_longest(params[::2], params[1::2], fillvalue=0)):
                r = max(0, min(r, 5.0))  # Clip squeezing amplitude to [0, 5]
                phi = phi % (2 * np.pi)  # Wrap phase angle to [0, 2π]
                if r == 0 and phi == 0:
                    terminate_loops = True
                    break
                qml.Squeezing(r, phi, wires=i)

            if terminate_loops:
                break

            # Apply Beamsplitter gates
            offset = 2 * self.num_wires
            for (theta, phi), (i, j) in zip(zip_longest(params[offset::2], params[offset+1::2], fillvalue=0),
                                            zip(range(self.num_wires - 1), range(1, self.num_wires))):
                theta = max(0, min(theta, np.pi / 2))  # Clip transmissivity angle to [0, π/2]
                phi = phi % (2 * np.pi)  # Wrap phase angle to [0, 2π]
                if theta == 0 and phi == 0:
                    terminate_loops = True
                    break
                qml.Beamsplitter(theta, phi, wires=[i, j])

            if terminate_loops:
                break

            # Apply Rotation gates
            offset += 2 * (self.num_wires - 1)
            for i, theta in zip(range(self.num_wires), params[offset:offset + self.num_wires]):
                theta = theta % (2 * np.pi)  # Wrap rotation angle to [0, 2π]
                if theta == 0:
                    terminate_loops = True
                    break
                qml.Rotation(theta, wires=i)

            if terminate_loops:
                break

            # Apply Displacement gates
            offset += self.num_wires
            for i, (alpha, phi) in zip(range(self.num_wires), zip_longest(params[offset::2], params[offset+1::2], fillvalue=0)):
                alpha = max(0, min(alpha, 5.0))  # Clip displacement magnitude to [0, 5]
                phi = phi % (2 * np.pi)  # Wrap phase angle to [0, 2π]
                if alpha == 0 and phi == 0:
                    terminate_loops = True
                    break
                qml.Displacement(alpha, phi, wires=i)

            if terminate_loops:
                break

            # Apply Kerr gates
            offset += 2 * self.num_wires
            for i, kappa in zip(range(self.num_wires), params[offset:offset + self.num_wires]):
                kappa = max(-10.0, min(kappa, 10.0))  # Clip nonlinearity strength to [-10, 10]
                if kappa == 0:
                    terminate_loops = True
                    break
                qml.Kerr(kappa, wires=i)

            if terminate_loops:
                break