import pennylane as qml

class CustomFeatureEncoder:
    """Encodes input features into quantum states."""
    def __init__(self, num_wires):
        self.num_wires = num_wires

    def encode(self, x):
        for i in range(self.num_wires):
            if i < 5:
                qml.Rotation(x[i], wires=i)
            elif i == 5:
                qml.Squeezing(x[i], 0, wires=i)
            elif i == 6:
                qml.Displacement(x[i], 0, wires=i)
            elif i == 7:
                qml.Kerr(x[i], wires=i)

