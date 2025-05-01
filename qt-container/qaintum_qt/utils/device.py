import torch

def get_device():
    # Check if CUDA (GPU) is available, otherwise use CPU
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')