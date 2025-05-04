# Copyright 2025 The qAIntum.ai Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# http://www.apache.org/licenses/LICENSE-2.0
# ==============================================================================

import torch
import pytest
from qaintum_qt.utils import get_device

import warnings
warnings.filterwarnings("ignore", message="The retworkx package is deprecated")
warnings.filterwarnings("ignore", category=DeprecationWarning)

def test_get_device_returns_valid_torch_device():
    """Test that get_device returns a valid torch.device instance and type."""
    device = get_device()

    assert isinstance(device, torch.device), \
        f"Expected torch.device, got {type(device)}"

    assert device.type in ["cpu", "cuda"], \
        f"Expected device type to be 'cpu' or 'cuda', got {device.type}"
