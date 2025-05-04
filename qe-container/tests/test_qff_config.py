import pytest
from qaintum_qe.utils.qff_config import determine_qnn_parameters, find_optimal_cutoff_and_wires

# Fixtures for common test inputs
@pytest.fixture
def regression_task_params():
    """Fixture for regression task parameters."""
    return {
        "task_type": "regression",
    }

@pytest.fixture
def classification_task_params_small():
    """Fixture for classification task parameters (small num_classes)."""
    return {
        "task_type": "classification",
        "num_classes": 5,
    }

@pytest.fixture
def classification_task_params_large():
    """Fixture for classification task parameters (large num_classes)."""
    return {
        "task_type": "classification",
        "num_classes": 16,
    }

# Tests for `find_optimal_cutoff_and_wires`
def test_find_optimal_cutoff_and_wires():
    """Test that find_optimal_cutoff_and_wires returns valid values."""
    target_output_size = 100
    max_cutoff_dim = 10
    max_num_wires = 10

    cutoff_dim, num_wires = find_optimal_cutoff_and_wires(
        target_output_size=target_output_size,
        max_cutoff_dim=max_cutoff_dim,
        max_num_wires=max_num_wires
    )

    assert cutoff_dim >= 2, "Cutoff dimension should be at least 2."
    assert num_wires >= 1, "Number of wires should be at least 1."
    assert cutoff_dim ** num_wires >= target_output_size, "Hilbert space size should meet the target output size."

def test_find_optimal_cutoff_and_wires_invalid_constraints():
    """Test that find_optimal_cutoff_and_wires raises an error for invalid constraints."""
    with pytest.raises(ValueError, match="No valid cutoff_dim and num_wires found"):
        find_optimal_cutoff_and_wires(target_output_size=1000, max_cutoff_dim=3, max_num_wires=3)

def test_find_optimal_cutoff_and_wires_exact_match():
    """Test when target output size is exactly a power of a cutoff_dim."""
    cutoff_dim, num_wires = find_optimal_cutoff_and_wires(target_output_size=64)
    assert cutoff_dim ** num_wires == 64, "Hilbert space should exactly match target when possible."

# Tests for `determine_qnn_parameters`
def test_determine_qnn_parameters_classification_small_num_classes(classification_task_params_small):
    """Test that determine_qnn_parameters sets num_wires = num_classes for small num_classes (num_classes = 5)."""
    params = determine_qnn_parameters(**classification_task_params_small)
    assert params["num_wires"] == 5, "Number of wires should match num_classes."
    assert params["output_size"] == "multi", "Output size should be 'multi' for small num_classes."

def test_determine_qnn_parameters_classification_large_num_classes(classification_task_params_large):
    """Test that determine_qnn_parameters works correctly for classification tasks (num_classes = 16)."""
    params = determine_qnn_parameters(**classification_task_params_large)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["output_size"] == "probabilities", "Output size should be 'probabilities' for large num_classes."
    assert params["cutoff_dim"]**params["num_wires"] >= classification_task_params_large["num_classes"], "cutoff_dim ** num_wires should be greater than or equal to num_classes."

def test_determine_qnn_parameters_regression(regression_task_params):
    """Test that determine_qnn_parameters works correctly for regression tasks."""
    params = determine_qnn_parameters(**regression_task_params)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["output_size"] == "single", "Output size should be 'single' for regression tasks."

def test_determine_qnn_parameters_unsupported_task_type():
    """Test that determine_qnn_parameters raises an error for unsupported task types."""
    with pytest.raises(ValueError, match="Unsupported task type: invalid_task"):
        determine_qnn_parameters(task_type="invalid_task")
