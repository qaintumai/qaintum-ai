import pytest
from qaintum_qt.utils.qff_config import determine_qnn_parameters, find_optimal_cutoff_and_wires

# Fixtures for common test inputs
@pytest.fixture
def sequence_task_params():
    """Fixture for sequence task parameters."""
    return {
        "task_type": "sequence",
        "vocab_size": 100,
        "sequence_length": 10,
    }

@pytest.fixture
def generation_task_params():
    """Fixture for generation task parameters."""
    return {
        "task_type": "generation",
        "vocab_size": 500,
        "sequence_length": 20,
    }

@pytest.fixture
def regression_task_params():
    """Fixture for regression task parameters."""
    return {
        "task_type": "regression",
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

def test_determine_qnn_parameters_missing_vocab_size():
    """Test that determine_qnn_parameters raises an error when vocab_size is missing for sequence tasks."""
    with pytest.raises(ValueError, match="vocab_size must be provided for sequence or generation tasks."):
        determine_qnn_parameters(task_type="sequence", sequence_length=10)

def test_determine_qnn_parameters_missing_sequence_length():
    """Test that determine_qnn_parameters raises an error when sequence_length is missing for sequence tasks."""
    with pytest.raises(ValueError, match="sequence_length must be provided for sequence or generation tasks."):
        determine_qnn_parameters(task_type="sequence", vocab_size=100)

def test_determine_qnn_parameters_missing_num_classes():
    """Test that determine_qnn_parameters raises an error when num_classes is missing for classification tasks."""
    with pytest.raises(ValueError, match="num_classes must be provided for classification tasks."):
        determine_qnn_parameters(task_type="classification")

# Tests for `determine_qnn_parameters`
def test_determine_qnn_parameters_sequence(sequence_task_params):
    """Test that determine_qnn_parameters works correctly for sequence tasks."""
    params = determine_qnn_parameters(**sequence_task_params)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["num_wires"] > 1, "Number of wires should be greater than 1."
    assert params["output_size"] == "probabilities", "Output size should be 'probabilities' for sequence tasks."

def test_determine_qnn_parameters_generation(generation_task_params):
    """Test that determine_qnn_parameters works correctly for generation tasks."""
    params = determine_qnn_parameters(**generation_task_params)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["num_wires"] > 1, "Number of wires should be greater than 1."
    assert params["output_size"] == "probabilities", "Output size should be 'probabilities' for generation tasks."

def test_determine_qnn_parameters_classification_small_num_classes():
    """Test that determine_qnn_parameters sets num_wires = num_classes for small num_classes."""
    params = determine_qnn_parameters(task_type="classification", num_classes=5, cutoff_dim=2)
    assert params["num_wires"] == 5, "Number of wires should match num_classes."
    assert params["output_size"] == "multi", "Output size should be 'multi' for small num_classes."

def test_determine_qnn_parameters_classification_large_num_classes():
    """Test that determine_qnn_parameters works correctly for classification tasks."""
    params = determine_qnn_parameters(task_type="classification", num_classes=50)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["num_wires"] > 1, "Number of wires should be greater than 1."
    assert params["output_size"] == "probabilities", "Output size should be 'probabilities' for classification tasks with <= 20 classes."

def test_determine_qnn_parameters_regression(regression_task_params):
    """Test that determine_qnn_parameters works correctly for regression tasks."""
    params = determine_qnn_parameters(**regression_task_params)
    assert params["cutoff_dim"] > 1, "Cutoff dimension should be greater than 1."
    assert params["output_size"] == "single", "Output size should be 'single' for regression tasks."

def test_determine_qnn_parameters_unsupported_task_type():
    """Test that determine_qnn_parameters raises an error for unsupported task types."""
    with pytest.raises(ValueError, match="Unsupported task type: invalid_task"):
        determine_qnn_parameters(task_type="invalid_task")