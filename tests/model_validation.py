import inspect
import numpy as np
from stream.utils import TMDataset
from stream.models.abstract_helper_models.base import TrainingStatus


class ModelLoggerValidator:
    def __init__(self, model_instance):
        self.model_instance = model_instance
        self.status_changes = []

    def log_status_change(self, status):
        self.status_changes.append(status)

    def validate_status_changes(self):
        expected_sequence = [
            TrainingStatus.NOT_STARTED,
            TrainingStatus.INITIALIZED,
            TrainingStatus.RUNNING,
            TrainingStatus.SUCCEEDED,
        ]
        return self.status_changes[:2] == expected_sequence[:2] and (
            self.status_changes[-1] in expected_sequence[2:]
        )


def validate_model(ModelClass, **model_args):
    """
    Validates if the provided model class adheres to the required guidelines.

    Parameters
    ----------
    ModelClass : type
        The model class to be validated.

    Returns
    -------
    bool
        True if the model class adheres to the guidelines, False otherwise.

    Raises
    ------
    AssertionError
        If the model class does not adhere to the guidelines.
    """
    required_methods = ["get_info", "fit", "predict"]
    required_attributes = ["topic_dict"]
    optional_attributes = ["beta", "theta"]
    alternative_methods = ["get_beta", "get_theta"]

    # Check for required methods
    for method in required_methods:
        assert hasattr(
            ModelClass, method
        ), f"Model class must have a '{method}' method."
        assert callable(
            getattr(ModelClass, method)
        ), f"'{method}' must be a callable method."
        print(f"Check for method {method}: passed")

    # Create a dummy dataset for testing
    dataset = TMDataset()
    dataset.fetch_dataset("DummyDataset")
    print("Check for dataset creation: passed")

    # Instantiate the model and validator
    model = ModelClass(**model_args)
    validator = ModelLoggerValidator(model)

    # Validate the get_info method
    info = model.get_info()
    assert isinstance(info, dict), "'get_info' must return a dictionary."
    required_info_keys = ["model_name", "trained"]
    for key in required_info_keys:
        assert (
            key in info
        ), f"'{key}' must be present in the dictionary returned by 'get_info'."
    print("Check for 'get_info' method return value and required keys: passed")

    # Override the _status attribute to log status changes
    class ModelWithLogging(ModelClass):
        @property
        def _status(self):
            return self.__dict__.get("_actual_status", TrainingStatus.NOT_STARTED)

        @_status.setter
        def _status(self, value):
            validator.log_status_change(value)
            self.__dict__["_actual_status"] = value

    model = ModelWithLogging(**model_args)

    # Fit the model with dummy data
    model.fit(dataset)
    print("Check for model fitting: passed")

    # Check for required attributes after fitting
    for attr in required_attributes:
        assert hasattr(
            model, attr
        ), f"Model class must have a '{attr}' attribute after fitting."
        print(f"Check for attribute {attr}: passed")

    # Check for optional attributes or alternative methods
    for attr, method in zip(optional_attributes, alternative_methods):
        if not hasattr(model, attr):
            assert hasattr(model, method) and callable(
                getattr(model, method)
            ), f"Model class must have a '{attr}' attribute or a '{method}' method after fitting."
            print(f"Check for method {method} or attribute {attr}: passed")

    # Validate the shape of theta
    if hasattr(model, "theta"):
        theta = model.theta
    else:
        theta = model.get_theta()

    n_topics = model.n_topics  # Number of topics used in fit method
    corpus_length = len(dataset.get_corpus())
    expected_shape = (corpus_length, n_topics)
    assert (
        theta.shape == expected_shape
    ), f"The shape of 'theta' must be {expected_shape}, but got {theta.shape}."
    print("Check for theta shape: passed")

    # Validate the sum of theta values
    theta_sum = np.sum(theta, axis=1)
    assert np.allclose(
        theta_sum, np.ones(corpus_length)
    ), "The sum of 'theta' values along axis 1 must be all ones."
    print("Check for sum of theta values: passed")

    # Validate the shape of beta
    if hasattr(model, "beta"):
        beta = model.beta
    else:
        beta = model.get_beta()

    expected_shape = model.n_topics  # Number of topics used in fit method
    assert (
        beta.shape[1] == expected_shape
    ), f"The second dim (column dimensino) of 'beta' must be {expected_shape}, but got {beta.shape[1]}."
    print("Check for beta shape: passed")

    # Check method signatures (optional)
    sig = inspect.signature(ModelClass.fit)
    assert "dataset" in sig.parameters, "'fit' method must have a 'dataset' parameter."
    print("Check for fit method signature: passed")

    # Validate the status changes
    assert (
        validator.validate_status_changes()
    ), "Model did not transition status correctly."
    print("Check for status transitions: passed")

    print("Success! Model class validation passed.")
    return True
