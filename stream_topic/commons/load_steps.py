import json
import os


def load_model_preprocessing_steps(model_type, filepath=None):
    """
    Load the default preprocessing steps from a JSON file.

    Parameters
    ----------
    model_type : str
        The model type to load the preprocessing steps for.
    filepath : str
        The path to the JSON file containing the default preprocessing steps.

    Returns
    -------
    dict
        The default preprocessing steps.
    """
    if filepath is None:
        # Determine the absolute path based on the current file's location
        current_dir = os.path.dirname(__file__)
        filepath = os.path.join(
            current_dir, "..", "preprocessor", "config", "default_preprocessing_steps.json"
        )
        filepath = os.path.abspath(filepath)

    with open(filepath, "r") as file:
        all_steps = json.load(file)
    return all_steps.get(model_type, {})
