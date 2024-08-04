import json
import os


def check_dataset_steps(dataset, logger, model_type, preprocessing_steps=None):
    """
    Check if the dataset has been preprocessed according to the required steps for the model.

    Parameters
    ----------
    dataset : TMDataset
        The dataset object that has been preprocessed.
    model_type : str
        The model type to check the preprocessing steps for.
    preprocessing_steps : dict, optional
        The preprocessing steps to check against. If None, the default steps are loaded.

    Returns
    -------
    bool
        True if the dataset has been preprocessed according to the required steps, False otherwise.
    """
    if preprocessing_steps is None:
        preprocessing_steps = load_model_preprocessing_steps(model_type)

    missing_steps = []

    # Check if the dataset has been preprocessed according to the required steps
    for step, value in preprocessing_steps.items():
        if isinstance(value, bool):
            if (
                step not in dataset.preprocessing_steps
                or preprocessing_steps[step] != dataset.preprocessing_steps[step]
            ):
                missing_steps.append(step)

    if missing_steps:
        logger.warning(
            "The following preprocessing steps are recommended for the {} model:\n{}\nInclude them by running: dataset.preprocess(model_type='{}')".format(
                model_type, ",\n".join(missing_steps), model_type
            )
        )


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
            current_dir, "..", "preprocessor", "default_preprocessing_steps.json"
        )
        filepath = os.path.abspath(filepath)

    with open(filepath, "r") as file:
        all_steps = json.load(file)
    return all_steps.get(model_type, {})
