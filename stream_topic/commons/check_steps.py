import json
import os

from .load_steps import load_model_preprocessing_steps


def check_dataset_steps(dataset, logger, model_type, preprocessing_steps=None):
    """
    Check if the dataset has been preprocessed according to the required steps for the model.

    Parameters
    ----------
    dataset : TMDataset
        The dataset object that has been preprocessed.
    logger : logger object from the script
        The logger object to use for logging messages.
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
