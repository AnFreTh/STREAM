from inspect import signature
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .dataset import TMDataset

__all__ = ["benchmarking"]


def benchmarking(
    models: List[Any],
    num_topics: int,
    metrics: List[Callable[..., Any]],
    model_args: Optional[List[Dict[str, Any]]] = None,
    metric_args: List[Dict[str, Any]] = None,
    dataset: TMDataset = None,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark multiple models against specified metrics. Initialization parameters for models
    are handled dynamically to accommodate different model requirements.

    Parameters:
    - models : List of model instances to benchmark.
    - num_topics : Integer specifying the number of topics for models that require it.
    - metrics : List of metric functions to evaluate the models.
    - model_args : Optional list of dictionaries with initialization parameters for each model.
    - metric_args : List of dictionaries containing arguments for each metric function.
    - dataset : The dataset to train the models on.
    - embedding_model_name : Default embedding model name, used if applicable.

    Returns:
    - A dictionary mapping model names to another dictionary of metric names and their corresponding scores.
    """
    if model_args is None:
        model_args = [{}] * len(models)
    if metric_args is None:
        metric_args = [{}] * len(metrics)

    results = {}
    metric_instances = []
    for metric, m_args in zip(metrics, metric_args):
        try:
            # Ensure n_topics is included in metric arguments if the metric requires it
            metric_params = signature(metric).parameters
            if "n_topics" in metric_params:
                m_args.setdefault("n_topics", num_topics)
            if "dataset" in metric_params:
                m_args.setdefault("dataset", dataset)
            if "measure" in metric_params:
                m_args.setdefault("measure", "u_mass")
            # Initialize metric with specific arguments
            filtered_m_args = {k: v for k,
                               v in m_args.items() if k in metric_params}
            metric_instances.append(metric(**filtered_m_args))

        except Exception as e:
            print(f"Error processing metric {metric.__name__}: {e}")

    for model, args in zip(models, model_args):
        results[str(model).split(".")[-2]] = {}

        # Ensure num_topics is included in the arguments
        args.setdefault("num_topics", num_topics)

        # Set default embedding model name if necessary and applicable
        if (
            "embedding_model_name" in signature(model).parameters
            and "embedding_model_name" not in args
        ):
            args["embedding_model_name"] = embedding_model_name
        elif "bert_model" in signature(model).parameters and "bert_model" not in args:
            args["bert_model"] = embedding_model_name

        if str(model).split(".")[-2] == "som":
            m = int(np.sqrt(num_topics))
            n = int(np.sqrt(num_topics))
            args.setdefault("m", m)
            args.setdefault("n", n)
            print(m, n)

        # Initialize model with specific arguments, using only keys that exist in model's constructor
        model_init_params = signature(model).parameters
        filtered_args = {k: v for k,
                         v in args.items() if k in model_init_params}
        initialized_model = model(**filtered_args)
        output = initialized_model.train_model(dataset)

        for metric_instance in metric_instances:
            score = metric_instance.score(output)
            results[str(model).split(".")[-2]][
                str(metric_instance).split(".")[-1].split(" ")[0]
            ] = score

    return results
