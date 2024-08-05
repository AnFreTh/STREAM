from abc import ABC, abstractmethod


class BaseMetric(ABC):
    """
    Abstract base class for metrics.

    Attributes
    ----------
    dataset : any
        Dataset used for the metric.
    n_words : int
        Number of words used in the metric evaluation.
    metric_name : str
        Name of the metric.
    metric_value : any
        Value of the metric.

    Methods
    -------
    get_info()
        Get information about the metric.
    evaluate(topics, **kwargs)
        Evaluate the metric using topics.
    score_per_topic(topics, **kwargs)
        Score the metric per topic.
    """

    def __init__(self, dataset, n_words=10, **kwargs):
        """
        Initialize BaseMetric with dataset and number of words.

        Parameters
        ----------
        dataset : any
            Dataset used for the metric.
        n_words : int, optional
            Number of words used in the metric evaluation (default is 10).
        **kwargs : dict, optional
            Additional keyword arguments for metric initialization.
        """
        self.dataset = dataset
        self.n_words = n_words
        self.metric_name = kwargs.get("metric_name", "Unnamed Metric")
        self.metric_value = kwargs.get("metric_value", None)

    @abstractmethod
    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        str
            Information about the metric.
        """
        pass

    @abstractmethod
    def score(self, topics, **kwargs):
        """
        Evaluate the metric using topics.

        Parameters
        ----------
        topics : any
            Topics from the model.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        any
            The result of the evaluation.
        """
        pass

    @abstractmethod
    def score_per_topic(self, topics, **kwargs):
        """
        Score the metric per topic.

        Parameters
        ----------
        topics : any
            Topics from the model.
        **kwargs : dict, optional
            Additional keyword arguments.

        Returns
        -------
        any
            The score per topic.
        """
        pass
