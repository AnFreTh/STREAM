import json
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum

import umap.umap_ as umap
from loguru import logger


class BaseModel(ABC):
    """
    Abstract base class for topic modeling.

    Attributes:
        hparams (dict): Model hyperparameters.
        model_params: Parameters specific to the trained model.
        vocabulary: Vocabulary used in the model.
        document_topic_distribution: Distribution of topics across documents.
        topic_word_distribution: Distribution of words across topics.
        training_data: Data used for training the model.

    Methods:
        get_info():
            Get information about the model.

        fit(X, y=None):
            Train the topic model on the dataset X.

        predict(X):
            Predict topics for new documents X.

        get_topics(n_words=10):
            Retrieve the top words for each topic.

        get_beta():
            Retrieve the topic-word distribution matrix.

        get_theta():
            Retrieve the topic-document distribution matrix.

        save_hyperparameters(ignore=[]):
            Save the hyperparameters while ignoring specified keys.

        load_hyperparameters(path):
            Load the model hyperparameters from a JSON file.

        get_hyperparameters():
            Get the model hyperparameters.

        save_model(path):
            Save the model state and parameters to a file.

        load_model(path):
            Load the model state and parameters from a file.
    """

    def __init__(self, use_pretrained_embeddings=True, **kwargs):
        """
        Initialize BaseModel with hyperparameters.

        Parameters:
            **kwargs: Additional keyword arguments for model initialization.
        """
        self.hparams = kwargs
        self.model_params = None
        self.vocabulary = None
        self.document_topic_distribution = None
        self.topic_word_distribution = None
        self.training_data = None
        self.use_pretrained_embeddings = use_pretrained_embeddings

    @abstractmethod
    def get_info(self):
        """
        Get information about the model.

        """
        pass

    @abstractmethod
    def fit(self, X, y=None):
        """
        Train the topic model on the dataset X.

        Parameters:
            X: Input dataset for training.
            y: Target labels for supervised training (if applicable).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict topics for new documents X.

        Parameters:
            X: Input documents to predict topics for.

        Returns:
            Predicted topics for input documents.
        """
        pass

    def save_hyperparameters(self, ignore=[]):
        """
        Save the hyperparameters while ignoring specified keys.

        Parameters:
            ignore (list, optional): List of keys to ignore while saving hyperparameters. Defaults to [].
        """
        self.hparams = {k: v for k, v in self.hparams.items()
                        if k not in ignore}
        for key, value in self.hparams.items():
            setattr(self, key, value)

    def load_hyperparameters(self, path):
        """
        Load the model hyperparameters from a JSON file.

        Parameters:
            path (str): Path to the JSON file containing hyperparameters.
        """
        if os.path.exists(path):
            with open(path) as file:
                self.hparams = json.load(file)
        else:
            logger.error(f"Hyperparameters file not found at: {path}")
            raise FileNotFoundError(
                f"Hyperparameters file not found at: {path}")

    def get_hyperparameters(self):
        """
        Get the model hyperparameters.

        Returns:
            dict: Dictionary containing the model hyperparameters.
        """
        return self.hparams

    def save_model(self, path):
        """
        Save the model state and parameters to a file.

        Parameters:
            path (str): Path to save the model file.
        """
        model_state = {
            "hyperparameters": self.hparams,
            "model_params": self.model_params,
            "vocabulary": self.vocabulary,
            "document_topic_distribution": self.document_topic_distribution,
            "topic_word_distribution": self.topic_word_distribution,
            "training_data": self.training_data,
        }
        with open(path, "wb") as file:
            pickle.dump(model_state, file)

    def load_model(self, path):
        """
        Load the model state and parameters from a file.

        Parameters:
            path (str): Path to the saved model file.
        """
        if os.path.exists(path):
            with open(path, "rb") as file:
                model_state = pickle.load(file)
                self.hparams = model_state["hyperparameters"]
                self.model_params = model_state["model_params"]
                self.vocabulary = model_state["vocabulary"]
                self.document_topic_distribution = model_state[
                    "document_topic_distribution"
                ]
                self.topic_word_distribution = model_state["topic_word_distribution"]
                self.training_data = model_state["training_data"]
        else:
            logger.error(f"Model file not found at: {path}")
            raise FileNotFoundError(f"Model file not found at: {path}")

    def dim_reduction(self, logger):
        """
        Reduces the dimensionality of embeddings using UMAP.

        Raises
        ------
        ValueError
            If an error occurs during dimensionality reduction.
        """
        assert hasattr(
            self, "embeddings"
        ), "Model has no embeddings to reduce dimensions."
        assert hasattr(
            self, "umap_args"), "Model has no UMAP arguments specified."
        try:
            logger.info("--- Reducing dimensions ---")
            self.reducer = umap.UMAP(**self.umap_args)
            reduced_embeddings = self.reducer.fit_transform(self.embeddings)
        except Exception as e:
            raise RuntimeError(
                f"Error in dimensionality reduction: {e}") from e

        return reduced_embeddings

    def prepare_embeddings(self, dataset, logger):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be used for clustering.
        """

        if dataset.has_embeddings(self.embedding_model_name):
            logger.info(
                f"--- Loading precomputed {self.embedding_model_name} embeddings ---"
            )
            embeddings = dataset.get_embeddings(
                self.embedding_model_name,
                self.embeddings_path,
                self.embeddings_file_path,
            )

        else:
            logger.info(
                f"--- Creating {self.embedding_model_name} document embeddings ---"
            )
            embeddings = self.encode_documents(
                dataset.texts, encoder_model=self.embedding_model_name, use_average=True
            )
            if self.save_embeddings:
                dataset.save_embeddings(
                    embeddings,
                    self.embedding_model_name,
                    self.embeddings_path,
                    self.embeddings_file_path,
                )
        return dataset.dataframe, embeddings

    def get_topics(self, n_words=10):
        """
        Retrieve the top words for each topic.

        Parameters
        ----------
        n_words : int
            Number of top words to retrieve for each topic.

        Returns
        -------
        list of list of str
            List of topics with each topic represented as a list of top words.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        assert hasattr(self, "topic_dict"), "Model has no topic dictionary."
        return [
            [word for word, _ in self.topic_dict[key][:n_words]]
            for key in self.topic_dict
        ]

    def get_beta(self):
        """
        Retrieve the topic-word distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-word distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        assert hasattr(self, "beta"), "Model has no topic-word distribution."
        return self.beta

    def get_theta(self):
        """
        Retrieve the topic-document distribution matrix.

        Returns
        -------
        numpy.ndarray
            Topic-document distribution matrix.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        assert hasattr(
            self, "theta"), "Model has no topic-document distribution."
        return self.theta

    @abstractmethod
    def fit(self, dataset):
        pass


class TrainingStatus(str, Enum):
    """
    Represents the status of a training process.

    Attributes:
        INITIALIZED (str): The training process has been initialized.
        RUNNING (str): The training process is currently running.
        SUCCEEDED (str): The training process has successfully completed.
        INTERRUPTED (str): The training process was interrupted.
        FAILED (str): The training process has failed.
    """

    NOT_STARTED = "empty"
    INITIALIZED = "initialized"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    INTERRUPTED = "interrupted"
    FAILED = "failed"
