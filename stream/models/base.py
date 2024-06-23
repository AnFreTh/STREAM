import json
import os
import pickle
from abc import ABC, abstractmethod
from enum import Enum

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

        get_topic_word_matrix():
            Retrieve the topic-word distribution matrix.

        get_topic_document_matrix():
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

    @abstractmethod
    def get_topics(self, n_words=10):
        """
        Retrieve the top words for each topic.

        Parameters:
            n_words (int): Number of top words to retrieve for each topic.

        Returns:
            dict: A dictionary where keys are topic ids and values are lists of top words.
        """
        pass

    @abstractmethod
    def get_topic_word_matrix(self):
        """
        Retrieve the topic-word distribution matrix.

        Returns:
            np.array: A matrix of size (n_topics, n_words) representing the word distribution for each topic.
        """
        pass

    @abstractmethod
    def get_topic_document_matrix(self):
        """
        Retrieve the topic-document distribution matrix.

        Returns:
            np.array: A matrix of size (n_topics, n_documents) representing the topic distribution for each document.
        """
        pass

    def save_hyperparameters(self, ignore=[]):
        """
        Save the hyperparameters while ignoring specified keys.

        Parameters:
            ignore (list, optional): List of keys to ignore while saving hyperparameters. Defaults to [].
        """
        self.hparams = {k: v for k, v in self.hparams.items() if k not in ignore}
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
            raise FileNotFoundError(f"Hyperparameters file not found at: {path}")

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
