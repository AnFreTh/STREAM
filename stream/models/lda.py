from datetime import datetime

import gensim.corpora as corpora
import numpy as np
import pandas as pd
from gensim.models import ldamodel
from loguru import logger
from nltk.tokenize import word_tokenize

from ..commons.check_steps import check_dataset_steps
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

MODEL_NAME = "LDA"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class LDA(BaseModel):

    def __init__(self, id2word=None, id_corpus=None, random_state=None, **kwargs):
        """
        Initialize the LDA model.

        Parameters
        ----------
        id2word : Dictionary or None, optional
            A Gensim dictionary mapping word ids to words.
        id_corpus : List of lists or None, optional
            The corpus represented as a list of lists of (word_id, word_frequency) tuples.
        random_state : int or None, optional
            Seed for random number generation.
        """
        super().__init__(use_pretrained_embeddings=True, **kwargs)
        self.save_hyperparameters(ignore=["id2word", "id_corpus"])

        self._status = TrainingStatus.NOT_STARTED
        self.n_topics = None
        self.id2word = id2word
        self.id_corpus = id_corpus
        self.random_state = random_state

    def get_info(self):
        """
        Get information about the LDA model.

        Returns
        -------
        info : dict
            Dictionary containing model information.
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
        }
        return info

    def _assert_and_tokenize(self, dataset):
        """
        Ensure that the 'tokens' column exists and tokenize the entries if needed.

        Parameters
        ----------
        dataset : TMDataset
            The dataset containing the 'tokens' column.

        Raises
        ------
        ValueError
            If the 'tokens' column does not exist in the dataframe.
        """
        # Ensure the 'tokens' column exists
        if "tokens" not in dataset.dataframe.columns:
            raise ValueError(
                f"Column 'tokens' does not exist in the dataframe.")

        # Define a helper function to check if an entry is tokenized
        def is_tokenized(entry):
            return isinstance(entry, list) and all(
                isinstance(token, str) for token in entry
            )

        # Tokenize entries that are not tokenized
        dataset.dataframe["tokens"] = dataset.dataframe["tokens"].apply(
            lambda entry: word_tokenize(
                entry) if not is_tokenized(entry) else entry
        )

        return dataset

    def _prepare_documents(self, dataset):
        """
        Prepare the documents for LDA training.

        Parameters
        ----------
        dataset : TMDataset
            The dataset containing the documents to be prepared.
        """

        logger.info(f"--- Preparing the documents for {MODEL_NAME} ---")

        dataset = self._assert_and_tokenize(dataset)

        if self.id2word is None:
            self.id2word = corpora.Dictionary(dataset.dataframe["tokens"])

        if self.id_corpus is None:
            self.id_corpus = [
                self.id2word.doc2bow(document)
                for document in dataset.dataframe["tokens"]
            ]

    def fit(self, dataset: TMDataset = None, n_topics: int = 20, **lda_params):
        """
        Fit the LDA model to the dataset.

        Parameters
        ----------
        dataset : TMDataset, optional
            The dataset to fit the model to. Must be an instance of TMDataset.
        n_topics : int, optional
            The number of topics to extract (default is 20).
        **lda_params : dict, optional
            Additional parameters to pass to the Gensim LdaModel.

        Raises
        ------
        AssertionError
            If the dataset is not an instance of TMDataset.
        RuntimeError
            If there is an error during training.
        """
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)
        self.dataset = dataset

        self.n_topics = n_topics

        try:
            self._status = TrainingStatus.INITIALIZED
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            if not self.id_corpus and not self.id2word:
                self._prepare_documents(dataset)
            lda_params = {
                key: value
                for key, value in {**self.hparams, **lda_params}.items()
                if key != "n_topics"
            }
            self.model = ldamodel.LdaModel(
                self.id_corpus, num_topics=n_topics, **lda_params
            )
        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        self.theta = self.get_theta()
        self.labels = np.array(np.argmax(self.theta, axis=1))

        self.topic_dict = self._get_topic_word_dict()

    def optimize_and_fit(
        self,
        dataset,
        metric,
        min_topics=2,
        max_topics=20,
        n_trials=100,
    ):
        """
        A new method in the child class that calls the parent class's optimize_hyperparameters method.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        min_topics : int, optional
            Minimum number of topics to evaluate, by default 2.
        max_topics : int, optional
            Maximum number of topics to evaluate, by default 20.
        criterion : str, optional
            Criterion to use for optimization ('aic', 'bic', or 'custom'), by default 'aic'.
        n_trials : int, optional
            Number of trials for optimization, by default 100.
        custom_metric : object, optional
            Custom metric object with a `score` method for evaluation, by default None.

        Returns
        -------
        dict
            Dictionary containing the best parameters and the optimal number of topics.
        """
        best_params = super().optimize_hyperparameters(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion="custom",
            n_trials=n_trials,
            custom_metric=metric,
        )

        return best_params

    def predict(self, dataset):
        pass

    def get_theta(self):
        """
        Get the topic distribution for each document.

        Returns
        -------
        topic_document_matrix : pd.DataFrame
            DataFrame where each row corresponds to a document and each column to a topic,
            with the values representing the topic probabilities for each document.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or failed.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        topic_document_matrix = []
        for doc_bow in self.id_corpus:
            topic_distribution = self.model.get_document_topics(doc_bow)
            topic_document_matrix.append(topic_distribution)
        return self._convert_to_dataframe(topic_document_matrix, self.n_topics)

    def _convert_to_dataframe(self, topic_distributions, num_topics):
        """
        Convert topic distributions to a DataFrame.

        Parameters
        ----------
        topic_distributions : list of list of tuples
            List of topic distributions, where each distribution is a list of (topic_id, probability) tuples.
        num_topics : int
            The number of topics.

        Returns
        -------
        df : pd.DataFrame
            DataFrame where each row corresponds to a document and each column to a topic,
            with the values representing the topic probabilities for each document.
        """
        # Initialize an empty list to store the document-topic distributions
        data = []

        # Iterate through each document's topic distribution
        for doc_distribution in topic_distributions:
            # Create a dictionary with the topic probabilities
            doc_data = {
                f"topic_{topic_id}": prob for topic_id, prob in doc_distribution
            }
            # Add missing topics with probability 0
            for topic_id in range(num_topics):
                if f"topic_{topic_id}" not in doc_data:
                    doc_data[f"topic_{topic_id}"] = 0.0
            data.append(doc_data)

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(data)

        return df

    def get_beta(self):
        """
        Get the word distribution for each topic.

        Returns
        -------
        topic_word_matrix : list of list of tuples
            List of topics, where each topic is a list of (word_id, probability) tuples.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or failed.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        topic_word_matrix = []
        for topic_id in range(self.n_topics):
            word_distribution = self.model.get_topic_terms(
                topic_id, topn=len(self.id2word)
            )
            topic_word_matrix.append(word_distribution)

        n = max(max(t[0] for t in topic) for topic in topic_word_matrix) + 1
        num_topics = len(topic_word_matrix)
        num_words_per_topic = len(topic_word_matrix[0])

        # Initialize the matrix with zeros
        beta_matrix = np.zeros((n, num_topics))

        # Fill the matrix with values from the beta list
        for topic_idx, topic in enumerate(topic_word_matrix):
            for word_index, probability in topic:
                beta_matrix[word_index, topic_idx] = probability

        self.beta = beta_matrix
        return self.beta

    def _get_topic_word_dict(self, num_words=100):
        """
        Get the topic-word dictionary for the LDA model.

        Parameters
        ----------
        num_words : int, optional
            The number of top words to include for each topic (default is 100).

        Returns
        -------
        topic_word_dict : dict
            Dictionary where keys are topic ids and values are lists of tuples (word, probability).
        """
        topic_word_dict = {}

        for topic_id in range(self.model.num_topics):
            topic_terms = self.model.get_topic_terms(topic_id, topn=num_words)
            topic_word_dict[topic_id] = [
                (self.id2word[term_id], prob) for term_id, prob in topic_terms
            ]

        return topic_word_dict

    def suggest_hyperparameters(self, trial):
        # Suggest LDA-specific hyperparameters (e.g., alpha, beta)
        self.hparams["alpha"] = trial.suggest_float("alpha", 0.01, 1.0)
        self.hparams["eta"] = trial.suggest_float("eta", 0.01, 1.0)
