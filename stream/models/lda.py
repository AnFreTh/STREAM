from loguru import logger
from datetime import datetime
import gensim.corpora as corpora
from nltk.tokenize import word_tokenize
import numpy as np
from ..utils.dataset import TMDataset
from .base import BaseModel, TrainingStatus
from gensim.models import ldamodel
import pandas as pd

MODEL_NAME = "LDA"
time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class LDA(BaseModel):

    def __init__(self, id2word=None, id_corpus=None, random_state=None):
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
            raise ValueError(f"Column 'tokens' does not exist in the dataframe.")

        # Define a helper function to check if an entry is tokenized
        def is_tokenized(entry):
            return isinstance(entry, list) and all(
                isinstance(token, str) for token in entry
            )

        # Tokenize entries that are not tokenized
        dataset.dataframe["tokens"] = dataset.dataframe["tokens"].apply(
            lambda entry: word_tokenize(entry) if not is_tokenized(entry) else entry
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

        self.n_topics = n_topics

        try:
            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self._prepare_documents(dataset)
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

        self.soft_labels = self.get_topic_document_matrix()
        self.labels = np.array(np.argmax(self.soft_labels, axis=1))

        self.topic_dict = self._get_topic_word_dict()

    def predict(self, dataset):
        pass

    def get_topics(self, n_words: int = 10):
        """
        Get the top words for each topic.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to retrieve for each topic (default is 10).

        Returns
        -------
        topics : list of list of str
            List of topics, where each topic is a list of top words.

        Raises
        ------
        RuntimeError
            If the model has not been trained yet or failed.
        """
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")

        topics = []
        for i in range(self.n_topics):
            topic_words_list = []
            for word_tuple in self.model.get_topic_terms(i, n_words):
                topic_words_list.append(self.id2word[word_tuple[0]])
            topics.append(topic_words_list)
        return topics

    def get_topic_document_matrix(self):
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

    def get_topic_word_matrix(self):
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
        return topic_word_matrix

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
