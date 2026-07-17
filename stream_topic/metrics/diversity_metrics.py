import nltk
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseMetric
from ._helper_funcs import (
    cos_sim_pw,
    embed_stopwords,
)
from .constants import (
    EMBEDDING_PATH,
    NLTK_STOPWORD_LANGUAGE,
    PARAPHRASE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)
from .TopwordEmbeddings import TopwordEmbeddings


NLTK_STOPWORDS = stopwords.words(NLTK_STOPWORD_LANGUAGE)
STOPWORDS = list(
    set(list(NLTK_STOPWORDS) + list(ENGLISH_STOP_WORDS))
)


class TopicDiversity(BaseMetric):
    """
    Topic Diversity: proportion of unique words across all topic top-word lists.

    TD = |union(W_k)| / sum(|W_k|)

    TD = 1.0 means no word repetition across topics; low TD signals redundant topics.

    Parameters
    ----------
    n_words : int, optional
        Number of top words per topic. Defaults to 10.

    Examples
    --------
    >>> from stream_topic.metrics import TopicDiversity
    >>> td = TopicDiversity()
    >>> score = td.score(topics)
    """

    def __init__(self, n_words=10):
        self.n_words = n_words

    def get_info(self):
        return {
            "metric_name": "Topic Diversity",
            "n_words": self.n_words,
            "metric_range": "0 to 1, higher is better",
            "description": "Proportion of unique words across all topic top-word lists.",
        }

    def score(self, topic_words):
        """
        Compute topic diversity.

        Parameters
        ----------
        topic_words : list of list of str
            Top words per topic.

        Returns
        -------
        float
            Topic diversity score in [0, 1].
        """
        all_words = []
        for topic in topic_words:
            all_words.extend(topic[: self.n_words])
        if len(all_words) == 0:
            return 0.0
        return float(np.around(len(set(all_words)) / len(all_words), 5))

    def score_per_topic(self, topic_words):
        """
        Compute per-topic unique word ratio (words unique to this topic / n_words).

        Parameters
        ----------
        topic_words : list of list of str
            Top words per topic.

        Returns
        -------
        dict
            Topic string -> uniqueness ratio.
        """
        # Collect all words from all other topics for each topic
        results = {}
        all_topic_words = [set(t[: self.n_words]) for t in topic_words]
        for k, words in enumerate(all_topic_words):
            other_words = set()
            for j, ow in enumerate(all_topic_words):
                if j != k:
                    other_words |= ow
            unique_to_topic = words - other_words
            ratio = len(unique_to_topic) / max(len(words), 1)
            half = topic_words[k][: len(topic_words[k]) // 2]
            results[", ".join(half)] = float(np.around(ratio, 5))
        return results


class Embedding_Topic_Diversity(BaseMetric):
    """
    A metric class to calculate the diversity of topics based on word embeddings. It computes
    the mean cosine similarity of the mean vectors of the top words of all topics, providing
    a measure of how diverse the topics are in the embedding space.

    Attributes
    ----------
        n_words : int
            The number of top words to consider for each topic.
        metric_embedder : SentenceTransformer
            The SentenceTransformer model to use for embedding.

    Examples
    --------
    >>> topics = [
    ...     ["apple", "banana", "cherry", "date", "fig"],
    ...     ["dog", "cat", "rabbit", "hamster", "gerbil"]
    ... ]
    >>> beta = np.random.rand(2, 5)
    >>> diversity_metric = Embedding_Topic_Diversity()
    >>> info = diversity_metric.get_info()
    >>> print("Metric Info:", info)
    >>> scores = diversity_metric.score(topics, beta)
    >>> print("Diversity score:", scores)
    """

    def __init__(
        self,
        n_words=10,
        metric_embedder=SentenceTransformer(PARAPHRASE_TRANSFORMER_MODEL),
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the Embedding_Topic_Diversity object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename of the embeddings to load. Defaults to None.
        emb_path : str, optional
            The path to the embeddings file. Defaults to "/embeddings".
        """

        self.topword_embeddings = TopwordEmbeddings(
            word_embedding_model=metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        self.n_words = n_words

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, embedding model name,
            metric range, and metric description.
        """
        info = {
            "metric_name": "Embedding Topic Diversity",
            "n_words": self.n_words,
            "embedding_model_name": self.topword_embeddings.word_embedding_model,
            "metric_range": "0 to 1, smaller is better",
            "description": "The diversity metric measures the mean cosine similarity of the mean vectors of the top words of all topics.",
        }
        return info

    def score(self, topics, beta):
        """
        Calculates the overall diversity score for the given model output.

        This method computes the diversity of the topics by averaging the cosine similarity
        of the mean vectors of the top words of each topic. A lower score indicates higher diversity.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics and a topic-word matrix.
        beta : numpy.ndarray
            The topic-word distribution matrix.

        Returns
        -------
        float
            The overall diversity score for all topics.
        """
        topics_tw = topics  # size: (n_topics, n_words)
        n_topics = len(topics_tw)
        
        # Beta should be (n_topics, vocab_size), extract weights for top words
        if beta.shape[0] != n_topics:
            beta = beta.T  # Transpose if needed
        
        # Replace NaN with 0
        beta = np.nan_to_num(beta, nan=0.0)

        # Weights of the top n_words per topic, in descending order (aligned with
        # topics returned by get_topics, which are sorted by beta descending).
        topic_weights = -np.sort(-beta, axis=1)[:, : self.n_words]

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        topwords_embedded = self.topword_embeddings.embed_topwords(
            topics_tw, n_topwords_to_use=self.n_words
        )

        weighted_vecs = (
            topic_weights[:, :, None] * topwords_embedded
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        return float(cos_sim_pw(topic_means))

    def score_per_topic(self, topics, beta):
        """
        Calculates diversity scores for each topic individually based on embedding similarities.

        This method computes the diversity of each topic by calculating the cosine similarity
        of its mean vector with the mean vectors of other topics.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics and a topic-word matrix.
        beta : numpy.ndarray
            The topic-word distribution matrix.

        Returns
        -------
        numpy.ndarray
            An array of diversity scores for each topic.
        """
        topics_tw = topics  # size: (n_topics, n_words)
        n_topics = len(topics_tw)
        
        # Beta should be (n_topics, vocab_size), extract weights for top words
        if beta.shape[0] != n_topics:
            beta = beta.T  # Transpose if needed
        
        # Replace NaN with 0
        beta = np.nan_to_num(beta, nan=0.0)

        # Weights of the top n_words per topic, in descending order (aligned with
        # topics returned by get_topics, which are sorted by beta descending).
        topic_weights = -np.sort(-beta, axis=1)[:, : self.n_words]

        topic_weights = topic_weights / np.nansum(
            topic_weights, axis=1, keepdims=True
        ).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        topwords_embedded = self.topword_embeddings.embed_topwords(
            topics_tw, n_topwords_to_use=self.n_words
        )

        weighted_vecs = (
            topic_weights[:, :, None] * topwords_embedded
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        sim = cosine_similarity(
            topic_means
        )  # calculate the pairwise cosine similarity of the topic means
        sim_mean = (np.sum(sim, axis=1) - 1) / (
            len(sim) - 1
        )  # average the similarity of each topic's mean to the mean of every other topic

        results = {}
        for k in range(len(topics)):
            half_topic_words = topics_tw[k][
                : len(topics_tw[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(np.array(sim_mean)[k], 5)

        return results


class Expressivity(BaseMetric):
    """
    A metric class to calculate the expressivity of topics by measuring the distance between
    the mean vector of the top words in a topic and the mean vector of the embeddings of
    the stop words. Lower distances suggest higher expressivity, indicating that the topic's
    top words are distinct from common stopwords.

    Attributes
    ----------
    n_words : int
        The number of top words to consider for each topic.
    metric_embedder : SentenceTransformer
        The SentenceTransformer model to use for embedding.
    stopwords : list
        A list of stopwords to use for the expressivity calculation.

    Examples
    --------
    >>> from sentence_transformers import SentenceTransformer
    >>> topics = [
    ...     ["apple", "banana", "cherry", "date", "fig"],
    ...     ["dog", "cat", "rabbit", "hamster", "gerbil"]
    ... ]
    >>> expressivity_metric = Expressivity(
    ...     n_words=5,
    ...     stopwords=["the", "is", "at", "which", "on"],
    ...     metric_embedder=SentenceTransformer('paraphrase-MiniLM-L6-v2')
    ... )
    >>> info = expressivity_metric.get_info()
    >>> print("Metric Info:", info)
    >>> scores = expressivity_metric.score(topics, beta)
    >>> print("Expressivity scores:", scores)
    """

    def __init__(
        self,
        n_words=10,
        stopwords=list,
        metric_embedder=SentenceTransformer(PARAPHRASE_TRANSFORMER_MODEL),
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the Expressivity object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        stopwords : list, optional
            A list of stopwords to use for the expressivity calculation. Defaults includes list of NLTK, Gensim, and Scikit-learn stopwords.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename of the embeddings to load. Defaults to None.
        emb_path : str, optional
            The path to the embeddings file. Defaults to "/embeddings".
        """
        self.stopwords = stopwords
        if stopwords is None:
            self.stopwords = STOPWORDS

        self.topword_embeddings = TopwordEmbeddings(
            word_embedding_model=metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        self.n_words = n_words

        self.stopword_emb = embed_stopwords(
            self.stopwords, metric_embedder
        )  # embed all the stopwords size: (n_stopwords, emb_dim)
        self.stopword_mean = np.mean(
            np.array(self.stopword_emb), axis=0
        )  # mean of stopword embeddings

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, embedding model name,
            metric range, and metric description.
        """
        info = {
            "metric_name": "Expressivity",
            "n_words": self.n_words,
            "embedding_model_name": self.topword_embeddings.word_embedding_model,
            "metric_range": "0 to 1, smaller is better",
            "description": "The expressivity metric measures the distance between the mean vector of the top words in a topic and the mean vector of the embeddings of the stop words.",
        }
        return info

    def score(self, topics, beta, new_embeddings=True):
        """
        Calculates the overall expressivity score for the given model output.

        This method computes the expressivity of the topics by averaging the cosine similarity
        between the mean vectors of the top words of each topic and the mean vector of
        the stopwords. A lower score indicates higher expressivity.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics and a topic-word matrix.
        beta : numpy.ndarray
            The topic-word distribution matrix.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall expressivity score for all topics.
        """
        if new_embeddings:
            self.embeddings = None
        return float(
            np.mean(list(self.score_per_topic(topics, beta, new_embeddings).values()))
        )

    def score_per_topic(self, topics, beta, new_embeddings=True):
        """
        Calculates expressivity scores for each topic individually based on embedding distances.

        This method computes the expressivity of each topic by calculating the cosine similarity
        of its mean vector with the mean vector of the stopwords.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics and a topic-word matrix.
        beta : numpy.ndarray
            The topic-word distribution matrix.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        numpy.ndarray
            An array of expressivity scores for each topic.
        """
        if new_embeddings:
            self.embeddings = None

        topics_tw = topics  # size: (n_topics, n_words)
        n_topics = len(topics_tw)
        
        # Beta should be (n_topics, vocab_size), extract weights for top words
        if beta.shape[0] != n_topics:
            beta = beta.T  # Transpose if needed
        
        # Replace NaN with 0
        beta = np.nan_to_num(beta, nan=0.0)

        # Weights of the top n_words per topic, in descending order (aligned with
        # topics returned by get_topics, which are sorted by beta descending).
        topic_weights = -np.sort(-beta, axis=1)[:, : self.n_words]

        topic_weights = topic_weights / np.nansum(
            topic_weights, axis=1, keepdims=True
        ).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        emb_tw = self.topword_embeddings.embed_topwords(
            topics_tw, n_topwords_to_use=self.n_words
        )

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        if np.isnan(topic_means.sum()) != 0:
            # raise ValueError("There are some nans in the topic means")
            print("There are some nans in the topic means")

        topword_sims = []
        valid_topic_means = []

        for mean in topic_means:
            if not np.isnan(
                mean
            ).any():  # Check if there are no NaNs in the current mean
                # Append non-NaN mean to the valid list
                valid_topic_means.append(mean)

        # Compute cosine similarity for valid topic means only
        for mean in valid_topic_means:
            topword_sims.append(
                cosine_similarity(
                    mean.reshape(1, -1), self.stopword_mean.reshape(1, -1)
                )[0, 0]
            )

        results = {}
        for k in range(
            len(valid_topic_means)
        ):  # Adjust range to the length of valid_topic_means
            half_topic_words = topics_tw[k][
                : len(topics_tw[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(
                np.array(topword_sims)[k], 5
            )

        return results
