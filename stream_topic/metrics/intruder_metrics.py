import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from .base import BaseMetric
from .constants import (
    EMBEDDING_PATH,
    PARAPHRASE_TRANSFORMER_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
)
from .TopwordEmbeddings import TopwordEmbeddings


class ISIM(BaseMetric):
    """
    A metric class to calculate the Intruder Similarity Metric (ISIM) for topics. This metric evaluates
    the distinctiveness of topics by measuring the average cosine similarity between the top words of
    a topic and randomly chosen intruder words from other topics. Lower scores suggest higher topic
    distinctiveness.

    Attributes
    ----------
    n_words : int
        The number of top words to consider for each topic.
    metric_embedder : SentenceTransformer
        The SentenceTransformer model to use for embedding.
    n_intruders : int
        The number of intruder words to draw for each topic.

    Examples
    --------
    >>> from stream_topic.metrics import SentenceTransformer
    >>> topics = [
    ...     ["apple", "banana", "cherry", "date", "fig"],
    ...     ["dog", "cat", "rabbit", "hamster", "gerbil"]
    ... ]
    >>> isim = ISIM(n_words=5, n_intruders=1, metric_embedder=SentenceTransformer('paraphrase-MiniLM-L6-v2'))
    >>> info = isim.get_info()
    >>> print("Metric Info:", info)
    >>> scores = isim.score(topics)
    >>> print("ISIM scores:", scores)
    """

    def __init__(
        self,
        n_words=10,
        n_intruders=1,
        metric_embedder=SentenceTransformer(PARAPHRASE_TRANSFORMER_MODEL),
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the ISIM object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        n_intruders : int, optional
            The number of intruder words to draw for each topic. Defaults to 1.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename for the embedding model. Defaults to None.
        emb_path : str, optional
            The path to the embedding model. Defaults to EMBEDDING_PATH.
        """

        self.topword_embeddings = TopwordEmbeddings(
            word_embedding_model=metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        self.n_words = n_words
        self.n_intruders = n_intruders

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, number of intruders, embedding model name,
            metric range and metric description.
        """

        info = {
            "metric_name": "Intruder Similarity Metric (ISIM)",
            "n_words": self.n_words,
            "n_intruders": self.n_intruders,
            "embedding_model_name": self.metric_embedder,
            "metric_range": "0 to 1, smaller is better",
            "description": " the average cosine similarity between every word in a topic and an intruder word.",
        }

        return info

    def score_one_intr_per_topic(self, topics, new_embeddings=True):
        """
        Calculates the ISIM score for each topic individually using only one intruder word.

        This method computes the ISIM score for each topic by averaging the cosine similarity
        between one randomly chosen intruder word and the top words of that topic.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        numpy.ndarray
            An array of ISIM scores for each topic with one intruder word.
        """
        emb_tw = self.topword_embeddings.embed_topwords(
            topics, n_topwords_to_use=self.n_words
        )  # embed the top words
        avg_sim_topic_list = (
            []
        )  # iterate over each topic and append the average similarity to the intruder word
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True)  # mask out the current topic
            mask[idx] = False

            other_topics = emb_tw[
                mask
            ]  # embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(
                other_topics.shape[0]
            )  # select random topic index
            intr_word_idx = np.random.randint(
                other_topics.shape[1]
            )  # select random word index

            intr_embedding = other_topics[
                intr_topic_idx, intr_word_idx
            ]  # select random word

            sim = cosine_similarity(
                intr_embedding.reshape(1, -1), topic
            )  # calculate all pairwise similarities of intruder words and top words

            avg_sim_topic_list.append(np.mean(sim))

        return np.array(avg_sim_topic_list)

    def score_one_intr(self, topics, new_embeddings=True):
        """
        Calculates the overall ISIM score for all topics combined using only one intruder word.

        This method computes the overall ISIM score by averaging the ISIM scores obtained
        from each topic using one randomly chosen intruder word.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall ISIM score for all topics with one intruder word.
        """
        if new_embeddings:
            self.embeddings = None
        return np.mean(self.score_one_intr_per_topic(topics, new_embeddings))

    def score_per_topic(self, topics, new_embeddings=True):
        """
        Calculates the ISIM scores for each topic individually using several intruder words.

        This method computes the ISIM score for each topic by averaging the cosine similarity
        between multiple randomly chosen intruder words and the top words of that topic.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        dict
            A dictionary with topics as keys and their corresponding ISIM scores as values.
        """
        if new_embeddings:
            self.embeddings = None
        score_lis = []
        for _ in range(self.n_intruders):  # iterate over the number of intruder words
            score_per_topic = self.score_one_intr_per_topic(
                topics, new_embeddings=False
            )  # calculate the intruder score, but re-use embeddings
            score_lis.append(score_per_topic)  # and append to list

        res = np.vstack(
            score_lis
        ).T  # stack all scores and transpose to get a (n_topics, n_intruder words) matrix

        mean_scores = np.mean(res, axis=1)
        ntopics = len(topics)
        topic_words = topics
        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(np.around(mean_scores[k], 5))

        return results  # return the mean score for each topic

    def score(self, topics, new_embeddings=True):
        """
        Calculates the overall ISIM score for all topics combined using several intruder words.

        This method computes the overall ISIM score by averaging the ISIM scores obtained
        from each topic using multiple randomly chosen intruder words.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall ISIM score for all topics with several intruder words.
        """
        if new_embeddings:
            self.embeddings = None

        return float(np.mean(list(self.score_per_topic(topics).values())))


class INT(BaseMetric):
    """
    A metric class to calculate the Intruder Topic Metric (INT) for topics. This metric assesses the distinctiveness
    of topics by calculating the embedding intruder cosine similarity accuracy. It involves selecting intruder words
    from different topics and then measuring the accuracy by which the top words of a topic are least similar to these
    intruder words. Higher scores suggest better topic distinctiveness.

    Attributes
    ----------
    n_words : int
        The number of top words to consider for each topic.
    metric_embedder : SentenceTransformer
        The SentenceTransformer model to use for embedding.
    n_intruders : int
        The number of intruder words to draw for each topic.

    Examples
    --------
    >>> topics = [
    ...     ["apple", "banana", "cherry", "date", "fig"],
    ...     ["dog", "cat", "rabbit", "hamster", "gerbil"]
    ... ]
    >>> int_metric = INT()
    >>> info = int_metric.get_info()
    >>> print("Metric Info:", info)
    >>> scores = int_metric.score(topics)
    >>> print("INT scores:", scores)
    """

    def __init__(
        self,
        n_words=10,
        n_intruders=1,
        metric_embedder=SentenceTransformer(PARAPHRASE_TRANSFORMER_MODEL),
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the INT object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        n_intruders : int, optional
            The number of intruder words to draw for each topic. Defaults to 1.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename to use for saving embeddings. Defaults to None.
        emb_path : str, optional
            The path to use for saving embeddings. Defaults to "Embeddings/".
        """

        self.topword_embeddings = TopwordEmbeddings(
            word_embedding_model=metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        self.n_words = n_words
        self.n_intruders = n_intruders

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, number of intruders, embedding model name,
            metric range and metric description.
        """
        info = {
            "metric_name": "Intruder Topic Metric (INT)",
            "n_words": self.n_words,
            "n_intruders": self.n_intruders,
            "embedding_model_name": self.metric_embedder,
            "metric_range": "0 to 1, higher is better",
            "description": "The accuracy with which the top words of a topic are least similar to intruder words.",
        }
        return info

    def score_one_intr_per_topic(self, topics, new_embeddings=True):
        """
        Calculates the INT score for each topic individually using only one intruder word.

        This method computes the INT score for each topic by measuring the accuracy with which
        the top words of the topic are least similar to one randomly chosen intruder word.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        numpy.ndarray
            An array of INT scores for each topic with one intruder word.
        """
        if new_embeddings:
            self.embeddings = None
        topics_tw = topics

        emb_tw = self.topword_embeddings.embed_topwords(
            topics_tw, n_topwords_to_use=self.n_words
        )  # embed the top words
        avg_sim_topic_list = []
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True)  # mask out the current topic
            mask[idx] = False

            other_topics = emb_tw[
                mask
            ]  # embeddings of every other topic except the current one

            intr_topic_idx = np.random.randint(
                other_topics.shape[0]
            )  # select random topic index
            intr_word_idx = np.random.randint(
                other_topics.shape[1]
            )  # select random word index

            intr_embedding = other_topics[
                intr_topic_idx, intr_word_idx
            ]  # select random word

            new_words = np.vstack(
                [intr_embedding, topic]
            )  # stack the intruder embedding above the other embeddings to get a matrix with shape ((1+n_topwords), n_embedding_dims)

            sim = cosine_similarity(
                new_words
            )  # calculate all pairwise similarities for matrix of shape ((1+n_topwords, 1+n_topwords))

            least_similar = np.argmin(
                sim[1:], axis=1
            )  # for each word, except the intruder, calculate the index of the least similar word
            intr_acc = np.mean(
                least_similar == 0
            )  # calculate the fraction of words for which the least similar word is the intruder word (at index 0)

            avg_sim_topic_list.append(
                intr_acc
            )  # append intruder accuracy for this sample

        return np.array(avg_sim_topic_list)

    def score_one_intr(self, topics, new_embeddings=True):
        """
        Calculates the overall INT score for all topics combined using only one intruder word.

        This method computes the overall INT score by averaging the INT scores obtained
        from each topic using one randomly chosen intruder word.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall INT score for all topics with one intruder word.
        """
        if new_embeddings:
            self.embeddings = None
        self.embeddings = None

        return np.mean(self.score_one_intr_per_topic(topics))

    def score_per_topic(self, topics, new_embeddings=True):
        """
        Calculates the INT scores for each topic individually using several intruder words.

        This method computes the INT score for each topic by averaging the accuracy scores
        obtained with multiple randomly chosen intruder words.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        numpy.ndarray
            An array of INT scores for each topic with several intruder words.
        """
        if new_embeddings:
            self.embeddings = None

        score_lis = []
        for _ in range(self.n_intruders):
            score_per_topic = self.score_one_intr_per_topic(
                topics, new_embeddings=False
            )
            score_lis.append(score_per_topic)
        self.embeddings = None
        res = np.vstack(score_lis).T

        mean_scores = np.mean(res, axis=1)
        ntopics = len(topics)
        topic_words = topics
        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(np.around(mean_scores[k], 5))

        return results  # return the mean score for each topic

    def score(self, topics, new_embeddings=True):
        """
        Calculates the overall INT score for all topics combined using several intruder words.

        This method computes the overall INT score by averaging the INT scores obtained
        from each topic using multiple randomly chosen intruder words.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall INT score for all topics with several intruder words.
        """
        if new_embeddings:
            self.embeddings = None

        return float(np.mean(list(self.score_per_topic(topics).values())))


class ISH(BaseMetric):
    """
    A metric class to calculate the Intruder Similarity to Mean (ISH) for topics. This metric evaluates
    the distinctiveness of topics by measuring the average cosine similarity between the mean of the
    top words in a topic and several randomly chosen intruder words from other topics. Lower scores
    suggest higher topic distinctiveness.

    Attributes
    ----------
    n_words : int
        The number of top words to consider for each topic.
    metric_embedder : SentenceTransformer
        The SentenceTransformer model to use for embedding.
    n_intruders : int
        The number of intruder words to draw for each topic.

    Examples
    --------
    >>> from sentence_transformers import SentenceTransformer
    >>> topics = [
    ...     ["apple", "banana", "cherry", "date", "fig"],
    ...     ["dog", "cat", "rabbit", "hamster", "gerbil"]
    ... ]
    >>> ish_metric = ISH(n_words=5, n_intruders=1, metric_embedder=SentenceTransformer('paraphrase-MiniLM-L6-v2'))
    >>> info = ish_metric.get_info()
    >>> print("Metric Info:", info)
    >>> scores = ish_metric.score(topics)
    >>> print("ISH scores:", scores)
    """

    def __init__(
        self,
        n_words=10,
        n_intruders=1,
        metric_embedder=SentenceTransformer(PARAPHRASE_TRANSFORMER_MODEL),
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the ISH object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        n_intruders : int, optional
            The number of intruder words to draw for each topic. Defaults to 1.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename to use for saving embeddings. Defaults to None.
        emb_path : str, optional
            The path to use for saving embeddings. Defaults to "Embeddings/".
        """

        self.topword_embeddings = TopwordEmbeddings(
            word_embedding_model=metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        self.n_words = n_words

        self.embeddings = None
        self.n_intruders = n_intruders

    def get_info(self):
        """
        Get information about the metric.

        Returns
        -------
        dict
            Dictionary containing model information including metric name,
            number of top words, number of intruders, embedding model name,
            metric range and metric description.
        """
        info = {
            "metric_name": "Intruder Similarity to Mean (ISH)",
            "n_words": self.n_words,
            "n_intruders": self.n_intruders,
            "embedding_model_name": self.metric_embedder,
            "metric_range": "0 to 1, smaller is better",
            "description": "The average cosine similarity between the mean of the top words in a topic and several randomly chosen intruder words from other topics.",
        }
        return info

    def score(self, topics, new_embeddings=True):
        """
        Calculate the overall ISH score for all topics combined.

        This method computes the overall ISH score by averaging the ISH scores obtained
        from each topic using several randomly chosen intruder words.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        float
            The overall ISH score for all topics with several intruder words.
        """
        if new_embeddings:
            self.embeddings = None

        return float(np.mean(list(self.score_per_topic(topics).values())))

    def score_per_topic(self, topics, new_embeddings=None):
        """
        Calculate the ISH scores for each topic individually using several intruder words.

        This method computes the ISH score for each topic by averaging the cosine similarity
        between the mean of the top words in the topic and several randomly chosen intruder words
        from other topics.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.
        new_embeddings : bool, optional
            Whether to recalculate embeddings. Defaults to True.

        Returns
        -------
        dict
            A dictionary with topics as keys and their corresponding ISH scores as values.
        """
        if new_embeddings:  # for this function, reuse embeddings per default
            self.embeddings = None

        topics_tw = topics

        emb_tw = self.topword_embeddings.embed_topwords(
            topics_tw, n_topwords_to_use=self.n_words
        )

        score_topic_list = []
        for idx, topic in enumerate(emb_tw):
            mask = np.full(emb_tw.shape[0], True)  # mask out the current topic
            mask[idx] = False

            intruder_words_idx_topic = np.random.choice(
                np.arange(len(emb_tw))[mask], size=self.n_intruders
            )  # select self.n_intruders topics to get the intruder words from
            intruder_words = emb_tw[intruder_words_idx_topic]

            intruder_words_idx_word = np.random.choice(
                np.arange(intruder_words.shape[1]), size=1
            )  # select one intruder word from each topic
            intruder_words = intruder_words[:, intruder_words_idx_word, :].squeeze()

            topic_mean = np.mean(topic, axis=0)

            topic_sims = cosine_similarity(
                topic_mean.reshape(1, -1), intruder_words.reshape(1, -1)
            )
            score_topic_list.append(np.mean(topic_sims))

        results = {}
        ntopics = len(topics)
        topic_words = topics
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(
                np.around(np.array(score_topic_list)[k], 5)
            )

        return results  # return the mean score for each topic
