from octis.evaluation_metrics.metrics import AbstractMetric
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ._helper_funcs import (
    cos_sim_pw,
    Embed_topic,
    Embed_corpus,
    Update_corpus_dic_list,
    Embed_stopwords,
)
from sentence_transformers import SentenceTransformer
import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim

gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS
nltk_stopwords = stopwords.words("english")
stopwords = list(
    set(nltk_stopwords + list(gensim_stopwords) + list(ENGLISH_STOP_WORDS))
)


class Embedding_Topic_Diversity(AbstractMetric):
    """
    A metric class to calculate the diversity of topics based on word embeddings. It computes
    the mean cosine similarity of the mean vectors of the top words of all topics, providing
    a measure of how diverse the topics are in the embedding space.

    Attributes:
        n_words (int): The number of top words to consider for each topic.
        corpus_dict (dict): A dictionary mapping each word in the corpus to its embedding.
    """

    def __init__(
        self,
        dataset,
        n_words=10,
        embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        Initializes the Embedding_Topic_Diversity object with a dataset, number of words,
        embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for embedding topic diversity calculation.
            n_words (int, optional): The number of top words to consider for each topic.
                Defaults to 10.
            embedder (SentenceTransformer, optional): The embedding model to use.
                Defaults to SentenceTransformer("paraphrase-MiniLM-L6-v2").
            emb_filename (str, optional): Filename to store embeddings. Defaults to None.
            emb_path (str, optional): Path to store embeddings. Defaults to "Embeddings/".
            expansion_path (str, optional): Path for expansion embeddings. Defaults to "Embeddings/".
            expansion_filename (str, optional): Filename for expansion embeddings. Defaults to None.
            expansion_word_list (list, optional): List of words for expansion. Defaults to None.
        """

        tw_emb = Embed_corpus(
            dataset,
            embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_words = n_words
        self.corpus_dict = tw_emb

    def score(self, model_output):
        """
        Calculates the overall diversity score for the given model output.

        This method computes the diversity of the topics by averaging the cosine similarity
        of the mean vectors of the top words of each topic. A lower score indicates higher diversity.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.

        Returns:
            float: The overall diversity score for all topics.
        """
        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        return float(cos_sim_pw(topic_means))

    def score_per_topic(self, model_output):
        """
        Calculates diversity scores for each topic individually based on embedding similarities.

        This method computes the diversity of each topic by calculating the cosine similarity
        of its mean vector with the mean vectors of other topics.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.

        Returns:
            numpy.ndarray: An array of diversity scores for each topic.
        """
        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words size: (n_topics, n_topwords)

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
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
        for k in range(len(model_output["topics"])):
            half_topic_words = topics_tw[k][
                : len(topics_tw[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(np.array(sim_mean)[k], 5)

        return results


class Expressivity(AbstractMetric):
    """
    A metric class to calculate the expressivity of topics by measuring the distance between
    the mean vector of the top words in a topic and the mean vector of the embeddings of
    the stop words. Lower distances suggest higher expressivity, indicating that the topic's
    top words are distinct from common stopwords.

    Attributes:
        stopword_list (list): A list of stopwords to use for comparison.
        n_words (int): The number of top words to consider for each topic.
        corpus_dict (dict): A dictionary mapping each word in the corpus to its embedding.
        embeddings (numpy.ndarray): The embeddings for the top words of the topics.
        stopword_emb (numpy.ndarray): The embeddings for the stopwords.
        stopword_mean (numpy.ndarray): The mean vector of the embeddings of the stopwords.
    """

    def __init__(
        self,
        dataset,
        stopword_list=stopwords,
        n_words=10,
        embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        Initializes the Expressivity object with a dataset, a list of stopwords, number of
        words, embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for expressivity calculation.
            stopword_list (list, optional): A list of stopwords for comparison. Defaults to a standard list.
            n_words (int, optional): The number of top words to consider for each topic. Defaults to 10.
            embedder (SentenceTransformer, optional): The embedding model to use.
                Defaults to SentenceTransformer("paraphrase-MiniLM-L6-v2").
            emb_filename (str, optional): Filename to store embeddings. Defaults to None.
            emb_path (str, optional): Path to store embeddings. Defaults to "Embeddings/".
            expansion_path (str, optional): Path for expansion embeddings. Defaults to "Embeddings/".
            expansion_filename (str, optional): Filename for expansion embeddings. Defaults to None.
            expansion_word_list (list, optional): List of words for expansion. Defaults to None.
        """

        tw_emb = Embed_corpus(
            dataset,
            embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )

        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )
        self.stopword_list = stopword_list

        self.n_words = n_words
        self.corpus_dict = tw_emb
        self.embeddings = None

        self.stopword_emb = Embed_stopwords(
            stopword_list, embedder
        )  # embed all the stopwords size: (n_stopwords, emb_dim)
        self.stopword_mean = np.mean(
            np.array(self.stopword_emb), axis=0
        )  # mean of stopword embeddings

    def score(self, model_output, new_Embeddings=True):
        """
        Calculates the overall expressivity score for the given model output.

        This method computes the expressivity of the topics by averaging the cosine similarity
        between the mean vectors of the top words of each topic and the mean vector of
        the stopwords. A lower score indicates higher expressivity.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            float: The overall expressivity score for all topics.
        """
        if new_Embeddings:
            self.embeddings = None
        return float(
            np.mean(list(self.score_per_topic(model_output, new_Embeddings).values()))
        )

    def score_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculates expressivity scores for each topic individually based on embedding distances.

        This method computes the expressivity of each topic by calculating the cosine similarity
        of its mean vector with the mean vector of the stopwords.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            numpy.ndarray: An array of expressivity scores for each topic.
        """
        if new_Embeddings:
            self.embeddings = None

        ntopics = len(model_output["topics"])

        topics_tw = model_output["topics"]  # size: (n_topics, voc_size)
        topic_weights = model_output["topic-word-matrix"][
            :, : self.n_words
        ]  # select the weights of the top words

        topic_weights = topic_weights / np.sum(topic_weights, axis=1).reshape(
            -1, 1
        )  # normalize the weights such that they sum up to one

        if self.embeddings is None:
            emb_tw = Embed_topic(
                topics_tw, self.corpus_dict, self.n_words
            )  # embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
                :, : self.n_words, :
            ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = self.embeddings

        weighted_vecs = (
            topic_weights[:, :, None] * emb_tw
        )  # multiply each embedding vector with its corresponding weight
        topic_means = np.sum(
            weighted_vecs, axis=1
        )  # calculate the sum, which yields the weighted average

        if np.isnan(topic_means.sum()) != 0:
            raise ValueError("There are some nans in the topic means")

        topword_sims = []
        # iterate over every topic and append the cosine similarity of the topic's centroid and the stopword mean
        for mean in topic_means:
            topword_sims.append(
                cosine_similarity(
                    mean.reshape(1, -1), self.stopword_mean.reshape(1, -1)
                )[0, 0]
            )

        results = {}
        for k in range(ntopics):
            half_topic_words = topics_tw[k][
                : len(topics_tw[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(
                np.array(topword_sims)[k], 5
            )

        return results
