from octis.evaluation_metrics.metrics import AbstractMetric
from sentence_transformers import SentenceTransformer
from ._helper_funcs import (
    Embed_corpus,
    Embed_topic,
    Update_corpus_dic_list,
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class ISIM(AbstractMetric):
    """
    A metric class to calculate the Intruder Similarity Metric (ISIM) for topics. This metric evaluates
    the distinctiveness of topics by measuring the average cosine similarity between the top words of
    a topic and randomly chosen intruder words from other topics. Lower scores suggest higher topic
    distinctiveness.

    Attributes:
        n_intruders (int): The number of intruder words to draw for each topic.
        n_words (int): The number of top words to consider for each topic.
        corpus_dict (dict): A dictionary mapping each word in the corpus to its embedding.
        embeddings (numpy.ndarray): The embeddings for the top words of the topics.
    """

    def __init__(
        self,
        dataset,
        n_intruders=1,
        n_words=10,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        Initializes the ISIM object with a dataset, number of intruders, number of words,
        embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for ISIM calculation.
            n_intruders (int, optional): The number of intruder words to draw for each topic. Defaults to 1.
            n_words (int, optional): The number of top words to consider for each topic. Defaults to 10.
            metric_embedder (SentenceTransformer, optional): The embedding model to use.
                Defaults to SentenceTransformer("paraphrase-MiniLM-L6-v2").
            emb_filename (str, optional): Filename to store embeddings. Defaults to None.
            emb_path (str, optional): Path to store embeddings. Defaults to "Embeddings/".
            expansion_path (str, optional): Path for expansion embeddings. Defaults to "Embeddings/".
            expansion_filename (str, optional): Filename for expansion embeddings. Defaults to None.
            expansion_word_list (list, optional): List of words for expansion. Defaults to None.
        """

        tw_emb = Embed_corpus(
            dataset,
            metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )
        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                metric_embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_intruders = n_intruders
        self.corpus_dict = tw_emb
        self.n_words = n_words
        self.embeddings = None

    def score_one_intr_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculates the ISIM score for each topic individually using only one intruder word.

        This method computes the ISIM score for each topic by averaging the cosine similarity
        between one randomly chosen intruder word and the top words of that topic.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            numpy.ndarray: An array of ISIM scores for each topic with one intruder word.
        """
        if new_Embeddings:  # for this function, reuse embeddings per default
            self.embeddings = None

        topics_tw = model_output["topics"]

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

    def score_one_intr(self, model_output, new_Embeddings=True):
        """
        Calculates the overall ISIM score for all topics combined using only one intruder word.

        This method computes the overall ISIM score by averaging the ISIM scores obtained
        from each topic using one randomly chosen intruder word.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            float: The overall ISIM score for all topics with one intruder word.
        """
        if new_Embeddings:
            self.embeddings = None
        return np.mean(self.score_one_intr_per_topic(model_output, new_Embeddings))

    def score_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculates the ISIM scores for each topic individually using several intruder words.

        This method computes the ISIM score for each topic by averaging the cosine similarity
        between multiple randomly chosen intruder words and the top words of that topic.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            numpy.ndarray: An array of ISIM scores for each topic with several intruder words.
        """
        if new_Embeddings:
            self.embeddings = None
        score_lis = []
        for _ in range(self.n_intruders):  # iterate over the number of intruder words
            score_per_topic = self.score_one_intr_per_topic(
                model_output, new_Embeddings=False
            )  # calculate the intruder score, but re-use embeddings
            score_lis.append(score_per_topic)  # and append to list

        res = np.vstack(
            score_lis
        ).T  # stack all scores and transpose to get a (n_topics, n_intruder words) matrix

        mean_scores = np.mean(res, axis=1)
        ntopics = len(model_output["topics"])
        topic_words = model_output["topics"]
        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(np.around(mean_scores[k], 5))

        return results  # return the mean score for each topic

    def score(self, model_output, new_Embeddings=True):
        """
        Calculates the overall ISIM score for all topics combined using several intruder words.

        This method computes the overall ISIM score by averaging the ISIM scores obtained
        from each topic using multiple randomly chosen intruder words.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            float: The overall ISIM score for all topics with several intruder words.
        """
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined but only with several intruder words
        """

        return float(np.mean(list(self.score_per_topic(model_output).values())))


class INT(AbstractMetric):
    """
    A metric class to calculate the Intruder Topic Metric (INT) for topics. This metric assesses the distinctiveness
    of topics by calculating the embedding intruder cosine similarity accuracy. It involves selecting intruder words
    from different topics and then measuring the accuracy by which the top words of a topic are least similar to these
    intruder words. Higher scores suggest better topic distinctiveness.

    Attributes:
        n_intruders (int): The number of intruder words to draw for each topic.
        n_words (int): The number of top words to consider for each topic.
        corpus_dict (dict): A dictionary mapping each word in the corpus to its embedding.
        embeddings (numpy.ndarray): The embeddings for the top words of the topics.
    """

    def __init__(
        self,
        dataset,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
        n_intruders=1,
        n_words=10,
    ):
        """
        Initializes the INT object with a dataset, number of intruders, number of words,
        embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for INT calculation.
            metric_embedder (SentenceTransformer, optional): The embedding model to use.
                Defaults to SentenceTransformer("paraphrase-MiniLM-L6-v2").
            emb_filename (str, optional): Filename to store embeddings. Defaults to None.
            emb_path (str, optional): Path to store embeddings. Defaults to "Embeddings/".
            expansion_path (str, optional): Path for expansion embeddings. Defaults to "Embeddings/".
            expansion_filename (str, optional): Filename for expansion embeddings. Defaults to None.
            expansion_word_list (list, optional): List of words for expansion. Defaults to None.
            n_intruders (int, optional): The number of intruder words to draw for each topic. Defaults to 1.
            n_words (int, optional): The number of top words to consider for each topic. Defaults to 10.
        """

        tw_emb = Embed_corpus(
            dataset,
            metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )
        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                metric_embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_intruders = n_intruders
        self.corpus_dict = tw_emb
        self.n_words = n_words
        self.embeddings = None

    def score_one_intr_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculates the INT score for each topic individually using only one intruder word.

        This method computes the INT score for each topic by measuring the accuracy with which
        the top words of the topic are least similar to one randomly chosen intruder word.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            numpy.ndarray: An array of INT scores for each topic with one intruder word.
        """
        if new_Embeddings:
            self.embeddings = None
        topics_tw = model_output["topics"]

        if self.embeddings is None:
            emb_tw = Embed_topic(
                topics_tw, self.corpus_dict, self.n_words
            )  # embed the top words
            emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
                :, : self.n_words, :
            ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
            self.embeddings = emb_tw
        else:
            emb_tw = (
                self.embeddings
            )  # create tensor of size (n_topics, n_topwords, n_embedding_dims)

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

    def score_one_intr(self, model_output, new_Embeddings=True):
        """
        Calculates the overall INT score for all topics combined using only one intruder word.

        This method computes the overall INT score by averaging the INT scores obtained
        from each topic using one randomly chosen intruder word.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            float: The overall INT score for all topics with one intruder word.
        """
        if new_Embeddings:
            self.embeddings = None
        self.embeddings = None

        return np.mean(self.score_one_intr_per_topic(model_output))

    def score_per_topic(self, model_output, new_Embeddings=True):
        """
        Calculates the INT scores for each topic individually using several intruder words.

        This method computes the INT score for each topic by averaging the accuracy scores
        obtained with multiple randomly chosen intruder words.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            numpy.ndarray: An array of INT scores for each topic with several intruder words.
        """
        if new_Embeddings:
            self.embeddings = None

        score_lis = []
        for _ in range(self.n_intruders):
            score_per_topic = self.score_one_intr_per_topic(
                model_output, new_Embeddings=False
            )
            score_lis.append(score_per_topic)
        self.embeddings = None
        res = np.vstack(score_lis).T

        mean_scores = np.mean(res, axis=1)
        ntopics = len(model_output["topics"])
        topic_words = model_output["topics"]
        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(np.around(mean_scores[k], 5))

        return results  # return the mean score for each topic

    def score(self, model_output, new_Embeddings=True):
        """
        Calculates the overall INT score for all topics combined using several intruder words.

        This method computes the overall INT score by averaging the INT scores obtained
        from each topic using multiple randomly chosen intruder words.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics
                                 and a topic-word matrix.
            new_Embeddings (bool, optional): Whether to recalculate embeddings. Defaults to True.

        Returns:
            float: The overall INT score for all topics with several intruder words.
        """
        if new_Embeddings:
            self.embeddings = None

        return float(np.mean(list(self.score_per_topic(model_output).values())))


class ISH(AbstractMetric):
    """
    For each topic, draw several intruder words that are not from the same topic by first selecting some topics that are not the specific topic and
    then selecting one word from each of those topics.
    The embedding intruder distance to mean is then calculated as the average distance that each intruder word has to the mean of the other words.
    """

    def __init__(
        self,
        dataset,
        n_intruders=1,
        n_words=10,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        Initializes the ISH object with a dataset, number of intruders, number of words,
        embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for ISIM calculation.
            n_intruders (int, optional): The number of intruder words to draw for each topic. Defaults to 1.
            n_words (int, optional): The number of top words to consider for each topic. Defaults to 10.
            metric_embedder (SentenceTransformer, optional): The embedding model to use.
            Defaults to SentenceTransformer("paraphrase-MiniLM-L6-v2").
            emb_filename (str, optional): Filename to store embeddings. Defaults to None.
            emb_path (str, optional): Path to store embeddings. Defaults to "Embeddings/".
            expansion_path (str, optional): Path for expansion embeddings. Defaults to "Embeddings/".
            expansion_filename (str, optional): Filename for expansion embeddings. Defaults to None.
            expansion_word_list (list, optional): List of words for expansion. Defaults to None.
        """

        tw_emb = Embed_corpus(
            dataset,
            metric_embedder,
            emb_filename=emb_filename,
            emb_path=emb_path,
        )
        if expansion_word_list is not None:
            tw_emb = Update_corpus_dic_list(
                expansion_word_list,
                tw_emb,
                metric_embedder,
                emb_filename=expansion_filename,
                emb_path=expansion_path,
            )

        self.n_intruders = n_intruders
        self.corpus_dict = tw_emb
        self.n_words = n_words
        self.embeddings = None
        """
        corpus_dict: dict that maps each word in the corpus to its embedding
        n_words: number of top words to consider
        """

        self.n_intruders = n_intruders

    def score(self, model_output, new_Embeddings=True):
        if new_Embeddings:
            self.embeddings = None
        """
        Calculate the score for all topics combined
        """

        return float(np.mean(list(self.score_per_topic(model_output).values())))

    def score_per_topic(self, model_output, new_Embeddings=None):
        if new_Embeddings:  # for this function, reuse embeddings per default
            self.embeddings = None

        topics_tw = model_output["topics"]

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
        ntopics = len(model_output["topics"])
        topic_words = model_output["topics"]
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = float(
                np.around(np.array(score_topic_list)[k], 5)
            )

        return results  # return the mean score for each topic
