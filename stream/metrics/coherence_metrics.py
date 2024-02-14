import numpy as np
import nltk
import re
from octis.evaluation_metrics.metrics import AbstractMetric
from ._helper_funcs import (
    cos_sim_pw,
    Embed_corpus,
    Embed_topic,
    Update_corpus_dic_list,
)
from octis.dataset.dataset import Dataset
from sentence_transformers import SentenceTransformer

nltk.download("stopwords")
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import gensim

gensim_stopwords = gensim.parsing.preprocessing.STOPWORDS


nltk_stopwords = stopwords.words("english")

stopwords = list(
    set(nltk_stopwords + list(gensim_stopwords) + list(ENGLISH_STOP_WORDS))
)


class NPMI(AbstractMetric):
    """
    A class for calculating Normalized Pointwise Mutual Information (NPMI) for topics.

    NPMI is a metric used in topic modeling to measure the coherence of topics by evaluating
    the co-occurrence of pairs of words across the documents. Higher NPMI scores typically
    indicate more coherent topics.

    Attributes:
        stopwords (list): A list of stopwords to exclude from analysis.
        ntopics (int): The number of topics to evaluate.
        dataset: The dataset used for calculating NPMI.
        files (list): Processed text data from the dataset.
    """

    def __init__(
        self,
        dataset,
        stopwords=stopwords,
        n_topics=20,
    ):
        """
        Initializes the NPMI object with a dataset, stopwords, and a specified number of topics.

        Parameters:
            dataset: The dataset to be used for NPMI calculation.
            stopwords (list, optional): A list of stopwords to exclude from analysis.
            n_topics (int, optional): The number of topics to evaluate. Defaults to 20.
        """
        self.stopwords = stopwords
        self.ntopics = n_topics
        self.dataset = dataset

        files = self.dataset.get_corpus()
        self.files = [" ".join(words) for words in files]

    def _create_vocab_preprocess(self, data, preprocess=5, process_data=False):
        """
        Creates and preprocesses a vocabulary from the given data.

        This method processes the text data to create a vocabulary, filtering out stopwords
        and applying other preprocessing steps.

        Parameters:
            data (list): The text data to process.
            preprocess (int): The minimum number of documents a word must appear in.
            process_data (bool, optional): Whether to return the processed data. Defaults to False.

        Returns:
            tuple: A tuple containing word-to-document mappings, multiple word-to-document mappings,
            and optionally processed data.
        """
        word_to_file = {}
        word_to_file_mult = {}

        process_files = []
        for file_num in range(0, len(data)):
            words = data[file_num].lower()
            words = words.strip()
            words = re.sub("[^a-zA-Z0-9]+\s*", " ", words)
            words = re.sub(" +", " ", words)
            # .translate(strip_punct).translate(strip_digit)
            words = words.split()
            # words = [w.strip() for w in words]
            proc_file = []

            for word in words:
                if word in self.stopwords or word == "dlrs" or word == "revs":
                    continue
                if word in word_to_file:
                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)
                else:
                    word_to_file[word] = set()
                    word_to_file_mult[word] = []

                    word_to_file[word].add(file_num)
                    word_to_file_mult[word].append(file_num)

            process_files.append(proc_file)

        for word in list(word_to_file):
            if len(word_to_file[word]) <= preprocess or len(word) <= 3:
                word_to_file.pop(word, None)
                word_to_file_mult.pop(word, None)

        if process_data:
            vocab = word_to_file.keys()
            files = []
            for proc_file in process_files:
                fil = []
                for w in proc_file:
                    if w in vocab:
                        fil.append(w)
                files.append(" ".join(fil))

            data = files

        return word_to_file, word_to_file_mult, data

    def _create_vocab_and_files(self, preprocess=5):
        """
        Creates vocabulary and files necessary for NPMI calculation.

        Parameters:
            preprocess (int, optional): The minimum number of documents a word must appear in.
                Defaults to 5.

        Returns:
            tuple: A tuple containing word-to-document mappings and other relevant data for NPMI calculation.
        """
        return self._create_vocab_preprocess(self.files, preprocess)

    def score(self, model_output):
        """
        Calculates the average NPMI score for the given model output.

        The method computes the NPMI score for each pair of words in every topic and then
        averages these scores to evaluate the overall topic coherence.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics.

        Returns:
            float: The average NPMI score for the topics.
        """
        topic_words = model_output["topics"]
        (
            word_doc_counts,
            dev_word_to_file_mult,
            dev_files,
        ) = self._create_vocab_and_files(preprocess=1)
        nfiles = len(dev_files)
        eps = 10 ** (-12)

        all_topics = []
        for k in range(self.ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw - 1):
                for j in range(i + 1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(
                        word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set())
                    )
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log(
                        (w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps
                    )
                    npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        avg_score = np.around(np.mean(all_topics), 5)

        return avg_score

    def score_per_topic(self, model_output, preprocess=5):
        """
        Calculates NPMI scores per topic for the given set of topics.

        This method evaluates the coherence of each topic individually by computing NPMI scores
        for each pair of words within the topic.

        Parameters:
            topic_words (list): A list of lists containing words in each topic.
            ntopics (int): The number of topics.
            preprocess (int, optional): The minimum number of documents a word must appear in.
                Defaults to 5.

        Returns:
            dict: A dictionary with topics as keys and their corresponding NPMI scores as values.
        """
        topic_words = model_output["topics"]
        ntopics = len(topic_words)

        (
            word_doc_counts,
            dev_word_to_file_mult,
            dev_files,
        ) = self._create_vocab_and_files(preprocess=preprocess)
        nfiles = len(dev_files)
        eps = 10 ** (-12)

        all_topics = []

        for k in range(ntopics):
            topic_score = []

            ntopw = len(topic_words[k])

            for i in range(ntopw - 1):
                for j in range(i + 1, ntopw):
                    w1 = topic_words[k][i]
                    w2 = topic_words[k][j]

                    w1w2_dc = len(
                        word_doc_counts.get(w1, set()) & word_doc_counts.get(w2, set())
                    )
                    w1_dc = len(word_doc_counts.get(w1, set()))
                    w2_dc = len(word_doc_counts.get(w2, set()))

                    # Correct eps:
                    pmi_w1w2 = np.log(
                        (w1w2_dc * nfiles) / ((w1_dc * w2_dc) + eps) + eps
                    )
                    npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(npmi_w1w2)

            all_topics.append(np.mean(topic_score))

        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(all_topics[k], 5)

        return results


class Embedding_Coherence(AbstractMetric):
    """
    A metric class to calculate the coherence of topics based on word embeddings. It computes
    the average cosine similarity between all top words in each topic.

    Attributes:
        n_words (int): The number of top words to consider for each topic.
        corpus_dict (dict): A dictionary mapping each word in the corpus to its embedding.
        embeddings (numpy.ndarray): The embeddings for the top words of the topics.
    """

    def __init__(
        self,
        dataset,
        n_words=10,
        metric_embedder=SentenceTransformer("paraphrase-MiniLM-L6-v2"),
        emb_filename=None,
        emb_path="Embeddings/",
        expansion_path="Embeddings/",
        expansion_filename=None,
        expansion_word_list=None,
    ):
        """
        Initializes the Embedding_Coherence object with a dataset, number of words,
        embedding model, and paths for storing embeddings.

        Parameters:
            dataset: The dataset to be used for embedding coherence calculation.
            n_words (int, optional): The number of top words to consider for each topic.
                Defaults to 10.
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

        self.n_words = n_words
        self.corpus_dict = tw_emb
        self.embeddings = None

    def score_per_topic(self, model_output):
        """
        Calculates coherence scores for each topic individually based on embedding similarities.

        This method computes the coherence of each topic by calculating the average pairwise
        cosine similarity between the embeddings of the top words in each topic.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics.

        Returns:
            numpy.ndarray: An array of coherence scores for each topic.
        """
        topics_tw = model_output["topics"]
        topic_words = model_output["topics"]
        ntopics = len(topic_words)

        emb_tw = Embed_topic(
            topics_tw, self.corpus_dict, self.n_words
        )  # embed the top words
        emb_tw = np.dstack(emb_tw).transpose(2, 0, 1)[
            :, : self.n_words, :
        ]  # create tensor of size (n_topics, n_topwords, n_embedding_dims)
        self.embeddings = emb_tw

        topic_sims = []
        for (
            topic_emb
        ) in (
            emb_tw
        ):  # for each topic append the average pairwise cosine similarity within its words
            topic_sims.append(float(cos_sim_pw(topic_emb)))

        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(np.array(topic_sims)[k], 5)

        return results

    def score(self, model_output):
        """
        Calculates the overall average coherence score for the given model output.

        This method computes the overall coherence of the topics by averaging the coherence
        scores obtained from each topic.

        Parameters:
            model_output (dict): The output of a topic model, containing a list of topics.

        Returns:
            float: The average coherence score for all topics.
        """
        res = self.score_per_topic(model_output).values()
        return sum(res) / len(res)


def _load_default_texts():
    """
    Loads default general texts

    Returns
    -------
    result : default 20newsgroup texts
    """
    dataset = Dataset()
    dataset.fetch_dataset("20NewsGroup")
    return dataset.get_corpus()
