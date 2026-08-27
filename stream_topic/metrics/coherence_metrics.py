import re
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from .base import BaseMetric
from ._helper_funcs import cos_sim_pw
from .constants import (
    EMBEDDING_PATH,
    NLTK_STOPWORD_LANGUAGE,
    PARAPHRASE_TRANSFORMER_MODEL,
)
from .TopwordEmbeddings import TopwordEmbeddings
import os
from .metrics_config import MetricsConfig
try:
    import jieba  # optional; only needed for the (disabled) Chinese tokenization path
except ImportError:
    jieba = None
from collections import defaultdict


NLTK_STOPWORDS = stopwords.words(NLTK_STOPWORD_LANGUAGE)
STOPWORDS = list(
    set(list(NLTK_STOPWORDS) + list(ENGLISH_STOP_WORDS))
)


class CV(BaseMetric):
    """
    Gensim-based C_V coherence metric.

    C_V combines sliding window segmentation, indirect cosine similarity,
    and NPMI confirmation, achieving the highest correlation with human
    topic ratings among automated coherence measures (Röder et al., 2015).

    Parameters
    ----------
    dataset : TMDataset
        The dataset used for computing coherence.
    n_words : int, optional
        Number of top words per topic to evaluate. Defaults to 10.

    Examples
    --------
    >>> from stream_topic.metrics import CV
    >>> cv = CV(dataset)
    >>> score = cv.score(topics)
    """

    def __init__(self, dataset, n_words=10):
        self.n_words = n_words
        # Cache the tokenized texts + gensim Dictionary on the dataset object.
        # Both depend only on dataset.dataframe["text"] (not on n_words or the
        # topics being scored), yet CV is re-instantiated once per top-k cutoff
        # and per (model, seed), so the full-corpus Dictionary build is repeated
        # hundreds of times per dataset. The cached artifacts are read-only inputs
        # to gensim's CoherenceModel (verified not to mutate the dictionary), so
        # scores are bit-identical to building them fresh each call.
        cache = getattr(dataset, "_cv_cache", None)
        if cache is None:
            texts = [doc.split() for doc in dataset.dataframe["text"].tolist()]
            dictionary = Dictionary(texts)
            cache = (texts, dictionary)
            try:
                dataset._cv_cache = cache
            except Exception:
                pass
        self.texts, self.dictionary = cache

    def get_info(self):
        return {
            "metric_name": "C_V",
            "n_words": self.n_words,
            "description": "Gensim C_V coherence (Röder et al., 2015)",
        }

    def score(self, topic_words):
        """
        Compute average C_V coherence across all topics.

        Parameters
        ----------
        topic_words : list of list of str
            Top words per topic.

        Returns
        -------
        float
            Average C_V score.
        """
        topics_trimmed = [t[: self.n_words] for t in topic_words]
        cm = CoherenceModel(
            topics=topics_trimmed,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence="c_v",
        )
        return float(np.around(cm.get_coherence(), 5))

    def score_per_topic(self, topic_words):
        """
        Compute C_V coherence per topic.

        Parameters
        ----------
        topic_words : list of list of str
            Top words per topic.

        Returns
        -------
        dict
            Topic string -> C_V score.
        """
        topics_trimmed = [t[: self.n_words] for t in topic_words]
        cm = CoherenceModel(
            topics=topics_trimmed,
            texts=self.texts,
            dictionary=self.dictionary,
            coherence="c_v",
        )
        per_topic = cm.get_coherence_per_topic()
        results = {}
        for k, score in enumerate(per_topic):
            half = topic_words[k][: len(topic_words[k]) // 2]
            results[", ".join(half)] = float(np.around(score, 5))
        return results


class NPMI(BaseMetric):
    """
    A class for calculating Normalized Pointwise Mutual Information (NPMI) for topics.

    NPMI is a metric used in topic modeling to measure the coherence of topics by evaluating
    the co-occurrence of pairs of words across the documents. Higher NPMI scores typically
    indicate more coherent topics.

    Attributes
    ----------
    stopwords : list
        A list of stopwords to exclude from analysis.
    ntopics : int
        The number of topics to evaluate.
    dataset
        The dataset used for calculating NPMI.
    files : list
        Processed text data from the dataset.

    Examples
    --------
    >>> from stream_topic.metrics import NPMI
    >>> npmi = NPMI(dataset)
    >>> avg_npmi_score = npmi.score(topic_words)
    >>> print("Average NPMI score:", avg_npmi_score)
    >>> per_topic_scores = npmi.score_per_topic(topic_words)
    >>> print("NPMI scores per topic:", per_topic_scores)
    """

    def __init__(
        self,
        dataset,
        language: str = NLTK_STOPWORD_LANGUAGE,
        stopwords: list = None,
    ):
        """
        Initializes the NPMI object with a dataset, stopwords, and a specified number of topics.

        Parameters
        ----------
        dataset
            The dataset to be used for NPMI calculation.
        stopwords : list, optional
            A list of stopwords to exclude from analysis. Default includes GenSim, NLTK, and Scikit-learn stopwords.
        """
        if stopwords is None:
            if language != "chinese":
                self.stopwords = STOPWORDS
            else:
                raise ValueError(f"Please provide Chinese stopwords list!")
        else:
            with open(stopwords, 'r', encoding='UTF-8') as f:
                self.stopwords = [line.strip() for line in f]
        self.language = language
        self.dataset = dataset

        # Cache the joined corpus on the dataset: get_corpus() + join is
        # deterministic in the dataset and NPMI is re-instantiated once per top-k
        # cutoff (and per model/seed). Read-only downstream, so identical.
        files_cache = getattr(dataset, "_npmi_files", None)
        if files_cache is None:
            files = self.dataset.get_corpus()
            files_cache = [" ".join(words) for words in files]
            try:
                dataset._npmi_files = files_cache
            except Exception:
                pass
        self.files = files_cache

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
            "metric_name": "NPMI",
            "n_words": self.n_words,
            "description": "NPMI coherence",
        }

        return info

    def _create_vocab_preprocess(self, data, preprocess=5, process_data=False):
        """
        Creates and preprocesses a vocabulary from the given data.

        This method processes the text data to create a vocabulary, filtering out stopwords
        and applying other preprocessing steps.

        Parameters
        ----------
        data : list
            The text data to process.
        preprocess : int
            The minimum number of documents a word must appear in.
        process_data : bool, optional
            Whether to return the processed data. Defaults to False.

        Returns
        -------
        tuple
            A tuple containing word-to-document mappings, multiple word-to-document mappings,
            and optionally processed data.
        """
        word_to_file = {}
        word_to_file_mult = {}

        process_files = []
        if self.language == NLTK_STOPWORD_LANGUAGE:
            for file_num in range(0, len(data)):
                words = data[file_num].lower()
                words = words.strip()
                words = re.sub(r"[^a-zA-Z0-9]+\s*", " ", words)
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
        elif self.language == "chinese":
            for file_num in range(0, len(data)):
                words = data[file_num]
                words = words.strip()
                words = re.sub(r"[^\u4e00-\u9fff\d]+", " ", words)
                words = re.sub(" +", " ", words)
                # words = list(jieba.cut(words))
                words = words.split()
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

        if self.language == "chinese":
            for word in list(word_to_file):
                if len(word_to_file[word]) <= preprocess or len(word) <= 1:
                    word_to_file.pop(word, None)
                    word_to_file_mult.pop(word, None)
        else:
            for word in list(word_to_file):
                # Keep words of length >= 3 to match the benchmark preprocessing
                # (min_word_length=3). Dropping len<=3 here excised 3-char topic
                # words (war, law, tax, oil, ...) from the co-occurrence vocab
                # while models still emit them, forcing every pair involving them
                # to NPMI=-1 and biasing NPMI down unevenly across models.
                if len(word_to_file[word]) <= preprocess or len(word) < 3:
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

        Parameters
        ----------
        preprocess : int, optional
            The minimum number of documents a word must appear in. Defaults to 5.

        Returns
        -------
        tuple
            A tuple containing word-to-document mappings and other relevant data for NPMI calculation.
        """
        # Memoize the (word_to_file, word_to_file_mult, data) triple per dataset.
        # It is a pure function of (self.files, self.language, self.stopwords,
        # preprocess) -- none of which depend on the topics being scored -- yet
        # score() rebuilds the full-corpus co-occurrence vocab on every call
        # (4 top-k cutoffs x every model x seed). The returned structures are only
        # read (via .get()) in score()/score_per_topic(), so results are identical.
        key = (self.language, id(self.stopwords), preprocess)
        cache = getattr(self.dataset, "_npmi_vocab_cache", None)
        if cache is None:
            cache = {}
            try:
                self.dataset._npmi_vocab_cache = cache
            except Exception:
                pass
        if key not in cache:
            cache[key] = self._create_vocab_preprocess(self.files, preprocess)
        return cache[key]

    def score(self, topic_words):
        """
        Calculates the average NPMI score for the given model output.

        The method computes the NPMI score for each pair of words in every topic and then
        averages these scores to evaluate the overall topic coherence.

        Parameters
        ----------
        topic_words : list of list of str
            The output of a topic model, containing a list of topics.

        Returns
        -------
        float
            The average NPMI score for the topics.
        """
        self.ntopics = len(topic_words)
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

    def score_per_topic(self, topic_words, preprocess=5):
        """
        Calculates NPMI scores per topic for the given set of topics.

        This method evaluates the coherence of each topic individually by computing NPMI scores
        for each pair of words within the topic.

        Parameters
        ----------
        topic_words : list of list of str
            A list of lists containing words in each topic.
        preprocess : int, optional
            The minimum number of documents a word must appear in. Defaults to 5.

        Returns
        -------
        dict
            A dictionary with topics as keys and their corresponding NPMI scores as values.
        """

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

class Embedding_Coherence(BaseMetric):
    """
    A metric class to calculate the coherence of topics based on word embeddings. It computes
    the average cosine similarity between all top words in each topic.

    Attributes
    ----------
    n_words : int
        The number of top words to consider for each topic.
    metric_embedder : SentenceTransformer
        The SentenceTransformer model to use for embedding.

    Examples
    --------
    >>> metric = Embedding_Coherence()
    >>> topic_scores = metric.score_per_topic(topics)
    >>> print("Coherence scores per topic:", topic_scores)
    >>> overall_score = metric.score(topics)
    >>> print("Overall coherence score:", overall_score)
    """

    def __init__(
        self,
        n_words=10,
        metric_embedder: str = None,
        emb_filename=None,
        emb_path: str = EMBEDDING_PATH,
    ):
        """
        Initializes the Embedding_Coherence object with the number of top words to consider
        and the embedding model to use.

        Parameters
        ----------
        n_words : int, optional
            The number of top words to consider for each topic. Defaults to 10.
        metric_embedder : SentenceTransformer, optional
            The SentenceTransformer model to use for embedding. Defaults to "paraphrase-MiniLM-L6-v2".
        emb_filename : str, optional
            The filename for the embedding model. Defaults to None.
        emb_path : str, optional
            The path to the embedding model. Defaults to EMBEDDING_PATH.
        """

        # Check if embedder is a local path or model name and load accordingly
        if not metric_embedder:
            metric_embedder_name = MetricsConfig.PARAPHRASE_embedder or PARAPHRASE_TRANSFORMER_MODEL
            if os.path.exists(metric_embedder_name):
                print(f"Loading model from local path: {metric_embedder_name}")
                metric_embedder = SentenceTransformer(metric_embedder_name)
            else:
                print(f"Downloading model: {metric_embedder_name}")
                metric_embedder = SentenceTransformer(metric_embedder_name)
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
            number of top words, number of intruders, embedding model name,
            metric range and metric description.
        """

        info = {
            "metric_name": "Embedding Coherence",
            "n_words": self.n_words,
            "description": "Embedding Coherence coherence",
        }

        return info
    
    def score_per_topic(self, topics):
        """
        Calculates coherence scores for each topic individually based on embedding similarities.

        This method computes the coherence of each topic by calculating the average pairwise
        cosine similarity between the embeddings of the top words in each topic.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.

        Returns
        -------
        dict
            A dictionary where the keys are comma-separated top words of each topic and the values are the coherence scores.
        """
        topics = topics
        n_topics = len(topics)
        topwords_embedded = self.topword_embeddings.embed_topwords(
            topics, n_topwords_to_use=self.n_words
        )

        topic_sims = []
        for (
            topic_emb
        ) in (
            topwords_embedded
        ):  # for each topic append the average pairwise cosine similarity within its words
            topic_sims.append(float(cos_sim_pw(topic_emb)))

        results = {}
        for k in range(n_topics):
            half_topic_words = topics[k][
                : len(topics[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(np.array(topic_sims)[k], 5)

        return results

    def score(self, topics):
        """
        Calculates the overall average coherence score for the given model output.

        This method computes the overall coherence of the topics by averaging the coherence
        scores obtained from each topic.

        Parameters
        ----------
        topics : list of list of str
            The output of a topic model, containing a list of topics.

        Returns
        -------
        float
            The average coherence score for all topics.
        """
        res = self.score_per_topic(topics).values()
        return sum(res) / len(res)
