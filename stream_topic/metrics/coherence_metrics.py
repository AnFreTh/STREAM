import re
import gensim
import numpy as np
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
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
import jieba
from collections import defaultdict 

GENSIM_STOPWORDS = gensim.parsing.preprocessing.STOPWORDS


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
        language: str = None,
        custom_stopwords: list = None,
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
        # Dynamically set NLTK stopwords based on the chosen language
        if custom_stopwords is None:
            try:
                nltk_stopwords = stopwords.words(language)
                self.language = language
            except OSError:
                print(f"Language '{language}' not supported in NLTK. Defaulting to English.")
                nltk_stopwords = stopwords.words(NLTK_STOPWORD_LANGUAGE)
                self.language = NLTK_STOPWORD_LANGUAGE
            self.custom_stopwords = list(set(list(nltk_stopwords) + list(GENSIM_STOPWORDS) + list(ENGLISH_STOP_WORDS)))
        else:
            self.custom_stopwords = custom_stopwords
            self.language = language
        self.dataset = dataset

        files = self.dataset.get_corpus()
        self.files = [" ".join(words) for words in files]

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
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
                words = list(jieba.cut(words))
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
        else:
            for file_num in range(0, len(data)):
                words = data[file_num]
                words = words.strip()
                words = re.sub(r"[^\u4e00-\u9fff\d]+", " ", words)
                words = re.sub(" +", " ", words)
                # .translate(strip_punct).translate(strip_digit)
                words = words.split()
                # words = [w.strip() for w in words]
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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

        Parameters
        ----------
        preprocess : int, optional
            The minimum number of documents a word must appear in. Defaults to 5.

        Returns
        -------
        tuple
            A tuple containing word-to-document mappings and other relevant data for NPMI calculation.
        """
        return self._create_vocab_preprocess(self.files, preprocess)

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

class PMI(BaseMetric):
    """
    A class for calculating Pointwise Mutual Information (PMI) for topics.

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
        language: str = None,
        custom_stopwords: list = None,
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
        # Dynamically set NLTK stopwords based on the chosen language
        if custom_stopwords is None:
            try:
                nltk_stopwords = stopwords.words(language)
                self.language = language
            except OSError:
                print(f"Language '{language}' not supported in NLTK. Defaulting to English.")
                nltk_stopwords = stopwords.words(NLTK_STOPWORD_LANGUAGE)
                self.language = NLTK_STOPWORD_LANGUAGE
            self.custom_stopwords = list(set(list(nltk_stopwords) + list(GENSIM_STOPWORDS) + list(ENGLISH_STOP_WORDS)))
        else:
            self.custom_stopwords = custom_stopwords
            self.language = language
        self.dataset = dataset

        files = self.dataset.get_corpus()
        self.files = [" ".join(words) for words in files]

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
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
                words = list(jieba.cut(words))
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
        else:
            for file_num in range(0, len(data)):
                words = data[file_num]
                words = words.strip()
                words = re.sub(r"[^\u4e00-\u9fff\d]+", " ", words)
                words = re.sub(" +", " ", words)
                # .translate(strip_punct).translate(strip_digit)
                words = words.split()
                # words = [w.strip() for w in words]
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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

        Parameters
        ----------
        preprocess : int, optional
            The minimum number of documents a word must appear in. Defaults to 5.

        Returns
        -------
        tuple
            A tuple containing word-to-document mappings and other relevant data for NPMI calculation.
        """
        return self._create_vocab_preprocess(self.files, preprocess)

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
                    # npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(pmi_w1w2)

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
                    # npmi_w1w2 = pmi_w1w2 / (-np.log((w1w2_dc) / nfiles + eps))

                    topic_score.append(pmi_w1w2)

            all_topics.append(np.mean(topic_score))

        results = {}
        for k in range(ntopics):
            half_topic_words = topic_words[k][
                : len(topic_words[k]) // 2
            ]  # Take only the first half of the words
            results[", ".join(half_topic_words)] = np.around(all_topics[k], 5)

        return results

class cPMI(BaseMetric):
    def __init__(
        self, 
        dataset, 
        language: str = None,
        custom_stopwords: list = None,
        threshold=50, 
        epsilon=0.7, 
        delta=0.9
        ):
        self.threshold  = threshold 
        self.epsilon  = epsilon 
        self.delta  = delta 
        self.documents = list(dataset.dataframe['text'].apply(lambda x: x.split()))
        self.K = len(self.documents) 
        self.l = [len(doc) for doc in self.documents] 
        
        self.index_mappings  = self._build_index_mapping()
    
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
            "metric_name": "cPMI",
            "description": "cPMI coherence",
        }

        return info
    
    def _build_index_mapping(self):
        """构建词索引映射"""
        index_mappings = []
        for doc in self.documents: 
            index_map = defaultdict(list)
            for idx, word in enumerate(doc):
                index_map[word].append(idx)
            index_mappings.append(index_map) 
        return index_mappings 
    
    def _count_nonoverlapping_intervals(self, word_pairs):
        """计算非重叠区间"""
        results = {pair: [] for pair in word_pairs}
        
        for index_map in self.index_mappings: 
            for word1, word2 in word_pairs:
                positions1 = sorted(index_map.get(word1,  []))
                positions2 = sorted(index_map.get(word2,  []))
                
                if not positions1 or not positions2:
                    results[(word1, word2)].append(0)
                    continue 
                
                # 确定哪个词先出现 
                if positions1[0] <= positions2[0]:
                    positions_a, positions_b = positions1, positions2 
                else:
                    positions_a, positions_b = positions2, positions1 
                
                # 计算非重叠区间 
                count = 0 
                last_start, last_end = -1, -1 
                i, j = 0, 0 
                
                while i < len(positions_a) and j < len(positions_b):
                    a, b = positions_a[i], positions_b[j]
                    
                    if a < b:
                        if a > last_end and b > last_end:
                            count += 1 
                            last_start, last_end = a, b 
                            i += 1 
                            j += 1 
                        else:
                            i += 1 
                    else:
                        if b > last_end and a > last_end:
                            count += 1 
                            last_start, last_end = b, a 
                            i += 1 
                            j += 1 
                        else:
                            j += 1 
                
                results[(word1, word2)].append(count)
        
        return results 
    
    def _count_nonoverlapping_intervals_with_threshold(self, word_pairs):
        """计算带阈值的非重叠区间"""
        results = {pair: [] for pair in word_pairs}
        
        for index_map in self.index_mappings: 
            for word1, word2 in word_pairs:
                positions1 = sorted(index_map.get(word1,  []))
                positions2 = sorted(index_map.get(word2,  []))
                
                if not positions1 or not positions2:
                    results[(word1, word2)].append(0)
                    continue 
                
                # 确定哪个词先出现 
                if positions1[0] <= positions2[0]:
                    positions_a, positions_b = positions1, positions2 
                else:
                    positions_a, positions_b = positions2, positions1 
                
                # 计算非重叠区间 
                count = 0 
                last_start, last_end = -1, -1 
                i, j = 0, 0 
                
                while i < len(positions_a) and j < len(positions_b):
                    a, b = positions_a[i], positions_b[j]
                    
                    if abs(a - b) - 1 > self.threshold: 
                        if a < b:
                            i += 1 
                        else:
                            j += 1 
                        continue 
                    
                    if a > last_end and b > last_end:
                        count += 1 
                        last_start, last_end = min(a, b), max(a, b)
                        i += 1 
                        j += 1 
                    else:
                        if a <= last_end:
                            i += 1 
                        if b <= last_end:
                            j += 1 
                
                results[(word1, word2)].append(count)
        
        return results 
    
    @staticmethod 
    def _compute_hist_optimized(f, l, x, memo=None):
        """优化后的直方图计算"""
        if memo is None:
            memo = {}
            
        if (f, l) in memo:
            return memo[(f, l)]
        
        hist_f_l = np.zeros(f  + 1, dtype=np.float64) 
        
        if f > l:
            return hist_f_l 
        if f == 0:
            hist_f_l[0] = 1 
            memo[(f, l)] = hist_f_l 
            return hist_f_l 
        
        for i in range(1, l):
            for j in range(i + 1, l + 1):
                hist_f_minus_1_l_minus_j = cPMI._compute_hist_optimized(f - 1, l - j, x, memo)
                
                for k in range(f):
                    if (j - i) < x:
                        hist_f_l[k + 1] += hist_f_minus_1_l_minus_j[k]
                    else:
                        hist_f_l[k] += hist_f_minus_1_l_minus_j[k]
        
        memo[(f, l)] = hist_f_l 
        return hist_f_l 
    def score(self, topics):
        """计算主题列表的 CSR 评分"""
        self.scores  = {}
        self.topic_scores  = []
        
        for topic in topics:
            # 将主题列表转换为字符串形式 
            topic_str = " ".join(topic)
            # 生成所有可能的两两单词对 
            word_pairs = []
            for i in range(len(topic)):
                for j in range(i+1, len(topic)):
                    word_pairs.append((topic[i],  topic[j]))
            
            # 初始化结果字典 
            result_f = {}
            result_f_hat = {}
            
            for word_pair in word_pairs:
                # 计算非重叠区间 
                f = self._count_nonoverlapping_intervals([word_pair])
                f_hat = self._count_nonoverlapping_intervals_with_threshold([word_pair])
                result_f[word_pair] = f[word_pair]
                result_f_hat[word_pair] = f_hat[word_pair]
            
            # 计算所有单词对的pi和pi2 
            all_pi = []
            all_pi2 = []
            
            for word_pair in word_pairs:
                f_values = result_f[word_pair]
                f_hat_values = result_f_hat[word_pair]
                
                pi = []
                pi2 = []
                
                for i in range(self.K):
                    N_fl = self._compute_hist_optimized(f_values[i], self.l[i], self.threshold) 
                    total = np.sum(N_fl) 
                    
                    # 计算p_i 
                    pi_val = np.sum(N_fl[f_hat_values[i]:])  / total if total != 0 else 0 
                    pi.append(pi_val) 
                    
                    # 找第一个满足条件的j 
                    found = False 
                    for j in range(f_hat_values[i] + 1):
                        current_sum = np.sum(N_fl[j:])  / total if total != 0 else 0 
                        if current_sum < self.epsilon: 
                            pi2_val = current_sum 
                            pi2.append(pi2_val) 
                            found = True 
                            break 
                    if not found:
                        pi2.append(0)   # 如果没有找到，则设为0 
                
                all_pi.extend(pi) 
                all_pi2.extend(pi2) 
            
            Z = sum(1 for prob in all_pi if prob < self.epsilon) 
            EZ = sum(all_pi2)
            
            # 计算CSR 
            denominator = EZ + np.sqrt(-self.K  * np.log(self.delta)  / 2)
            csr_value = Z / denominator if denominator != 0 else 0 
            
            self.scores[topic_str]  = csr_value 
            self.topic_scores.append(csr_value) 
        
        return sum(self.topic_scores)  / len(topics) if topics else 0 
    
    def score_per_topic(self):
        """返回每个主题的 CSR 评分"""
        return self.scores  

class cPMI_d(BaseMetric):
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
        language: str = None,
        custom_stopwords: list = None,
        delta: float = 0.9,
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
        # Dynamically set NLTK stopwords based on the chosen language
        if custom_stopwords is None:
            try:
                nltk_stopwords = stopwords.words(language)
                self.language = language
            except OSError:
                print(f"Language '{language}' not supported in NLTK. Defaulting to English.")
                nltk_stopwords = stopwords.words(NLTK_STOPWORD_LANGUAGE)
                self.language = NLTK_STOPWORD_LANGUAGE
            self.custom_stopwords = list(set(list(nltk_stopwords) + list(GENSIM_STOPWORDS) + list(ENGLISH_STOP_WORDS)))
        else:
            self.custom_stopwords = custom_stopwords
            self.language = language
        self.dataset = dataset
        self.delta = delta

        files = self.dataset.get_corpus()
        self.files = [" ".join(words) for words in files]

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
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
                words = list(jieba.cut(words))
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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
        else:
            for file_num in range(0, len(data)):
                words = data[file_num]
                words = words.strip()
                words = re.sub(r"[^\u4e00-\u9fff\d]+", " ", words)
                words = re.sub(" +", " ", words)
                # .translate(strip_punct).translate(strip_digit)
                words = words.split()
                # words = [w.strip() for w in words]
                proc_file = []

                for word in words:
                    if word in self.custom_stopwords or word == "dlrs" or word == "revs":
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

        Parameters
        ----------
        preprocess : int, optional
            The minimum number of documents a word must appear in. Defaults to 5.

        Returns
        -------
        tuple
            A tuple containing word-to-document mappings and other relevant data for NPMI calculation.
        """
        return self._create_vocab_preprocess(self.files, preprocess)

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
                    cpmi_d = w1w2_dc/((w1_dc * w2_dc / nfiles) + np.sqrt(-w1_dc * np.log(self.delta) / 2) + eps)

                    topic_score.append(cpmi_d)

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
                    cpmi_d = w1w2_dc/((w1_dc * w2_dc / nfiles) + np.sqrt(-w1_dc * np.log(self.delta) / 2) + eps)

                    topic_score.append(cpmi_d)

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
            "metric_name": "Embedding_Coherence",
            "n_words": self.n_words,
            "description": "Embedding_Coherence",
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
