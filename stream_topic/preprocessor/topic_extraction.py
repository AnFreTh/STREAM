import re
from itertools import compress

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from nltk import pos_tag
from nltk.corpus import brown as nltk_words
from nltk.corpus import words as eng_dict
from numpy.linalg import norm
from octis.dataset.dataset import Dataset as OCDataset

from ._embedder import BaseEmbedder


class TopicExtractor:
    """
    Class to extract topics from a given dataset and topic assignments.

    Args:
        dataset (Dataset): Dataset object
        topic_assignments (list): List of topic assignments
        n_topics (int): Number of topics
        embedding_model (str): Path to the embedding model

    """

    def __init__(
        self,
        dataset,
        topic_assignments,
        n_topics,
        embedding_model,
    ):
        self.dataset = dataset
        self.topic_assignments = topic_assignments
        self.embedder = BaseEmbedder(embedding_model)
        self.n_topics = n_topics

    def _noun_extractor_haystack(self, embeddings, n, corpus="octis", only_nouns=True):
        """
        Extracts the topics most probable words, which are the words nearest to the topics centroid.
        We extract all nouns from the corpus and the brown corpus. Afterwards we compute the cosine similarity between every word and every centroid.

        Note, that here we did not use the  sklearn.metrics.pairwise cosine_similarity function due to a faster computation when using numpy.
        Hecen we used:
            np.inner(centroids, nouns) / np.multiply.outer(
                norm(centroids, axis=1), norm(nouns, axis=1)
            )

        Args:
            embeddings (_type_): _document embeddings to compute centroid of the topic
            n (_type_): n_top number of words per topic

        Returns:
            dict: extracted topics
        """

        # define whether word is a noun
        def is_noun(pos):
            return pos[:2] == "NN"

        # not used for now
        # DATADIR = "../datasets/preprocessed_datasets"

        # extend the corpus
        if corpus == "brown":
            word_list = nltk_words.words()
            word_list = [word.lower().strip() for word in word_list]
            word_list = [re.sub(r"[^a-zA-Z0-9]+\s*", "", word)
                         for word in word_list]
        elif corpus == "words":
            word_list = eng_dict.words()
            word_list = [word.lower().strip() for word in word_list]
            word_list = [re.sub(r"[^a-zA-Z0-9]+\s*", "", word)
                         for word in word_list]
        elif corpus == "octis":
            data = OCDataset()
            data.fetch_dataset("20NewsGroup")
            word_list = data.get_vocabulary()
            data.fetch_dataset("M10")
            word_list += data.get_vocabulary()
            data.fetch_dataset("BBC_News")
            word_list += data.get_vocabulary()

            # include reuters etc datasets
            # data.load_custom_dataset_from_folder(DATADIR + "/GN")
            # word_list += data.get_vocabulary()

            word_list += self.dataset.get_vocabulary()

            word_list = [word.lower().strip() for word in word_list]
            word_list = [re.sub(r"[^a-zA-Z0-9]+\s*", "", word)
                         for word in word_list]
        else:
            raise ValueError(
                "There are no words to be extracted for the Topics: Please specify a corpus"
            )

        if only_nouns:
            word_list = [word for (word, pos) in pos_tag(
                word_list) if is_noun(pos)]
        else:
            word_list = [word for (word, pos) in pos_tag(word_list)]

        word_list = list(set(word_list))

        # embedd the noun_corpus
        nouns = self.embedder.create_word_embeddings(word_list)

        if isinstance(self.embedder.embedder, Word2VecKeyedVectors):
            word_list = list(compress(word_list, list(~pd.isnull(nouns))))
            nouns = nouns[~pd.isnull(nouns)]
            try:
                nouns.shape[1]
            except IndexError:
                nouns = np.stack([noun for noun in nouns])

        mean_embeddings = []

        # create topic centroids
        for t in range(self.n_topics):
            weighted_topic = np.multiply(
                np.array(self.topic_assignments[t])[:, np.newaxis], embeddings
            )
            mean_embedding = sum(weighted_topic) / len(embeddings)
            mean_embeddings.append(mean_embedding)

        topic_words = []
        topic_word_scores = []

        # compute cosine similarity between all nouns and the centroids
        res = np.inner(mean_embeddings, nouns) / np.multiply.outer(
            norm(mean_embeddings, axis=1), norm(nouns, axis=1)
        )

        # get the top words according to the similarity
        top_words = np.flip(np.argsort(res, axis=1), axis=1)
        top_scores = np.flip(np.sort(res, axis=1), axis=1)

        # for cleaner visualization
        for words, scores in zip(top_words, top_scores):
            topic_words.append([word_list[i] for i in words[0:n]])
            topic_word_scores.append(scores[0:n])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        # return as dict of lists
        topics = []
        for i in range(len(topic_words)):
            topics.append(list(zip(topic_words[i], topic_word_scores[i])))

        topics_ = {}

        for i in range(len(topics)):
            topics_[i] = topics[i]

        # return topics and centroid of topics
        return topics_, mean_embeddings
