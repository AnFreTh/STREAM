from gensim.models import Word2Vec
import umap.umap_ as umap
from octis.models.model import AbstractModel
import numpy as np
from ..data_utils.dataset import TMDataset
from ..utils.embedder import BaseEmbedder, GensimBackend
from sklearn.mixture import GaussianMixture
import pandas as pd


class WordCluTM(AbstractModel):
    """
    A topic modeling class that uses Word2Vec embeddings and K-Means or GMM clustering on vocabulary to form coherent word clusters.
    """

    def __init__(
        self,
        num_topics: int = 20,
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 1,
        workers: int = 4,
        umap_args: dict = None,
        gmm_args: dict = {},
        random_state: int = None,
    ):
        """
        Initialize the WordCluTM model.

        Args:
            num_topics (int): Number of topics.
            vector_size (int): Dimensionality of the word vectors.
            window (int): Maximum distance between the current and predicted word within a sentence.
            min_count (int): Ignores all words with total frequency lower than this.
            workers (int): Number of worker threads to train the Word2Vec model.
            umap_args (dict): Arguments for UMAP dimensionality reduction.
            gmm_args (dict): Arguments for Gaussian Mixture Model (GMM).
            random_state (int): Random seed.
        """
        super().__init__()
        self.n_topics = num_topics
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.umap_args = (
            umap_args
            if umap_args
            else {
                "n_neighbors": 15,
                "n_components": 7,
                "metric": "cosine",
                "random_state": random_state,
            }
        )
        self.trained = False

        if gmm_args:
            self.gmm_args = gmm_args
        else:
            self.gmm_args = {
                "n_components": self.n_topics,
                "covariance_type": "full",
                "tol": 0.001,
                "reg_covar": 0.000001,
                "max_iter": 100,
                "n_init": 1,
                "init_params": "kmeans",
            }

    def train_word2vec(self, sentences, epochs):
        """
        Train a Word2Vec model on the given sentences.

        Args:
            sentences (list): List of tokenized sentences.
        """
        # Initialize Word2Vec model
        self.word2vec_model = Word2Vec(
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )

        # Build the vocabulary from the sentences
        self.word2vec_model.build_vocab(sentences)

        # Train the Word2Vec model
        self.word2vec_model.train(
            sentences, total_examples=len(sentences), epochs=epochs
        )

        # Initialize BaseEmbedder with GensimBackend
        self.base_embedder = BaseEmbedder(GensimBackend(self.word2vec_model.wv))

    def _clustering(self):
        """
        Perform clustering on the reduced embeddings.

        Returns:
            tuple: Soft labels and hard labels of the clusters.
        """
        assert (
            hasattr(self, "reduced_embeddings") and self.reduced_embeddings is not None
        ), "Reduced embeddings must be generated before clustering."

        try:
            self.GMM = GaussianMixture(
                **self.gmm_args,
            ).fit(self.reduced_embeddings)

            gmm_predictions = self.GMM.predict_proba(self.reduced_embeddings)
            self.soft_labels = gmm_predictions
            self.labels = self.GMM.predict(self.reduced_embeddings)
            return self.soft_labels, self.labels

        except Exception as e:
            raise ValueError(f"Error in clustering: {e}")

    def _dim_reduction(self, embeddings):
        """
        Perform dimensionality reduction on the word embeddings.

        Args:
            embeddings (numpy.ndarray): Word embeddings.

        Returns:
            numpy.ndarray: Reduced embeddings.
        """
        reducer = umap.UMAP(**self.umap_args)
        reduced_embeddings = reducer.fit_transform(embeddings)
        return reduced_embeddings

    def train_model(self, dataset, n_words=10, word2vec_epochs=100):
        """
        Train the WordCluTM model.

        Args:
            dataset (TMDataset): Dataset instance.
            n_words (int): Number of top words to include in each topic.

        Returns:
            dict: Output containing topics, topic-word matrix, topic dictionary, and topic-document matrix.
        """
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        self.dataset = dataset

        print("--- preparing the dataset ---")
        sentences = dataset.get_corpus()
        self.dataset.get_dataframe()
        self.dataframe = self.dataset.dataframe
        print("--- Train Word2Vec ---")
        self.train_word2vec(sentences, word2vec_epochs)  # Train Word2Vec model

        print("--- Compute Word Embeddings ---")
        unique_words = list(set(word for sentence in sentences for word in sentence))
        word_to_index = {word: i for i, word in enumerate(unique_words)}
        word_embeddings = np.array(
            [
                (
                    self.word2vec_model.wv[word]
                    if word in self.word2vec_model.wv
                    else np.zeros(self.vector_size)
                )
                for word in unique_words
            ]
        )

        print("--- Compute Document Embeddings ---")
        # Initialize an empty list to hold document embeddings
        self.doc_embeddings = []

        # Iterate over each document to compute its embedding
        for doc in sentences:
            doc_embedding = np.mean(
                [
                    word_embeddings[word_to_index[word]]
                    for word in doc
                    if word in word_to_index
                ],
                axis=0,
            )
            self.doc_embeddings.append(doc_embedding)

        # Convert the list of document embeddings to a numpy array
        self.doc_embeddings = np.array(self.doc_embeddings)

        print("--- Dimensionality Reduction ---")
        self.reduced_embeddings = self._dim_reduction(word_embeddings)

        print("--- Clustering ---")
        self.labels, _ = self._clustering()

        topics = {}

        # Iterate over each cluster
        for cluster_idx in range(self.labels.shape[1]):
            # Get the indices of the words sorted by their probability of belonging to this cluster, in descending order
            sorted_indices = np.argsort(self.labels[:, cluster_idx])[::-1]

            # Get the top n_words for this cluster based on the sorted indices
            top_words = [
                (unique_words[i], self.labels[i, cluster_idx])
                for i in sorted_indices[:n_words]
            ]

            # Store the top words and their probabilities in the dictionary
            topics[cluster_idx] = top_words

        words_list = []
        new_topics = {}
        for k in range(self.n_topics):
            words = [
                word
                for t in topics[k][0:n_words]
                for word in t
                if isinstance(word, str)
            ]
            weights = [
                weight
                for t in topics[k][0:10]
                for weight in t
                if isinstance(weight, float)
            ]
            weights = [weight / sum(weights) for weight in weights]
            new_topics[k] = list(zip(words, weights))
            words_list.append(words)

        corpus = (
            dataset.get_corpus()
        )  # List of documents, each document is a list of words

        # Initialize a dictionary to store the document scores for each label
        document_scores_per_label = {label: [] for label in range(len(topics))}

        # Convert top_words_per_cluster to a more easily searchable structure
        word_scores_per_label = {}
        for label, words_scores in topics.items():
            for word, score in words_scores:
                if word not in word_scores_per_label:
                    word_scores_per_label[word] = {}
                word_scores_per_label[word][label] = score

        # Iterate over each document
        for doc in corpus:
            # Initialize a score accumulator for each label for the current document
            doc_scores = {label: [] for label in range(len(topics))}

            # Iterate over each word in the document
            for word in doc:
                if word in word_scores_per_label:
                    # If the word has scores for any label, add those scores to the accumulator
                    for label, score in word_scores_per_label[word].items():
                        doc_scores[label].append(score)

            # Average the scores for each label and store them
            for label in doc_scores:
                if doc_scores[label]:  # Check if there are any scores to average
                    document_scores_per_label[label].append(np.mean(doc_scores[label]))
                else:
                    # If no scores for this label, you might want to set a default value
                    document_scores_per_label[label].append(0)

        n_documents = len(dataset.get_corpus())

        # Initialize the final array with zeros
        final_scores = np.zeros((self.n_topics, n_documents))

        # Populate the array with the computed scores
        for label, scores in document_scores_per_label.items():
            # Ensure there are as many scores as there are documents
            assert (
                len(scores) == n_documents
            ), "The number of scores must match the number of documents"

            # Assign the scores for this label (topic) to the corresponding row in the final array
            final_scores[label, :] = scores

        self.output = {}
        self.output["topics"] = words_list
        self.output["topic-word-matrix"] = self.labels
        self.output["topic_dict"] = topics
        self.output["topic-document-matrix"] = final_scores

        self.trained = True

        return self.output
