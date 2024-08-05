import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm


def get_top_tfidf_words_per_document(corpus, n=10):
    """
    Get the top TF-IDF words per document in a corpus.

    Args:
        corpus (list): List of documents.
        n (int, optional): Number of top words to retrieve per document (default is 10).

    Returns:
        list: A list of lists containing the top TF-IDF words for each document in the corpus.
    """
    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(corpus)
    feature_names = vectorizer.get_feature_names_out()

    top_words_per_document = []
    for row in X:
        sorted_indices = np.argsort(row.toarray()).flatten()[::-1]
        top_n_indices = sorted_indices[:n]
        top_words = [feature_names[i] for i in top_n_indices]
        top_words_per_document.append(top_words)

    return top_words_per_document


class DocumentCoherence:
    """
    A class for calculating the coherence between documents based on their top words.
    This is achieved through the use of Normalized Pointwise Mutual Information (NPMI).

    Attributes:
        documents (DataFrame): DataFrame containing documents and their top words.
        column (str): Column name in DataFrame that contains the top words for each document.
        stopwords (set): Set of stopwords to exclude from analysis.
        word_index (dict): Dictionary mapping each unique word to a unique index.
        doc_word_matrix (csr_matrix): Sparse matrix representing the occurrence of words in documents.
    """

    def __init__(self, documents, column="tfidf_top_words", stopwords=None):
        """
        Initializes the DocumentCoherence object with a DataFrame of documents.

        Parameters:
            documents (DataFrame): DataFrame containing documents and their top words.
            column (str): The column name in the DataFrame that contains the top words for each document.
            stopwords (list, optional): List of stopwords to exclude from analysis.
        """
        self.documents = documents
        self.column = column
        self.stopwords = set(stopwords) if stopwords else set()
        self.word_index = self._create_word_index()
        self.doc_word_matrix = self._create_doc_word_matrix()

    def _create_word_index(self):
        unique_words = set()
        for words in self.documents[self.column]:
            unique_words.update(words)
        unique_words -= self.stopwords
        return {word: idx for idx, word in enumerate(unique_words)}

    def _create_doc_word_matrix(self):
        print("--- create doc-word-matrix ---")
        rows, cols = [], []
        for idx, words in enumerate(self.documents[self.column]):
            words = set(words) - self.stopwords
            for word in words:
                if word in self.word_index:
                    rows.append(idx)
                    cols.append(self.word_index[word])
        data = [1] * len(rows)

        return csr_matrix(
            (data, (rows, cols)),
            shape=(
                len(self.documents),
                len(self.word_index),
            ),
        )

    def _calculate_co_occurrences(self):
        # Matrix multiplication to find co-occurrences
        return self.doc_word_matrix.T.dot(self.doc_word_matrix)

    def _calculate_npmi(self, co_occurrences, n_documents):
        eps = 1e-12
        word_prob = np.array(self.doc_word_matrix.sum(
            axis=0) / n_documents).flatten()

        # Convert sparse matrix to dense for the operation
        joint_prob = co_occurrences.toarray() / n_documents

        # Calculate PMI
        pmi = np.log((joint_prob + eps) /
                     (np.outer(word_prob, word_prob) + eps))

        # Calculate NPMI
        npmi = pmi / -np.log(joint_prob + eps)

        return npmi

    def calculate_document_coherence(self):
        """
        Calculate document coherence scores based on NP (Normalized Pointwise) Mutual Information (NPMI).

        Returns:
            pd.DataFrame: A DataFrame containing coherence scores between each pair of documents.
        """
        n_documents = self.doc_word_matrix.shape[0]
        co_occurrences = self._calculate_co_occurrences()
        npmi_matrix = self._calculate_npmi(co_occurrences, n_documents)

        # Initialize DataFrame for coherence scores
        coherence_scores = pd.DataFrame(
            np.nan, index=self.documents.index, columns=self.documents.index
        )

        # Precompute nonzero indices for each document
        doc_nonzero_indices = [
            set(self.doc_word_matrix[i, :].nonzero()[1]) for i in range(n_documents)
        ]

        for i in tqdm(range(n_documents)):
            for j in range(
                i + 1, n_documents
            ):  # Avoid redundant calculations by only doing one half of the matrix
                combined_indices = doc_nonzero_indices[i] & doc_nonzero_indices[j]
                if combined_indices:
                    selected_npmi_values = npmi_matrix[list(combined_indices)][
                        :, list(combined_indices)
                    ]
                    coherence_score = np.nanmean(selected_npmi_values)
                    coherence_scores.iat[i, j] = coherence_score
                    # Symmetric matrix
                    coherence_scores.iat[j, i] = coherence_score

        return coherence_scores
