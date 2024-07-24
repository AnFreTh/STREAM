import re
from collections.abc import Iterable
from typing import List

import numpy as np
import pandas as pd
from gensim.models.keyedvectors import Word2VecKeyedVectors
from tqdm import tqdm


class GensimBackend:
    """
    Gensim Embedding Model

    This class provides functionality to create document embeddings using Gensim Word2Vec embeddings.

    Args:
        embedding_model (Word2VecKeyedVectors): A Gensim Word2Vec model for word embeddings.

    Attributes:
        embedding_model (Word2VecKeyedVectors): The Gensim Word2Vec model used for embeddings.

    Methods:
        encode(documents: List[str], verbose: bool = False) -> np.ndarray:
            Embed a list of documents/words into a matrix of embeddings.

    """

    def __init__(self, embedding_model: Word2VecKeyedVectors):
        """
        Initialize the GensimBackend with a Word2VecKeyedVectors model.

        Args:
            embedding_model (Word2VecKeyedVectors): A Gensim Word2Vec model for word embeddings.

        Raises:
            ValueError: If the provided model is not a Word2VecKeyedVectors instance.

        """
        super().__init__()

        if isinstance(embedding_model, Word2VecKeyedVectors):
            self.embedding_model = embedding_model
        else:
            raise ValueError(
                "Please select a correct Gensim model: \n"
                "`import gensim.downloader as api` \n"
                "`ft = api.load('fasttext-wiki-news-subwords-300')`"
            )

    def encode(self, documents: List[str], verbose: bool = False) -> np.ndarray:
        """
        Embed a list of documents/words into an n-dimensional matrix of embeddings.

        Args:
            documents (List[str]): A list of documents or words to be embedded.
            verbose (bool, optional): Controls the verbosity of the process.

        Returns:
            np.ndarray: Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`.

        """
        # unused variables
        # vector_shape = self.embedding_model.get_vector(
        #     list(self.embedding_model.index_to_key)[0]
        # ).shape[0]
        # empty_vector = np.zeros(vector_shape)

        embeddings = []
        for doc in tqdm(documents, disable=not verbose, position=0, leave=True):
            doc_embedding = []

            # Extract word embeddings
            for word in doc.split(" "):
                try:
                    word_embedding = self.embedding_model.get_vector(word)
                    doc_embedding.append(word_embedding)
                except KeyError:
                    continue

            # Pool word embeddings
            doc_embedding = np.mean(doc_embedding, axis=0)
            embeddings.append(doc_embedding)

        embeddings = np.array(embeddings, dtype=object)
        return embeddings


class BaseEmbedder:
    """
    Base Embedder Class

    This class provides a base for creating document and word embeddings using different models.

    Args:
        embedding_model: The embedding model used for generating embeddings.

    Attributes:
        embedder: The specific backend embedder used for generating embeddings.
        embedding_model: The embedding model used for generating embeddings.

    Methods:
        _check_documents_type(documents: List[str]): Check if the provided documents are of the correct type.
        _clean_docs(documents: List[str]): Clean and preprocess a list of documents.
        create_doc_embeddings(documents: List[str], progress: bool = False): Create document embeddings.
        create_word_embeddings(word: List[str]): Create word embeddings.

    """

    def __init__(self, embedding_model):
        """
        Initialize the BaseEmbedder with an embedding model.

        Args:
            embedding_model: The embedding model used for generating embeddings.

        """
        if isinstance(embedding_model, Word2VecKeyedVectors):
            self.embedder = GensimBackend(embedding_model)
        else:
            self.embedder = embedding_model
        self.embedding_model = embedding_model

    def _check_documents_type(self, documents):
        """
        Check if the provided documents are of the correct type.

        Args:
            documents: The documents to check.

        Raises:
            TypeError: If the provided documents are not of the correct type.

        """
        if isinstance(documents, Iterable) and not isinstance(documents, str):
            if not any([isinstance(doc, str) for doc in documents]):
                raise TypeError(
                    "Make sure that the iterable only contains strings.")

        else:
            raise TypeError(
                "Make sure that the documents variable is an iterable containing strings only."
            )

    def _clean_docs(self, documents: List[str]):
        """
        Clean and preprocess a list of documents.

        Args:
            documents (List[str]): List of documents to clean.

        Returns:
            pd.DataFrame: A DataFrame with cleaned and lowercased documents.

        """

        documents = pd.DataFrame(
            {"docs": documents, "ID": range(len(documents)), "Topic": None}
        )

        for i in range(len(documents)):
            documents["docs"].loc[i] = re.compile(r"[/(){}\[\]\|@,;]").sub(
                "", documents["docs"][i]
            )
            documents["docs"].loc[i] = re.compile(
                r"\\").sub("", documents["docs"][i])
            documents["docs"].loc[i] = re.compile(
                "'").sub("", documents["docs"][i])
            documents["docs"].loc[i] = re.compile(
                "  ").sub(" ", documents["docs"][i])
        documents["docs"] = documents["docs"].str.lower()

        return documents

    def _clean_docs_(self, text):  #
        text = re.compile(r"[/(){}\[\]\|@,;]").sub("", text)
        text = re.compile(r"\\").sub("", text)
        text = re.compile("'").sub("", text)
        text = re.compile("  ").sub(" ", text)

        return text

    def create_doc_embeddings(self, documents: List[str], progress=False):
        """
        Create document embeddings for a list of documents.

        Args:
            documents (List[str]): List of documents to create embeddings for.
            progress (bool, optional): Controls the verbosity of the process.

        Returns:
            np.ndarray: Document embeddings.
            pd.DataFrame: A DataFrame with cleaned and lowercased documents.

        """

        self._check_documents_type(documents)

        documents = pd.DataFrame(
            {"docs": documents, "ID": range(len(documents)), "Topic": None}
        )

        documents["docs"] = documents["docs"].apply(self._clean_docs_)
        documents["docs"] = documents["docs"].str.lower()

        self.corpus_embeddings = self.embedder.encode(documents["docs"])

        return self.corpus_embeddings, documents

    def create_word_embeddings(self, word: List[str]):
        """
        Create word embeddings for a list of words.

        Args:
            word (List[str]): List of words to create embeddings for.

        Returns:
            np.ndarray: Word embeddings.

        """
        try:
            self.word_embedding = self.embedder.encode(
                word,
            )
            return self.word_embedding

        except KeyError:
            pass
