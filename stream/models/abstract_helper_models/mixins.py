from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"


class SentenceEncodingMixin:
    """
    Mixin class for models that require sentence encoding before fitting or transforming.
    """

    def encode_documents(
        self,
        documents: List[str],
        encoder_model: str = EMBEDDING_MODEL_NAME,
        use_average: bool = True,
    ) -> np.ndarray:
        """
        Encode a list of documents into embeddings.

        Parameters:
            documents (List[str]): List of documents to encode.
            encoder_model (str): Name or path of the sentence encoder model. Defaults to 'all-MiniLM-L6-v2'.
            use_average (bool): Whether to use average embeddings for long documents. Defaults to True.

        Returns:
            np.ndarray: Array of shape (n_documents, embedding_size) containing document embeddings.
        """
        encoder = SentenceTransformer(encoder_model)
        max_length = (
            encoder.max_seq_length
        )  # Extract maximum length from the encoder model
        embeddings = []
        for doc in tqdm(documents):
            if len(doc) > max_length and use_average:
                # Split document into segments of max_length and encode each segment
                segments = self.split_document(doc, max_length)
                segment_embeddings = [encoder.encode(seg) for seg in segments]
                # Compute average embedding across segments
                avg_embedding = np.mean(segment_embeddings, axis=0)
                embeddings.append(avg_embedding)
            else:
                # Encode the entire document
                embeddings.append(encoder.encode(doc))

        return np.array(embeddings)

    def split_document(self, document: str, max_length: int) -> List[str]:
        """
        Split a long document into segments of specified maximum length.

        Parameters:
            document (str): Document to split into segments.
            max_length (int): Maximum length of each segment.

        Returns:
            List[str]: List of document segments.
        """
        segments = []
        for start in range(0, len(document), max_length):
            segment = document[start: start + max_length]
            segments.append(segment)
        return segments
