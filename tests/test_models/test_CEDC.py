import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import random
import string
from stream.models.CEDC import CEDC
from stream.data_utils.dataset import TMDataset


class TestCEDC(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        self.mock_dataset = MagicMock(spec=TMDataset)
        # Generate mock embeddings of size (25, 128) using numpy
        self.mock_dataset.get_embeddings.return_value = np.random.rand(50, 128)

        # Generate random words and create a dataframe
        random_texts = [
            "sample text " + "".join(random.choices(string.ascii_lowercase, k=250))
            for _ in range(50)
        ]
        self.mock_dataset.dataframe = pd.DataFrame({"text": random_texts})
        # Generate a mock vocabulary
        self.mock_dataset.get_vocabulary.return_value = list(
            set([word for word in random_texts])
        )

        # Initialize the CEDC model
        self.model = CEDC(num_topics=10)

    def test_prepare_data(self):
        # Test data preparation
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.assertIsNotNone(self.model.embeddings)
        self.assertIsNotNone(self.model.dataframe)

    @patch("umap.umap_.UMAP")
    def test_dim_reduction(self, mock_umap):
        # Test dimensionality reduction
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.model._dim_reduction()
        mock_umap.assert_called_once()
        self.assertIsNotNone(self.model.reduced_embeddings)

    def test_clustering(self):
        # Test clustering
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.model._dim_reduction()
        self.model._clustering()
        self.assertIsNotNone(self.model.labels)

    @patch("STREAM.utils.embedder.BaseEmbedder.create_word_embeddings")
    def test_train_model(self, mock_create_word_embeddings):
        # Mock embed_documents to return embeddings of shape (25, 384)
        mock_create_word_embeddings.return_value = np.random.rand(50, 128)

        # Run the training process
        output = self.model.train_model(self.mock_dataset)

        # Add assertions here to validate the output of the training process
        # For example:
        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
