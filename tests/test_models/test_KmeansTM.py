import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from stream_topic.models.KmeansTM import KmeansTM
from stream_topic.utils.dataset import TMDataset


class TestKmeansTM(unittest.TestCase):
    def setUp(self):
        # Mock the TMDataset with initial embeddings of shape (25, 128)
        self.mock_dataset = MagicMock(spec=TMDataset)
        self.mock_dataset.get_embeddings.return_value = np.random.rand(25, 128)
        self.mock_dataset.dataframe = pd.DataFrame(
            {"text": ["sample text"] * 25})

        # Initialize the KmeansTM model
        self.model = KmeansTM(num_topics=10)

    def test_prepare_data(self):
        # Test data preparation
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.assertIsNotNone(self.model.embeddings)
        self.assertIsNotNone(self.model.dataframe)

    @patch("umap.umap_.UMAP")
    def test_dim_reduction(self, mock_umap):
        # Assuming self.model is an instance of KmeansTM and initialized somewhere in your test setup
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

    @patch("umap.umap_.UMAP.fit_transform")
    def test_train_model(self, mock_umap_fit_transform):
        mock_umap_fit_transform.return_value = np.random.rand(25, 15)

        output = self.model.train_model(self.mock_dataset)

        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
