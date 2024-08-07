import random
import string
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from stream_topic.data_utils import TMDataset
from stream_topic.models import SOMTM


class TestSomTM(unittest.TestCase):
    def setUp(self):
        self.n_topics = 5
        self.n_words_per_topic = 10
        self.n_documents = 150

        self.mock_dataset = MagicMock(spec=TMDataset)
        text_data = [
            " ".join(
                "".join(random.choices(
                    string.ascii_lowercase, k=random.randint(1, 15)))
                # Each document has 5-10 words
                for _ in range(random.randint(5, 10))
            )
            for _ in range(self.n_documents)  # 50 documents
        ]

        # Set vocabulary and corpus
        self.mock_dataset.get_vocabulary = lambda: list(
            set(word for text in text_data for word in text.split())
        )

        self.mock_dataset.get_embeddings.return_value = np.random.rand(
            self.n_documents, 384
        )

        self.mock_dataset.get_corpus = lambda: [
            text.split() for text in text_data]
        self.mock_dataset.dataframe = pd.DataFrame({"text": text_data})

        # Initialize the KmeansTM model
        self.model = SOMTM(m=self.n_topics, n=1, dim=384, n_iterations=5)

    def test_prepare_data(self):
        # Test data preparation
        self.model.dataset = self.mock_dataset
        self.model._prepare_data()
        self.assertIsNotNone(self.model.embeddings)
        self.assertIsNotNone(self.model.dataframe)

    def test_train_model(self):
        output = self.model.train_model(self.mock_dataset)

        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
