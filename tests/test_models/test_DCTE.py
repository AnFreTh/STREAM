import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import random
import string
from STREAM.models.DCTE import DCTE
from STREAM.data_utils.dataset import TMDataset


class TestDCTE(unittest.TestCase):
    def setUp(self):
        self.mock_dataset = MagicMock(spec=TMDataset)
        # Prepare diverse labels and text data
        labels = ["A", "B", "C", "D", "E"]
        text_data = [
            " ".join(
                "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 15)))
                for _ in range(random.randint(5, 10))  # Each document has 5-10 words
            )
            for _ in range(50)  # 50 documents
        ]

        label_data = [labels[i % len(labels)] for i in range(50)]
        self.mock_dataset.dataframe = pd.DataFrame(
            {"text": text_data, "label_text": label_data}
        )

        # Set vocabulary and corpus
        self.mock_dataset.get_vocabulary = lambda: list(
            set(word for text in text_data for word in text.split())
        )
        self.mock_dataset.get_corpus = lambda: [text.split() for text in text_data]

        self.model = DCTE(num_topics=3, num_iterations=1, num_epochs=1)

    def test_prepare_data(self):
        # Test data preparation
        self.model.train_dataset = self.mock_dataset
        self.model._prepare_data(val_split=0.2)
        self.assertIsNotNone(self.model.train_ds)
        self.assertIsNotNone(self.model.val_ds)

    def test_train_model(self):
        # Perform actual training and topic extraction
        output = self.model.train_model(self.mock_dataset, self.mock_dataset)

        # Assertions to validate output
        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
