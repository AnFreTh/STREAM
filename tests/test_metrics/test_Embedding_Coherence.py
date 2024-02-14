import unittest
import numpy as np
import random
import string
from unittest.mock import patch, MagicMock

from stream.metrics.coherence_metrics import Embedding_Coherence
from stream.data_utils.dataset import TMDataset


class TestEmbeddingCoherence(unittest.TestCase):
    def setUp(self):
        # Mock data
        self.n_topics = 10
        self.n_words_per_topic = 10
        self.n_documents = 50

        self.mock_dataset = MagicMock(spec=TMDataset)
        text_data = [
            " ".join(
                "".join(random.choices(string.ascii_lowercase, k=random.randint(1, 15)))
                for _ in range(random.randint(5, 10))  # Each document has 5-10 words
            )
            for _ in range(self.n_documents)  # 50 documents
        ]

        # Set vocabulary and corpus
        self.mock_dataset.get_vocabulary = lambda: list(
            set(word for text in text_data for word in text.split())
        )

        self.mock_dataset.get_corpus = lambda: [text.split() for text in text_data]

        unique_words = list(set(word for text in text_data for word in text.split()))
        random.shuffle(unique_words)

        # Creating mock model output with different topics
        self.mock_model_output = {
            "topics": [
                unique_words[
                    i * self.n_words_per_topic : (i + 1) * self.n_words_per_topic
                ]
                for i in range(self.n_topics)
            ],
            "topic-word-matrix": np.random.rand(self.n_topics, len(unique_words)),
        }

        # Create an instance of Embedding_Topic_Diversity
        self.metric = Embedding_Coherence(dataset=self.mock_dataset)

    def test_score(self):
        # Test the score method
        score = self.metric.score(self.mock_model_output)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_score_per_topic(self):
        # Test the score_per_topic method
        scores_per_topic = self.metric.score_per_topic(self.mock_model_output)
        self.assertEqual(len(scores_per_topic.keys()), self.n_topics)
        self.assertIsInstance(scores_per_topic, dict)
        self.assertIsInstance(list(scores_per_topic.values())[0], float)
        for score in list(scores_per_topic.values()):
            self.assertGreaterEqual(score, -1)
            self.assertLessEqual(score, 1)


if __name__ == "__main__":
    unittest.main()
