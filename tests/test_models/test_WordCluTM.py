import unittest

from stream_topic.models import WordCluTM
from stream_topic.utils import TMDataset


class TestWordCluTM(unittest.TestCase):
    def setUp(self):
        self.n_topics = 5
        self.n_words_per_topic = 10
        self.n_documents = 150
        self.dataset = TMDataset()
        self.dataset.fetch_dataset("BBC_News")

        # Initialize the KmeansTM model
        self.model = WordCluTM(num_topics=5)

    def test_word2vec(self):
        # Test data preparation
        self.model.dataset = self.dataset
        self.model.train_word2vec(self.dataset.get_corpus(), 2)
        self.assertIsNotNone(self.model.base_embedder)

    def test_train_model(self):
        output = self.model.train_model(self.dataset, word2vec_epochs=2)

        self.assertIn("topics", output)
        self.assertIn("topic-word-matrix", output)


if __name__ == "__main__":
    unittest.main()
