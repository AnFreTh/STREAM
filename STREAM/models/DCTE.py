from setfit import SetFitModel, Trainer, TrainingArguments
import pyarrow as pa
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from datasets import Dataset
from sentence_transformers.losses import CosineSimilarityLoss
from octis.models.model import AbstractModel
from sklearn import preprocessing
from ..utils.tf_idf import c_tf_idf, extract_tfidf_topics
from sklearn.model_selection import train_test_split
from ..data_utils.dataset import TMDataset


class DCTE(AbstractModel):
    """
    A document classification and topic extraction class that utilizes the SetFitModel for
    document classification and TF-IDF for topic extraction.

    This class inherits from the AbstractModel class and is designed for supervised
    document classification and unsupervised topic modeling.

    Attributes:
        n_topics (int): The number of topics to identify in the dataset.
        model (SetFitModel): The SetFitModel used for document classification.
        batch_size (int): The batch size used in training.
        num_iterations (int): The number of iterations for SetFit training.
        num_epochs (int): The number of epochs for SetFit training.
    """

    def __init__(
        self,
        num_topics: int = 20,
        model: str = "all-MiniLM-L6-v2",
        batch_size: int = 8,
        num_iterations: int = 20,
        num_epochs: int = 10,
    ):
        """
        Initializes the DCTE model with specified number of topics, embedding model,
        SetFit model, batch size, number of iterations, and number of epochs.

        Parameters:
            num_topics (int, optional): The number of topics to identify in the dataset.
                Defaults to 20.
            model (str, optional): The identifier of the SetFit model to be used.
                Defaults to "all-MiniLM-L6-v2".
            batch_size (int, optional): The batch size to use during training. Defaults to 8.
            num_iterations (int, optional): The number of iterations to run SetFit training.
                Defaults to 20.
            num_epochs (int, optional): The number of epochs to train SetFit model.
                Defaults to 10.
        """
        self.n_topics = num_topics
        self.trained = False
        self.model = SetFitModel.from_pretrained(f"sentence-transformers/{model}")
        self.trained = False
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.num_epochs = num_epochs

    def _prepare_data(self, val_split: float):
        """
        Prepares the dataset for clustering.

        """

        assert hasattr(self, "train_dataset") and isinstance(
            self.train_dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.train_dataset.get_dataframe()
        self.dataframe = self.train_dataset.dataframe

        self.dataframe["label"] = preprocessing.LabelEncoder().fit_transform(
            self.dataframe["label_text"]
        )

        print(
            "--- a train-validation split of 0.8 to 0.2 is performed --- \n---change 'val_split' if needed"
        )
        train_df, val_df = train_test_split(self.dataframe, test_size=val_split)

        ### convert to Huggingface dataset
        self.train_ds = Dataset(pa.Table.from_pandas(train_df))
        self.val_ds = Dataset(pa.Table.from_pandas(val_df))

    def _get_topic_document_matrix(self):
        assert (
            self.trained
        ), "Model must be trained before accessing the topic-document matrix."
        # Safely get the topic-document matrix with a default value of None if not found
        return self.output.get("topic-document-matrix", None)

    def _get_topics(self, predict_df: pd.DataFrame, top_words: int):
        docs_per_topic = predict_df.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(predict_df))
        topics = extract_tfidf_topics(
            tfidf,
            count,
            docs_per_topic,
            n=top_words,
        )

        new_topics = {}
        words_list = []
        for k in predict_df["predictions"].unique():
            words = [
                word
                for t in topics[k][0:top_words]
                for word in t
                if isinstance(word, str)
            ]
            weights = [
                weight
                for t in topics[k][0:top_words]
                for weight in t
                if isinstance(weight, float)
            ]
            weights = [weight / sum(weights) for weight in weights]
            new_topics[k] = list(zip(words, weights))
            words_list.append(words)

        res_dic = {}
        res_dic["topics"] = words_list
        res_dic["topic-word-matrix"] = tfidf.T
        res_dic["topic_dict"] = topics
        one_hot_encoder = OneHotEncoder(
            sparse=False
        )  # Use sparse=False to get a dense array
        predictions_one_hot = one_hot_encoder.fit_transform(predict_df[["predictions"]])

        # Transpose the one-hot encoded matrix to get shape (k, n)
        topic_document_matrix = predictions_one_hot.T
        res_dic["topic-document-matrix"] = topic_document_matrix

        return res_dic

    def train_model(
        self,
        train_dataset,
        predict_dataset,
        val_split: float = 0.2,
        n_top_words: int = 10,
        **training_args,
    ):
        """
        Trains the DCTE model using the given training dataset and then performs
        prediction and topic extraction on the specified prediction dataset.

        The method uses the SetFitTrainer for training and evaluates the model's performance.
        It then applies the trained model for prediction and extracts topics using TF-IDF.

        Parameters:
            train_dataset: The dataset used for training the model.
            predict_dataset: The dataset on which to perform prediction and topic extraction.
            val_split (float, optional): The fraction of the training data to use as
                validation data. Defaults to 0.2.
            top_words (int, optional): The number of top words to extract for each topic.
                Defaults to 10.

        Returns:
            dict: A dictionary containing the extracted topics and the topic-word matrix.
        """

        # Set default training arguments
        default_args = {
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "num_iterations": self.num_iterations,
            "load_best_model_at_end": True,
            "loss": CosineSimilarityLoss,
        }

        # Update default arguments with any user-provided arguments
        default_args.update(training_args)

        # Use the updated arguments
        args = TrainingArguments(**default_args)

        self.train_dataset = train_dataset
        self._prepare_data(val_split=val_split)

        assert hasattr(self, "train_ds") and hasattr(
            self, "val_ds"
        ), "The training and Validation datasets have to be processed before training"

        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
        )

        # train
        self.trainer.train()
        # evaluate accuracy
        metrics = self.trainer.evaluate()

        print("--- finished training ---")
        print(metrics)

        predict_df = pd.DataFrame({"tokens": predict_dataset.get_corpus()})
        predict_df["text"] = [" ".join(words) for words in predict_df["tokens"]]

        self.labels = self.model(predict_df["text"])
        predict_df["predictions"] = self.labels

        self.output = self._get_topics(predict_df, n_top_words)
        self.trained = True

        return self.output
