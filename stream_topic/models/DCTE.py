from datetime import datetime

import pandas as pd
import pyarrow as pa
from datasets import Dataset
from loguru import logger
from sentence_transformers.losses import CosineSimilarityLoss
from setfit import SetFitModel, TrainingArguments
from setfit import Trainer as SetfitTrainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from ..commons.check_steps import check_dataset_steps
from ..preprocessor._tf_idf import c_tf_idf, extract_tfidf_topics
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "DCTE"
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class DCTE(BaseModel):
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
        model: str = EMBEDDING_MODEL_NAME,
        **kwargs,
    ):
        """
        Initializes the DCTE model with specified number of topics, embedding model,
        SetFit model, batch size, number of iterations, and number of epochs.

        Parameters:
            model (str, optional): The identifier of the SetFit model to be used.
                Defaults to "all-MiniLM-L6-v2".

        """
        super().__init__(use_pretrained_embeddings=True, **kwargs)
        self.save_hyperparameters(
            ignore=[
                "embeddings_file_path",
                "embeddings_folder_path",
                "random_state",
                "save_embeddings",
            ]
        )
        self.n_topics = None

        self.model = SetFitModel.from_pretrained(f"sentence-transformers/{model}")
        self._status = TrainingStatus.NOT_STARTED
        self.n_topics = None
        self.embedding_model_name = self.hparams.get("embedding_model_name", model)

    def get_info(self):
        """
        Get information about the model.


        Returns
        -------
        dict
            Dictionary containing model information including model name,
            number of topics, embedding model name, UMAP arguments,
            K-Means arguments, and training status.
        """
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "embedding_model": self.embedding_model_name,
            "trained": self._status.name,
        }
        return info

    def _prepare_data(self, val_split: float):
        """
        Prepares the dataset for clustering.

        """

        assert hasattr(self, "train_dataset") and isinstance(
            self.train_dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.dataframe = self.train_dataset.dataframe

        self.dataframe.rename(columns={"labels": "label"}, inplace=True)

        print(
            "--- a train-validation split of 0.8 to 0.2 is performed --- \n---change 'val_split' if needed"
        )
        train_df, val_df = train_test_split(self.dataframe, test_size=val_split)

        # convert to Huggingface dataset
        self.train_ds = Dataset(pa.Table.from_pandas(train_df))
        self.val_ds = Dataset(pa.Table.from_pandas(val_df))

    def _get_topic_representation(self, predict_df: pd.DataFrame, top_words: int):
        docs_per_topic = predict_df.groupby(["predictions"], as_index=False).agg(
            {"text": " ".join}
        )
        tfidf, count = c_tf_idf(docs_per_topic["text"].values, m=len(predict_df))
        topic_dict = extract_tfidf_topics(
            tfidf,
            count,
            docs_per_topic,
            n=top_words,
        )

        one_hot_encoder = OneHotEncoder(sparse=False)
        predictions_one_hot = one_hot_encoder.fit_transform(predict_df[["predictions"]])

        beta = tfidf
        theta = predictions_one_hot

        return topic_dict, beta, theta

    def fit(
        self,
        dataset,
        val_split: float = 0.2,
        **training_args,
    ):
        """
        Trains the DCTE model using the given training dataset and then performs
        prediction and topic extraction on the specified prediction dataset.

        The method uses the SetFitTrainer for training and evaluates the model's performance.
        It then applies the trained model for prediction and extracts topics using TF-IDF.

        Parameters:
            dataset: The dataset used for training the model.
            val_split (float, optional): The fraction of the training data to use as
                validation data. Defaults to 0.2.
            top_words (int, optional): The number of top words to extract for each topic.
                Defaults to 10.

        Returns:
            dict: A dictionary containing the extracted topics and the topic-word matrix.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)

        # Set default training arguments
        default_args = {
            "batch_size": 6,
            "num_epochs": 10,
            "evaluation_strategy": "epoch",
            "save_strategy": "epoch",
            "num_iterations": 10,
            "load_best_model_at_end": True,
            "loss": CosineSimilarityLoss,
        }

        # Update default arguments with any user-provided arguments
        default_args.update(training_args)

        # Use the updated arguments
        args = TrainingArguments(**default_args)

        self.train_dataset = dataset
        self._status = TrainingStatus.INITIALIZED

        try:
            logger.info(f"--- Preparing {EMBEDDING_MODEL_NAME} Dataset ---")
            self._prepare_data(val_split=val_split)

            assert hasattr(self, "train_ds") and hasattr(
                self, "val_ds"
            ), "The training and Validation datasets have to be processed before training"

            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self.trainer = SetfitTrainer(
                model=self.model,
                args=args,
                train_dataset=self.train_ds,
                eval_dataset=self.val_ds,
            )

            # train
            self.trainer.train()
            # evaluate accuracy
            metrics = self.trainer.evaluate()

        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        return self

    def predict(self, dataset):
        """
        Predict topics for new documents.

        Parameters
        ----------
        dataset : TMDataset

        Returns
        -------
        list of int
            List of predicted topic labels.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        predict_df = pd.DataFrame({"tokens": dataset.get_corpus()})
        predict_df["text"] = [" ".join(words) for words in predict_df["tokens"]]

        labels = self.model(predict_df["text"])
        predict_df["predictions"] = labels

        return labels

    def get_topics(self, dataset, n_words=10):
        """
        Retrieve the top words for each topic.

        Parameters
        ----------
        n_words : int
            Number of top words to retrieve for each topic.

        Returns
        -------
        list of list of str
            List of topics with each topic represented as a list of top words.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        predict_df = pd.DataFrame({"tokens": dataset.get_corpus()})
        predict_df["text"] = [" ".join(words) for words in predict_df["tokens"]]

        labels = self.model(predict_df["text"])
        predict_df["predictions"] = labels

        topic_dict, beta, theta = self._get_topic_representation(predict_df, n_words)
        if self._status != TrainingStatus.SUCCEEDED:
            raise RuntimeError("Model has not been trained yet or failed.")
        return [[word for word, _ in topic_dict[key][:n_words]] for key in topic_dict]
