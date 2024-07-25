from .neural_base_models.etm_base import ETMBase
import numpy as np
from loguru import logger
from datetime import datetime
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.neural_basemodel import NeuralBaseModel
import lightning as pl
import pandas as pd
import torch
from ..utils.datamodule import TMDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint


time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "ETM"
logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class ETM(BaseModel):
    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(use_pretrained_embeddings=True, **kwargs)
        self.save_hyperparameters()

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
            "trained": self._status.name,
        }
        return info

    def _initialize_model(
        self, n_topics, lr, lr_patience, factor, weight_decay, **model_kwargs
    ):

        self.model = NeuralBaseModel(
            model_class=ETMBase,
            dataset=self.dataset,
            n_topics=n_topics,
            lr=lr,
            lr_patience=lr_patience,
            lr_factor=factor,
            weight_decay=weight_decay,
            **model_kwargs,
        )

    def _initialize_trainer(
        self,
        max_epochs,
        monitor,
        patience,
        mode,
        checkpoint_path,
        **trainer_kwargs,
    ):

        logger.info(f"--- Initializing Trainer for {MODEL_NAME} ---")
        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,  # Specify the directory to save checkpoints
            filename="best_model",
        )

        # Initialize the trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=[early_stop_callback, checkpoint_callback],
            **trainer_kwargs,
        )

    def _initialize_datamodule(
        self, dataset, batch_size, shuffle, val_size, random_state, **kwargs
    ):

        kwargs.setdefault("min_df", 3)

        logger.info(f"--- Initializing Datamodule for {MODEL_NAME} ---")
        self.data_module = TMDataModule(
            batch_size=batch_size,
            shuffle=shuffle,
            val_size=val_size,
            random_state=random_state,
        )

        self.data_module.preprocess_data(
            dataset=dataset,
            val=val_size,
            embeddings=False,
            bow=True,
            tf_idf=False,
            word_embeddings=False,
            random_state=random_state,
            **kwargs,
        )

        self.dataset = dataset

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        val_size: float = 0.2,
        lr: float = 1e-04,
        lr_patience: int = 15,
        patience: int = 15,
        factor: float = 0.5,
        weight_decay: float = 1e-07,
        max_epochs: int = 100,
        batch_size: int = 32,
        shuffle: bool = True,
        random_state: int = 101,
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        **kwargs,
    ):
        """
        Trains the K-Means topic model on the provided dataset.

        Parameters
        ----------
        dataset : Dataset
            The dataset to train the model on.
        n_topics : int, optional
            Number of topics to extract, by default 20

        Raises
        ------
        AssertionError
            If the dataset is not an instance of TMDataset.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.n_topics = n_topics

        try:

            self._status = TrainingStatus.INITIALIZED
            self._initialize_datamodule(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                val_size=val_size,
                random_state=random_state,
            )

            self._initialize_model(
                lr=lr,
                n_topics=n_topics,
                lr_patience=lr_patience,
                factor=factor,
                weight_decay=weight_decay,
            )

            self._initialize_trainer(
                max_epochs=max_epochs,
                monitor=monitor,
                patience=patience,
                mode=mode,
                checkpoint_path=checkpoint_path,
                **kwargs,
            )

            logger.info(f"--- Training {MODEL_NAME} topic model ---")
            self._status = TrainingStatus.RUNNING
            self.trainer.fit(self.model, self.data_module)

        except Exception as e:
            logger.error(f"Error in training: {e}")
            self._status = TrainingStatus.FAILED
            raise
        except KeyboardInterrupt:
            logger.error("Training interrupted.")
            self._status = TrainingStatus.INTERRUPTED
            raise

        if self.n_topics <= 0:
            raise ValueError("Number of topics must be greater than 0.")

        self._status = TrainingStatus.INITIALIZED
        try:
            pass

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

        self.theta = (
            self.model.model.get_theta(torch.tensor(self.dataset.bow), only_theta=True)
            .detach()
            .cpu()
            .numpy()
        )

        self.theta = self.theta / self.theta.sum(axis=1, keepdims=True)

        self.beta = self.model.model.get_beta().detach().cpu().numpy()
        self.labels = np.array(np.argmax(self.theta, axis=1))

        self.topic_dict = self.get_topic_word_dict(self.data_module.vocab)

    def get_topic_word_dict(self, vocab, num_words=100):
        """
        Get the topic-word dictionary.

        Parameters
        ----------
        vocab : list of str
            Vocabulary list corresponding to the word indices.
        num_words : int, optional
            Number of top words to retrieve for each topic, by default 100.

        Returns
        -------
        dict
            Dictionary where keys are topic indices and values are lists of tuples (word, probability).
        """

        topic_word_dict = {}
        for topic_idx, topic_dist in enumerate(self.beta):
            top_word_indices = topic_dist.argsort()[-num_words:][::-1]
            top_words_probs = [(vocab[i], topic_dist[i]) for i in top_word_indices]
            topic_word_dict[topic_idx] = top_words_probs
        return topic_word_dict

    def predict(self, dataset):
        pass
