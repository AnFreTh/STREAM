from datetime import datetime

import lightning as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import (EarlyStopping, ModelCheckpoint,
                                         ModelSummary)
from loguru import logger

from ..commons.check_steps import check_dataset_steps
from ..utils.datamodule import TMDataModule
from ..utils.dataset import TMDataset
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.neural_basemodel import NeuralBaseModel
from .neural_base_models.prodlda_base import ProdLDABase

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "ProdLDA"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)


class ProdLDA(BaseModel):
    """
    Initialize the ProdLDA model.

    Parameters
    ----------
    **kwargs : dict
        Additional keyword arguments to pass to the parent class constructor.
    """

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(use_pretrained_embeddings=False, **kwargs)
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
        """
        Initialize the neural base model.

        Parameters
        ----------
        n_topics : int
            Number of topics.
        lr : float
            Learning rate.
        lr_patience : int
            Patience for learning rate scheduler.
        factor : float
            Factor for learning rate scheduler.
        weight_decay : float
            Weight decay for the optimizer.
        **model_kwargs : dict
            Additional keyword arguments for the model.
        """

        self.model = NeuralBaseModel(
            model_class=ProdLDABase,
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
        """
        Initialize the PyTorch Lightning trainer.

        Parameters
        ----------
        max_epochs : int
            Maximum number of epochs for training.
        monitor : str
            Metric to monitor for early stopping and checkpointing.
        patience : int
            Patience for early stopping.
        mode : str
            Mode for the monitored metric (min or max).
        checkpoint_path : str
            Path to save model checkpoints.
        **trainer_kwargs : dict
            Additional keyword arguments for the trainer.
        """

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
            callbacks=[
                early_stop_callback,
                checkpoint_callback,
                ModelSummary(max_depth=2),
            ],
            **trainer_kwargs,
        )

    def _initialize_datamodule(
        self, dataset, batch_size, shuffle, val_size, random_state, **kwargs
    ):
        """
        Initialize the data module.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to be used for training.
        batch_size : int
            Batch size for training.
        shuffle : bool
            Whether to shuffle the data.
        val_size : float
            Proportion of the dataset to use for validation.
        random_state : int
            Random seed for reproducibility.
        **kwargs : dict
            Additional keyword arguments for data preprocessing.
        """

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
        Fits the ProdLDA topic model to the given dataset.

        Args:
            dataset (TMDataset, optional): The dataset to train the topic model on. Defaults to None.
            n_topics (int, optional): The number of topics to extract. Defaults to 20.
            val_size (float, optional): The proportion of the dataset to use for validation. Defaults to 0.2.
            lr (float, optional): The learning rate for the optimizer. Defaults to 1e-04.
            lr_patience (int, optional): The number of epochs with no improvement after which the learning rate will be reduced. Defaults to 15.
            patience (int, optional): The number of epochs with no improvement after which training will be stopped. Defaults to 15.
            factor (float, optional): The factor by which the learning rate will be reduced. Defaults to 0.5.
            weight_decay (float, optional): The weight decay (L2 penalty) for the optimizer. Defaults to 1e-07.
            max_epochs (int, optional): The maximum number of epochs to train for. Defaults to 100.
            batch_size (int, optional): The batch size for training. Defaults to 32.
            shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
            random_state (int, optional): The random seed for reproducibility. Defaults to 101.
            checkpoint_path (str, optional): The path to save model checkpoints. Defaults to "checkpoints".
            monitor (str, optional): The metric to monitor for early stopping. Defaults to "val_loss".
            mode (str, optional): The mode for early stopping. Defaults to "min".
            **kwargs: Additional keyword arguments to be passed to the trainer.

        Raises:
            ValueError: If the dataset is not an instance of TMDataset.
        """

        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        check_dataset_steps(dataset, logger, MODEL_NAME)
        self.dataset = dataset

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

        data = {
            "bow": torch.tensor(dataset.bow),
        }

        self.theta = (
            self.model.model.get_theta(
                data, only_theta=True).detach().cpu().numpy()
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
            top_words_probs = [(vocab[i], topic_dist[i])
                               for i in top_word_indices]
            topic_word_dict[topic_idx] = top_words_probs
        return topic_word_dict

    def predict(self, dataset):
        pass
