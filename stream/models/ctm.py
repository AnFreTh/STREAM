from .neural_base_models.ctm_base import CTMBase
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
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
import torch.nn as nn
from optuna.integration import PyTorchLightningPruningCallback

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "CTM"
# logger.add(f"{MODEL_NAME}_{time}.log", backtrace=True, diagnose=True)
EMBEDDING_MODEL_NAME = "paraphrase-MiniLM-L3-v2"


class CTM(BaseModel):
    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        encoder_dim=128,
        dropout=0.1,
        inference_type="combined",
        inference_activation=nn.Softplus(),
        model_type="ProdLDA",
        rescale_loss=False,
        rescale_factor=1e-2,
        batch_size=64,
        val_size=0.2,
        shuffle=True,
        random_state=42,
    ):
        """
        Initialize the ETM model.

        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments to pass to the parent class constructor.
        """

        super().__init__(
            use_pretrained_embeddings=False,
            dropout=dropout,
            inference_type=inference_type,
            encoder_dim=encoder_dim,
            inference_activation=inference_activation,
            model_type=model_type,
            rescale_loss=rescale_loss,
            rescale_factor=rescale_factor,
        )
        self.save_hyperparameters(
            ignore=[
                "embeddings_file_path",
                "embeddings_folder_path",
                "random_state",
                "save_embeddings",
            ]
        )

        self.embedding_model_name = self.hparams.get(
            "embedding_model_name", embedding_model_name
        )

        self.embeddings_path = embeddings_folder_path
        self.embeddings_file_path = embeddings_file_path
        self.save_embeddings = save_embeddings
        self.n_topics = None

        self._status = TrainingStatus.NOT_STARTED

        self.hparams["datamodule_args"] = {
            "batch_size": batch_size,
            "val_size": val_size,
            "shuffle": shuffle,
            "random_state": random_state,
            "embeddings": True,
            "bow": True,
            "tf_idf": False,
            "word_embeddings": False,
        }

        self.embeddings_prepared = False
        self.optimize = False

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
        self,
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
            model_class=CTMBase,
            dataset=self.dataset,
            **{
                k: v
                for k, v in self.hparams.items()
                if k not in ["datamodule_args", "max_epochs", "factor"]
            },
        )

    def _initialize_trainer(
        self,
        max_epochs,
        monitor,
        patience,
        mode,
        checkpoint_path,
        trial=None,
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

        model_callbacks = [
            early_stop_callback,
            checkpoint_callback,
            ModelSummary(max_depth=2),
        ]

        if self.optimize:
            model_callbacks.append(
                PyTorchLightningPruningCallback(trial, monitor="val_loss")
            )

        # Initialize the trainer
        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=model_callbacks,
            **trainer_kwargs,
        )

    def _initialize_datamodule(
        self,
        dataset,
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

        logger.info(f"--- Initializing Datamodule for {MODEL_NAME} ---")
        self.data_module = TMDataModule(
            batch_size=self.hparams["datamodule_args"]["batch_size"],
            shuffle=self.hparams["datamodule_args"]["shuffle"],
            val_size=self.hparams["datamodule_args"]["val_size"],
            random_state=self.hparams["datamodule_args"]["random_state"],
        )

        self.data_module.preprocess_data(
            dataset=dataset,
            **{
                k: v
                for k, v in self.hparams["datamodule_args"].items()
                if k not in ["batch_size", "shuffle", "val_size"]
            },
        )

        self.dataset = dataset

    def _prepare_embeddings(self, dataset):
        """
        Prepares the dataset for clustering.

        Parameters
        ----------
        dataset : Dataset
            The dataset to be used for clustering.
        """

        if dataset.has_embeddings(self.embedding_model_name):
            logger.info(
                f"--- Loading precomputed {EMBEDDING_MODEL_NAME} embeddings ---"
            )
            self.embeddings = dataset.get_embeddings(
                self.embedding_model_name,
                self.embeddings_path,
                self.embeddings_file_path,
            )
            self.dataframe = dataset.dataframe
        else:
            logger.info(f"--- Creating {EMBEDDING_MODEL_NAME} document embeddings ---")
            self.embeddings = self.encode_documents(
                dataset.texts, encoder_model=self.embedding_model_name, use_average=True
            )
            if self.save_embeddings:
                dataset.save_embeddings(
                    self.embeddings,
                    self.embedding_model_name,
                    self.embeddings_path,
                    self.embeddings_file_path,
                )

        self.embeddings_prepared = True

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
        trial=None,
        optimize=False,
        **kwargs,
    ):
        """
        Fits the CTM topic model to the given dataset.

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
        self.optimize = optimize
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."

        self.n_topics = n_topics

        self.hparams.update(
            {
                "n_topics": n_topics,
                "lr": lr,
                "lr_patience": lr_patience,
                "patience": patience,
                "factor": factor,
                "weight_decay": weight_decay,
                "max_epochs": max_epochs,
            }
        )

        self.hparams["datamodule_args"].update(
            {
                "batch_size": batch_size,
                "val_size": val_size,
                "shuffle": shuffle,
                "random_state": random_state,
            }
        )

        try:
            self._status = TrainingStatus.RUNNING
            if not self.embeddings_prepared:
                self.prepare_embeddings(dataset, logger)

            self._status = TrainingStatus.INITIALIZED
            self._initialize_datamodule(dataset=dataset)

            self._initialize_model()

            self._initialize_trainer(
                max_epochs=self.hparams["max_epochs"],
                monitor=monitor,
                patience=patience,
                mode=mode,
                checkpoint_path=checkpoint_path,
                trial=trial,
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
            "embedding": torch.tensor(dataset.embeddings),
            "bow": torch.tensor(dataset.bow),
        }

        self.theta = (
            self.model.model.get_theta(data, only_theta=True).detach().cpu().numpy()
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

    def suggest_hyperparameters(self, trial, max_topics=100):
        self.hparams["n_topics"] = trial.suggest_int("n_topics", 1, max_topics)
        self.hparams["encoder_dim"] = trial.suggest_int("encoder_dim", 16, 512)
        self.hparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        self.hparams["inference_type"] = trial.suggest_categorical(
            "inference_type", ["combined", "zeroshot"]
        )
        self.hparams["inference_activation"] = trial.suggest_categorical(
            "inference_activation", ["Softplus", "ReLU", "LeakyReLU", "Tanh"]
        )
        self.hparams["model_type"] = trial.suggest_categorical(
            "model_type", ["ProdLDA", "LDA"]
        )

        # Map string to actual PyTorch activation function
        activation_mapping = {
            "Softplus": nn.Softplus(),
            "ReLU": nn.ReLU(),
            "LeakyReLU": nn.LeakyReLU(),
            "Tanh": nn.Tanh(),
        }
        self.hparams["inference_activation"] = activation_mapping[
            self.hparams["inference_activation"]
        ]

        self.hparams["datamodule_args"]["batch_size"] = trial.suggest_int(
            "batch_size", 12, 512
        )

    def optimize_and_fit(
        self,
        dataset,
        min_topics=2,
        max_topics=20,
        criterion="val_loss",
        n_trials=100,
        custom_metric=None,
    ):
        """
        A new method in the child class that calls the parent class's optimize_hyperparameters method.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train the model on.
        min_topics : int, optional
            Minimum number of topics to evaluate, by default 2.
        max_topics : int, optional
            Maximum number of topics to evaluate, by default 20.
        criterion : str, optional
            Criterion to use for optimization ('aic', 'bic', or 'custom'), by default 'aic'.
        n_trials : int, optional
            Number of trials for optimization, by default 100.
        custom_metric : object, optional
            Custom metric object with a `score` method for evaluation, by default None.

        Returns
        -------
        dict
            Dictionary containing the best parameters and the optimal number of topics.
        """
        best_params = super().optimize_hyperparameters_neural(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
        )

        return best_params
