import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from datetime import datetime
import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, ModelSummary
from optuna.integration import PyTorchLightningPruningCallback

from ..utils.dataset import TMDataset
from ..utils.datamodule import TMDataModule
from .abstract_helper_models.base import BaseModel, TrainingStatus
from .abstract_helper_models.neural_basemodel import NeuralBaseModel
from .neural_base_models.ecrtm_base import ECRTMBase
from ..commons.check_steps import check_dataset_steps

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "ECRTM"


class ECRTM(BaseModel):
    """
    ECRTM: Effective Neural Topic Modeling with Embedding Clustering Regularization.

    Reference: Xiaobao Wu et al. ICML 2023

    Parameters
    ----------
    encoder_dim : int, optional
        Encoder dimension, by default 200
    dropout : float, optional
        Dropout rate, by default 0.0
    embed_size : int, optional
        Embedding size, by default 200
    beta_temp : float, optional
        Temperature for beta softmax, by default 0.2
    weight_loss_ECR : float, optional
        Weight for ECR loss, by default 100.0
    sinkhorn_alpha : float, optional
        Sinkhorn alpha parameter, by default 20.0
    sinkhorn_max_iter : int, optional
        Max Sinkhorn iterations, by default 1000
    pretrained_WE : np.ndarray, optional
        Pretrained word embeddings, by default None
    batch_size : int, optional
        Batch size, by default 64
    val_size : float, optional
        Validation set proportion, by default 0.2
    shuffle : bool, optional
        Whether to shuffle data, by default True
    random_state : int, optional
        Random seed, by default 42
    """

    def __init__(
        self,
        encoder_dim: int = 200,
        dropout: float = 0.0,
        embed_size: int = 200,
        beta_temp: float = 0.2,
        weight_loss_ECR: float = 100.0,
        sinkhorn_alpha: float = 20.0,
        sinkhorn_max_iter: int = 1000,
        pretrained_WE=None,
        batch_size: int = 256,
        val_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            use_pretrained_embeddings=False,
            encoder_dim=encoder_dim,
            dropout=dropout,
            embed_size=embed_size,
            beta_temp=beta_temp,
            weight_loss_ECR=weight_loss_ECR,
            sinkhorn_alpha=sinkhorn_alpha,
            sinkhorn_max_iter=sinkhorn_max_iter,
            pretrained_WE=pretrained_WE,
        )
        self.save_hyperparameters(ignore=["random_state"])

        self.n_topics = None
        self._status = TrainingStatus.NOT_STARTED

        self.hparams["datamodule_args"] = {
            "batch_size": batch_size,
            "val_size": val_size,
            "shuffle": shuffle,
            "random_state": random_state,
            "embeddings": False,
            "bow": True,
            "tf_idf": False,
            "word_embeddings": False,
        }

        self.optimize = False

    def get_info(self):
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
        }
        return info

    def _initialize_model(self):
        self.model = NeuralBaseModel(
            model_class=ECRTMBase,
            dataset=self.dataset,
            **{
                k: v
                for k, v in self.hparams.items()
                if k not in ["datamodule_args", "max_epochs"]
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
        logger.info(f"--- Initializing Trainer for {MODEL_NAME} ---")
        early_stop_callback = EarlyStopping(
            monitor=monitor, min_delta=0.00, patience=patience, verbose=False, mode=mode
        )

        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            dirpath=checkpoint_path,
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

        self.trainer = pl.Trainer(
            max_epochs=max_epochs,
            callbacks=model_callbacks,
            **trainer_kwargs,
        )

    def _initialize_datamodule(self, dataset):
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

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        val_size: float = 0.2,
        lr: float = 2e-03,
        lr_patience: int = 10,
        patience: int = 50,
        weight_decay: float = 1e-07,
        max_epochs: int = 1000,
        batch_size: int = 256,
        shuffle: bool = True,
        random_state: int = 101,
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        trial=None,
        optimize=False,
        **kwargs,
    ):
        self.optimize = optimize
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics
        self.dataset = dataset

        self.hparams.update(
            {
                "n_topics": n_topics,
                "lr": lr,
                "lr_patience": lr_patience,
                "patience": patience,
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

            # Load best checkpoint weights
            if hasattr(self.trainer, "checkpoint_callback") and self.trainer.checkpoint_callback and self.trainer.checkpoint_callback.best_model_path:
                import torch as _torch
                _ckpt = _torch.load(self.trainer.checkpoint_callback.best_model_path, weights_only=True)
                self.model.load_state_dict(_ckpt["state_dict"])
                logger.info(f"Loaded best checkpoint from epoch {self.trainer.checkpoint_callback.best_model_score}")

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
        topic_word_dict = {}
        for topic_idx, topic_dist in enumerate(self.beta):
            top_word_indices = topic_dist.argsort()[-num_words:][::-1]
            top_words_probs = [(vocab[i], topic_dist[i]) for i in top_word_indices]
            topic_word_dict[topic_idx] = top_words_probs
        return topic_word_dict

    def predict(self, dataset):
        pass

    def suggest_hyperparameters(self, trial, max_topics=100):
        self.hparams["encoder_dim"] = trial.suggest_int("encoder_dim", 100, 500)
        self.hparams["dropout"] = trial.suggest_float("dropout", 0.0, 0.5)
        self.hparams["beta_temp"] = trial.suggest_float("beta_temp", 0.1, 1.0)
        self.hparams["weight_loss_ECR"] = trial.suggest_float(
            "weight_loss_ECR", 10.0, 200.0
        )
        self.hparams["sinkhorn_alpha"] = trial.suggest_float(
            "sinkhorn_alpha", 10.0, 50.0
        )
        self.hparams["lr"] = trial.suggest_float("lr", 1e-5, 1e-2)
        self.hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3)
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
        timeout=None,
    ):
        best_params = super().optimize_hyperparameters_neural(
            dataset=dataset,
            min_topics=min_topics,
            max_topics=max_topics,
            criterion=criterion,
            n_trials=n_trials,
            custom_metric=custom_metric,
            timeout=timeout,
        )
        return best_params
