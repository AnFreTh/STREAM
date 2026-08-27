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
from .abstract_helper_models.mixins import SentenceEncodingMixin
from .neural_base_models.fastopic_base import FAStopicBase
from ..commons.check_steps import check_dataset_steps

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "FASTopic"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


class FASTopic(BaseModel, SentenceEncodingMixin):
    """
    FASTopic: Fast Adaptive Sparse Topic Model.

    A neural topic model that uses optimal transport and adaptive sparse
    regularization for better topic quality and faster inference.

    Parameters
    ----------
    embedding_model_name : str, optional
        Name of the sentence embedding model, by default "all-MiniLM-L6-v2"
    embeddings_folder_path : str, optional
        Path to folder containing embeddings, by default None
    embeddings_file_path : str, optional
        Path to specific embeddings file, by default None
    save_embeddings : bool, optional
        Whether to save generated embeddings, by default False
    DT_alpha : float, optional
        Sinkhorn alpha between document and topic embeddings, by default 3.0
    TW_alpha : float, optional
        Sinkhorn alpha between topic and word embeddings, by default 2.0
    theta_temp : float, optional
        Temperature parameter for softmax in doc-topic distributions, by default 1.0
    normalize_embeddings : bool, optional
        Whether to normalize document embeddings, by default False
    batch_size : int, optional
        Batch size for training, by default 64
    val_size : float, optional
        Proportion of dataset for validation, by default 0.2
    shuffle : bool, optional
        Whether to shuffle training data, by default True
    random_state : int, optional
        Random seed for reproducibility, by default 42
    """

    def __init__(
        self,
        embedding_model_name: str = EMBEDDING_MODEL_NAME,
        embeddings_folder_path: str = None,
        embeddings_file_path: str = None,
        save_embeddings: bool = False,
        DT_alpha: float = 3.0,
        TW_alpha: float = 2.0,
        theta_temp: float = 1.0,
        normalize_embeddings: bool = False,
        batch_size: int = None,  # None = full batch (required for global OT)
        val_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        super().__init__(
            use_pretrained_embeddings=False,
            DT_alpha=DT_alpha,
            TW_alpha=TW_alpha,
            theta_temp=theta_temp,
            normalize_embeddings=normalize_embeddings,
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
            "embedding_model_name": self.embedding_model_name,
        }

        self.embeddings_prepared = False
        self.optimize = False

    def get_info(self):
        """Get information about the model."""
        info = {
            "model_name": MODEL_NAME,
            "num_topics": self.n_topics,
            "trained": self._status.name,
        }
        return info

    def _initialize_model(self):
        """Initialize the neural base model."""
        self.model = NeuralBaseModel(
            model_class=FAStopicBase,
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
        """Initialize the PyTorch Lightning trainer."""
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

    @staticmethod
    def _subsample_for_training(dataset, n_max, seed):
        """Return a shallow-copy dataset with a random row subset of size n_max.

        The 4 row-indexed fields (texts, labels, embeddings, bow) plus the
        underlying dataframe are sliced to the same indices, keeping them
        aligned. The original dataset object is NOT mutated (it's shared across
        all 16 models per run, and needed for end-of-fit theta extraction).
        """
        import copy
        import numpy as np

        rng = np.random.default_rng(int(seed))
        n = len(dataset.texts)
        idx = np.sort(rng.choice(n, size=n_max, replace=False))
        idx_list = idx.tolist()

        sub = copy.copy(dataset)  # shallow; slices below replace the sliced attrs
        sub.texts = [dataset.texts[i] for i in idx_list]
        sub.labels = [dataset.labels[i] for i in idx_list]
        if dataset.embeddings is not None:
            sub.embeddings = dataset.embeddings[idx]
        if dataset.bow is not None:
            sub.bow = dataset.bow[idx]
        if getattr(dataset, "dataframe", None) is not None:
            sub.dataframe = dataset.dataframe.iloc[idx_list].reset_index(drop=True)
        logger.info(
            f"FASTopic: subsampled {n_max} of {n} training docs (seed={seed}) "
            f"to keep global-OT training within GPU memory."
        )
        return sub

    def _initialize_datamodule(self, dataset):
        """Initialize the data module.

        FASTopic requires a single full batch: its Sinkhorn optimal transport is
        defined globally over all documents, so mini-batching would compute per-
        batch (not global) transport plans and break the method. To keep the
        method faithful on very large corpora that would otherwise OOM (dense
        N x V tensors on GPU), we cap the training set at ``MAX_TRAIN_DOCS`` via
        a seeded random subsample -- still a SINGLE global-OT batch, just over a
        representative subsample. The full corpus is retained on ``self.dataset``
        so end-of-fit theta extraction uses every document.
        """
        logger.info(f"--- Initializing Datamodule for {MODEL_NAME} ---")

        # Keep the full corpus for end-of-fit theta extraction; use a
        # (possibly-subsampled) TRAINING view only for the datamodule.
        self.dataset = dataset
        MAX_TRAIN_DOCS = 50_000
        if len(dataset.texts) > MAX_TRAIN_DOCS:
            train_dataset = self._subsample_for_training(
                dataset,
                MAX_TRAIN_DOCS,
                seed=self.hparams["datamodule_args"].get("random_state", 42),
            )
        else:
            train_dataset = dataset

        batch_size = self.hparams["datamodule_args"]["batch_size"]
        if batch_size is None:
            batch_size = len(train_dataset.texts)
            logger.info(
                f"FASTopic: full-batch training over {batch_size} documents "
                f"(global optimal transport). Full corpus size = "
                f"{len(dataset.texts)}."
            )
        self.data_module = TMDataModule(
            batch_size=batch_size,
            shuffle=self.hparams["datamodule_args"]["shuffle"],
            val_size=self.hparams["datamodule_args"]["val_size"],
            random_state=self.hparams["datamodule_args"]["random_state"],
        )

        self.data_module.preprocess_data(
            dataset=train_dataset,
            **{
                k: v
                for k, v in self.hparams["datamodule_args"].items()
                if k not in ["batch_size", "shuffle", "val_size"]
            },
        )

    def fit(
        self,
        dataset: TMDataset = None,
        n_topics: int = 20,
        val_size: float = 0.2,
        lr: float = None,
        lr_patience: int = 10,
        patience: int = 50,
        weight_decay: float = None,
        max_epochs: int = 1000,
        batch_size: int = None,  # None = full batch (required for global OT)
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
        Fit the FASTopic model to the given dataset.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train on
        n_topics : int, optional
            Number of topics, by default 20
        val_size : float, optional
            Validation set proportion, by default 0.2
        lr : float, optional
            Learning rate, by default 0.002
        lr_patience : int, optional
            Learning rate scheduler patience, by default 15
        patience : int, optional
            Early stopping patience, by default 15
        weight_decay : float, optional
            Weight decay for optimizer, by default 1e-07
        max_epochs : int, optional
            Maximum training epochs, by default 200
        batch_size : int, optional
            Batch size, by default 64
        shuffle : bool, optional
            Whether to shuffle data, by default True
        random_state : int, optional
            Random seed, by default 101
        checkpoint_path : str, optional
            Path to save checkpoints, by default "checkpoints"
        monitor : str, optional
            Metric to monitor, by default "val_loss"
        mode : str, optional
            Monitoring mode, by default "min"
        trial : optuna.Trial, optional
            Optuna trial for optimization, by default None
        optimize : bool, optional
            Whether optimizing hyperparameters, by default False
        **kwargs
            Additional trainer arguments
        """
        self.optimize = optimize
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics
        self.dataset = dataset

        # Resolve tuned hyperparameters: explicit arg wins, else hparams, else default.
        lr = lr if lr is not None else self.hparams.get("lr", 0.002)
        weight_decay = weight_decay if weight_decay is not None else self.hparams.get("weight_decay", 1e-07)
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

            if not self.embeddings_prepared:
                dataset, embeddings = self.prepare_embeddings(dataset, logger)
                self.embeddings_prepared = True

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

        data = {
            "embedding": torch.tensor(dataset.embeddings),
            "bow": torch.tensor(dataset.bow),
        }

        # Extract theta deterministically (eval mode: no dropout / batchnorm
        # batch-stats). Affects labels/NMI/Purity/Perplexity; beta unaffected.
        self.model.model.eval()

        self.theta = (
            self.model.model.get_theta(data, only_theta=True).detach().cpu().numpy()
        )
        self.theta = self.theta / self.theta.sum(axis=1, keepdims=True)

        self.beta = self.model.model.get_beta().detach().cpu().numpy()
        self.labels = np.array(np.argmax(self.theta, axis=1))

        self.topic_dict = self.get_topic_word_dict(self.data_module.vocab)

    def get_topic_word_dict(self, vocab, num_words=100):
        """Get the topic-word dictionary."""
        topic_word_dict = {}
        for topic_idx, topic_dist in enumerate(self.beta):
            top_word_indices = topic_dist.argsort()[-num_words:][::-1]
            top_words_probs = [(vocab[i], topic_dist[i]) for i in top_word_indices]
            topic_word_dict[topic_idx] = top_words_probs
        return topic_word_dict

    def predict(self, dataset):
        pass

    def suggest_hyperparameters(self, trial, max_topics=100):
        """Suggest hyperparameters for Optuna optimization."""
        self.hparams["DT_alpha"] = trial.suggest_float("DT_alpha", 1.0, 5.0)
        self.hparams["TW_alpha"] = trial.suggest_float("TW_alpha", 1.0, 5.0)
        self.hparams["theta_temp"] = trial.suggest_float("theta_temp", 0.5, 2.0)
        self.hparams["lr"] = trial.suggest_float("lr", 1e-4, 1e-2)
        self.hparams["weight_decay"] = trial.suggest_float("weight_decay", 1e-7, 1e-3)
        # FASTopic requires full-batch training for global optimal transport
        # Do NOT optimize batch_size

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
        """
        Optimize hyperparameters and fit the model.

        Parameters
        ----------
        dataset : TMDataset
            The dataset to train on
        min_topics : int, optional
            Minimum number of topics, by default 2
        max_topics : int, optional
            Maximum number of topics, by default 20
        criterion : str, optional
            Optimization criterion, by default "val_loss"
        n_trials : int, optional
            Number of optimization trials, by default 100
        custom_metric : object, optional
            Custom metric for optimization, by default None

        Returns
        -------
        dict
            Best parameters and optimal number of topics
        """
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
