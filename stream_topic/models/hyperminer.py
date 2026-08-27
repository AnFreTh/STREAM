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
from .neural_base_models.hyperminer_base import HyperMinerBase
from ..commons.check_steps import check_dataset_steps

time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
MODEL_NAME = "HyperMiner"


class HyperMiner(BaseModel):
    """
    HyperMiner: Topic Taxonomy Mining with Hyperbolic Embedding.

    Hierarchical topic model using hyperbolic geometry for better hierarchy representation.

    Reference: Yishi Xu et al. NeurIPS 2022

    Parameters
    ----------
    n_topics_list : list, optional
        Number of topics per layer (top to bottom), by default [50, 36, 12]
    embed_size : int, optional
        Embedding size, by default 50
    hidden_size : int, optional
        Hidden layer size, by default 300
    pretrained_WE : np.ndarray, optional
        Pretrained word embeddings, by default None
    curvature : float, optional
        Hyperbolic curvature, by default -0.01
    clip_r : float, optional
        Clipping radius for embeddings, by default None
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
        n_topics_list=None,
        embed_size: int = 50,
        hidden_size: int = 300,
        pretrained_WE=None,
        curvature: float = -1.0,
        clip_r: float = None,
        batch_size: int = 256,
        val_size: float = 0.2,
        shuffle: bool = True,
        random_state: int = 42,
        **kwargs,
    ):
        # n_topics_list=None: net is built lazily in fit(); the benchmark computes
        # a per-dataset [K, 0.6K, 0.3K] hierarchy. A fixed default would shadow it
        # and train every dataset at K=50.
        # curvature=-1.0 (was -0.01): -0.01 is near-Euclidean and largely disables
        # the Poincare-ball geometry that is HyperMiner's whole contribution. -1.0
        # matches the paper's unit-curvature convention and the learnable init.
        super().__init__(
            use_pretrained_embeddings=False,
            n_topics_list=n_topics_list,
            embed_size=embed_size,
            hidden_size=hidden_size,
            pretrained_WE=pretrained_WE,
            curvature=curvature,
            clip_r=clip_r,
        )
        self.save_hyperparameters(ignore=["random_state"])

        self.n_topics = n_topics_list[0] if n_topics_list else None
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
            "n_topics_list": self.hparams.get("n_topics_list"),
            "curvature": self.hparams.get("curvature"),
            "trained": self._status.name,
        }
        return info

    def _initialize_model(self):
        # HPO registers helper keys 'n_layers' and 'n_topics_layer_i' as Optuna
        # trial params; these are converted into 'n_topics_list' and must NOT be
        # forwarded to HyperMinerBase, which does not accept them.
        _EXCLUDE = {"datamodule_args", "max_epochs", "n_layers", "batch_size"}
        self.model = NeuralBaseModel(
            model_class=HyperMinerBase,
            dataset=self.dataset,
            **{
                k: v
                for k, v in self.hparams.items()
                if k not in _EXCLUDE and not k.startswith("n_topics_layer_")
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
        n_topics_list=[50, 36, 12],
        n_topics=None,
        val_size: float = 0.2,
        lr: float = None,
        lr_patience: int = 10,
        patience: int = 50,
        weight_decay: float = None,
        max_epochs: int = 1000,
        batch_size: int = None,
        shuffle: bool = True,
        random_state: int = 101,
        checkpoint_path: str = "checkpoints",
        monitor: str = "val_loss",
        mode: str = "min",
        trial=None,
        optimize=False,
        **kwargs,
    ):
        # Hierarchical model. HPO tunes 'n_layers' and per-layer sizes
        # ('n_topics_layer_i'); reconstruct the tuned n_topics_list so the
        # post-HPO refit uses the BEST configuration, and so the scalar
        # n_topics never leaks into trainer kwargs.
        top = n_topics if n_topics is not None else self.hparams.get("n_topics")
        n_layers = self.hparams.get("n_layers")
        if top is not None and n_layers:
            rebuilt = [top]
            for i in range(1, n_layers):
                li = self.hparams.get(f"n_topics_layer_{i}")
                if li is None:
                    break
                rebuilt.append(li)
            n_topics_list = rebuilt
        elif top is not None and (not n_topics_list or n_topics_list[0] != top):
            n_topics_list = [top, max(3, top // 2), max(2, top // 4)]
        self.hparams["n_topics_list"] = n_topics_list
        self.optimize = optimize
        assert isinstance(
            dataset, TMDataset
        ), "The dataset must be an instance of TMDataset."
        check_dataset_steps(dataset, logger, MODEL_NAME)

        self.n_topics = n_topics_list[0]
        self.dataset = dataset

        # Resolve tuned hyperparameters: an explicitly passed value wins,
        # otherwise fall back to whatever is already in hparams (set by HPO
        # suggest / refit / eval override), and finally the canonical default.
        lr = lr if lr is not None else self.hparams.get("lr", 1e-02)
        weight_decay = weight_decay if weight_decay is not None else self.hparams.get("weight_decay", 1e-07)
        batch_size = batch_size if batch_size is not None else self.hparams.get("datamodule_args", {}).get("batch_size", 256)
        self.hparams.update(
            {
                "n_topics_list": n_topics_list,
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

        logger.info("--- Training completed successfully. ---")
        self._status = TrainingStatus.SUCCEEDED

        # Extract theta deterministically (eval mode: Weibull uses its mode, not
        # a sample). Affects labels/NMI/Purity/Perplexity; beta is unaffected.
        self.model.model.eval()

        # Get theta for top layer
        theta_list = self.model.model.get_theta(
            torch.tensor(self.dataset.bow), only_theta=True
        )
        self.theta = theta_list[0].detach().cpu().numpy()
        self.theta = self.theta / self.theta.sum(axis=1, keepdims=True)

        # Get beta for top layer
        beta_list = self.model.model.get_beta()
        self.beta = beta_list[0].detach().cpu().numpy()
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

    def suggest_hyperparameters(self, trial):
        # Top layer is fixed to n_topics (set by the benchmark)
        n_topics_top = self.hparams.get("n_topics", self.n_topics)
        n_layers = trial.suggest_int("n_layers", 2, 4)
        n_topics_list = [n_topics_top]
        for i in range(1, n_layers):
            prev = n_topics_list[-1]
            lower = max(3, prev // 3)
            upper = max(lower + 1, prev - 2)
            layer_size = trial.suggest_int(f"n_topics_layer_{i}", lower, upper)
            n_topics_list.append(layer_size)
            # Persist per-layer size so fit() rebuilds THIS trial's hierarchy.
            self.hparams[f"n_topics_layer_{i}"] = layer_size

        # Persist n_layers too, so fit()'s branch-1 reconstruction fires.
        self.hparams["n_layers"] = n_layers
        self.hparams["n_topics_list"] = n_topics_list
        self.hparams["hidden_size"] = trial.suggest_int("hidden_size", 128, 512)
        self.hparams["embed_size"] = trial.suggest_int("embed_size", 50, 200)
        self.hparams["curvature"] = trial.suggest_float("curvature", -1.0, -0.001)
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
