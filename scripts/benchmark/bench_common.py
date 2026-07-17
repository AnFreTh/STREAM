"""Shared benchmark infrastructure for TopicArena."""

import io
import os
import time
import json
import traceback
import numpy as np
import pandas as pd
import boto3
from loguru import logger

from stream_topic.utils import TMDataset

print("[BENCH] importing LDA...", flush=True)
from stream_topic.models import LDA

print("[BENCH] importing NMFTM...", flush=True)
from stream_topic.models import NMFTM

print("[BENCH] importing KmeansTM...", flush=True)
from stream_topic.models import KmeansTM

print("[BENCH] importing BERTopicTM...", flush=True)
from stream_topic.models import BERTopicTM

print("[BENCH] importing ETM...", flush=True)
from stream_topic.models import ETM

print("[BENCH] importing ProdLDA...", flush=True)
from stream_topic.models import ProdLDA

print("[BENCH] importing NeuralLDA...", flush=True)
from stream_topic.models import NeuralLDA

print("[BENCH] importing CTM...", flush=True)
from stream_topic.models import CTM

print("[BENCH] importing CTMNeg...", flush=True)
from stream_topic.models import CTMNeg

print("[BENCH] importing NSTM...", flush=True)
from stream_topic.models import NSTM

print("[BENCH] importing FASTopic...", flush=True)
from stream_topic.models import FASTopic

print("[BENCH] importing ECRTM...", flush=True)
from stream_topic.models import ECRTM

print("[BENCH] importing SawETM...", flush=True)
from stream_topic.models import SawETM

print("[BENCH] importing HyperMiner...", flush=True)
from stream_topic.models import HyperMiner

print("[BENCH] importing TNTM...", flush=True)
from stream_topic.models import TNTM

print("[BENCH] importing KmeansTM_PCA...", flush=True)
from stream_topic.models.kmeans_pca import KmeansTM_PCA

print("[BENCH] ALL MODEL IMPORTS DONE", flush=True)
from stream_topic.metrics import (
    CV,
    NPMI,
    TopicDiversity,
    ISIM,
    INT,
    ISH,
    Embedding_Coherence,
    Embedding_Topic_Diversity,
)

print("[BENCH] ALL METRIC IMPORTS DONE", flush=True)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
# S3 bucket/prefix and AWS settings come from the central config (env vars with
# public defaults). See scripts/benchmark/config.py and .env.example.
from config import (  # noqa: E402
    S3_BUCKET,
    S3_PREFIX,
    STORAGE_BACKEND,
    LOCAL_RESULTS_DIR,
    boto_session,
)

MODELING_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EVAL_EMBEDDING_MODEL = "all-mpnet-base-v2"

BENCHMARK_PREPROCESS = {
    "min_df": 25,
    "max_df": 0.7,
    "lemmatize": True,
    "stem": False,
    "remove_stopwords": True,
    "remove_punctuation": True,
    "remove_numbers": True,
    "remove_accents": True,
    "remove_words_with_numbers": True,
    "remove_words_with_special_chars": True,
}

SEEDS = [42, 84, 126, 168, 210]

DATASETS = [
    "BBC_News",
    "20Newsgroups",
    "Poliblogs",
    "UN",
    "WHO",
    "NeurIPS",
    "ACL",
    "Reuters",
    "NYT",
    "Spotify",
    "Reddit_GME",
    "IMDB",
    "AG_News",
    "Arxiv",
    "WikiText",
    "PubMed",
    "DBpedia",
    "Yahoo_Answers",
]

DATASET_TOPICS = {
    "BBC_News": 10,
    "20Newsgroups": 20,
    "Poliblogs": 20,
    "UN": 15,
    "WHO": 15,
    "NeurIPS": 25,
    "ACL": 50,
    "Reuters": 50,
    "NYT": 50,
    "Spotify": 25,
    "Reddit_GME": 25,
    "IMDB": 20,
    "AG_News": 20,
    "Arxiv": 15,
    "WikiText": 25,
    "PubMed": 25,
    "DBpedia": 14,
    "Yahoo_Answers": 10,
}

# Models that need sentence embeddings for training
EMBEDDING_MODELS = {
    "KmeansTM",
    "KmeansTM_PCA",
    "BERTopicTM",
    "CTM",
    "CTMNeg",
    "TNTM",
    "FASTopic",
}

# Models that use AIC/BIC for HPO (non-neural)
AIC_HPO_MODELS = {"LDA", "NMFTM", "KmeansTM", "KmeansTM_PCA", "BERTopicTM"}

# Hierarchical models that take n_topics_list instead of n_topics
HIERARCHICAL_MODELS = {"SawETM", "HyperMiner"}

ALL_MODELS = [
    ("LDA", LDA),
    ("NMFTM", NMFTM),
    ("KmeansTM", KmeansTM),
    ("KmeansTM_PCA", KmeansTM_PCA),
    ("BERTopicTM", BERTopicTM),
    ("ETM", ETM),
    ("ProdLDA", ProdLDA),
    ("NeuralLDA", NeuralLDA),
    ("CTM", CTM),
    ("CTMNeg", CTMNeg),
    ("NSTM", NSTM),
    ("FASTopic", FASTopic),
    ("ECRTM", ECRTM),
    ("SawETM", SawETM),
    ("HyperMiner", HyperMiner),
    ("TNTM", TNTM),
]


# ---------------------------------------------------------------------------
# Storage backend (local filesystem or S3)
# ---------------------------------------------------------------------------
# Results are addressed by a "key" (e.g. "TopicArena/BBC_News/v1/LDA/x.json").
# The S3 backend uses the key as the object key; the local backend maps the key
# to a path under config.LOCAL_RESULTS_DIR. The runners are backend-agnostic —
# they only call put_csv / put_json / obj_exists / key().


class _S3Backend:
    """Persist results to S3. The boto client is created lazily so a local run
    never requires AWS credentials just by importing this module."""

    def __init__(self):
        self._client = None

    @property
    def client(self):
        if self._client is None:
            self._client = boto_session().client("s3")
        return self._client

    def put_csv(self, df, key):
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        self.client.put_object(Bucket=S3_BUCKET, Key=key, Body=buf.getvalue())
        logger.info(f"S3 PUT s3://{S3_BUCKET}/{key}")

    def put_json(self, obj, key):
        self.client.put_object(
            Bucket=S3_BUCKET,
            Key=key,
            Body=json.dumps(obj, indent=2, default=str),
        )
        logger.info(f"S3 PUT s3://{S3_BUCKET}/{key}")

    def exists(self, key):
        try:
            self.client.head_object(Bucket=S3_BUCKET, Key=key)
            return True
        except Exception:
            return False


class _LocalBackend:
    """Persist results to the local filesystem under config.LOCAL_RESULTS_DIR."""

    def __init__(self, root):
        self.root = root

    def _path(self, key):
        return os.path.join(self.root, key)

    def put_csv(self, df, key):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"WROTE {path}")

    def put_json(self, obj, key):
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(obj, f, indent=2, default=str)
        logger.info(f"WROTE {path}")

    def exists(self, key):
        return os.path.exists(self._path(key))


if STORAGE_BACKEND == "s3":
    _STORAGE = _S3Backend()
    logger.info(f"Storage backend: s3 (s3://{S3_BUCKET}/{S3_PREFIX})")
else:
    _STORAGE = _LocalBackend(LOCAL_RESULTS_DIR)
    logger.info(f"Storage backend: local ({LOCAL_RESULTS_DIR})")


def put_csv(df, key):
    """Write a DataFrame as CSV to the configured backend at `key`."""
    _STORAGE.put_csv(df, key)


def put_json(obj, key):
    """Write an object as JSON to the configured backend at `key`."""
    _STORAGE.put_json(obj, key)


def obj_exists(key):
    """Return True if `key` already exists in the configured backend."""
    return _STORAGE.exists(key)


def result_key(dataset, version, model, filename):
    return f"{S3_PREFIX}/{dataset}/{version}/{model}/{filename}"


# Backwards-compatible aliases (the harness historically used s3_* names).
s3_put_csv = put_csv
s3_put_json = put_json
s3_key = result_key


# ---------------------------------------------------------------------------
# Dataset loading with pre-embedded documents
# ---------------------------------------------------------------------------
def load_and_preprocess(dataset_name):
    """Load dataset, preprocess, and compute BOW."""
    ds = TMDataset()
    ds.fetch_dataset(dataset_name)
    ds.preprocess(**BENCHMARK_PREPROCESS)
    ds.get_bow()  # trigger BOW computation with min_df/max_df
    return ds


def precompute_embeddings(dataset, embedding_model_name=MODELING_EMBEDDING_MODEL):
    """Compute and cache document embeddings on the dataset object."""
    from sentence_transformers import SentenceTransformer

    if dataset.embeddings is not None:
        return

    logger.info(
        f"Computing {embedding_model_name} document embeddings for {dataset.name}..."
    )
    model = SentenceTransformer(embedding_model_name)
    dataset.embeddings = model.encode(
        dataset.texts, show_progress_bar=True, convert_to_numpy=True
    )
    logger.info(f"Embeddings shape: {dataset.embeddings.shape}")


# ---------------------------------------------------------------------------
# Model instantiation
# ---------------------------------------------------------------------------
def make_model(model_name, model_cls, dataset, hparams_override=None):
    """Instantiate a model with the right embedding config and optional hparam overrides."""
    if model_name == "TNTM":
        model = model_cls(
            embedding_model_name=MODELING_EMBEDDING_MODEL,
            word_embedding_model_name=MODELING_EMBEDDING_MODEL,
            save_embeddings=False,
            save_word_embeddings=False,
        )
    elif model_name in EMBEDDING_MODELS:
        model = model_cls(
            embedding_model_name=MODELING_EMBEDDING_MODEL,
            save_embeddings=False,
        )
    else:
        model = model_cls()

    # Apply HPO'd hyperparameters if provided
    if hparams_override is not None:
        for k, v in hparams_override.items():
            model.hparams[k] = v

    return model


def fit_model(model, model_name, dataset, n_topics, seed=42):
    """Fit a model, handling hierarchical models and seeding."""
    import random, torch
    import lightning as pl

    # Full reproducibility
    pl.seed_everything(seed, workers=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if model_name in HIERARCHICAL_MODELS:
        # Use HPO'd n_topics_list if available in hparams, otherwise compute default
        if (
            "n_topics_list" in model.hparams
            and model.hparams["n_topics_list"] is not None
        ):
            n_topics_list = model.hparams["n_topics_list"]
        else:
            n_topics_list = [
                n_topics,
                max(int(n_topics * 0.6), 5),
                max(int(n_topics * 0.3), 3),
            ]
        model.fit(dataset, n_topics_list=n_topics_list)
    else:
        model.fit(dataset, n_topics=n_topics)


# ---------------------------------------------------------------------------
# Metric evaluation
# ---------------------------------------------------------------------------
EVAL_TOP_K = [5, 10, 15, 20]

# Datasets with meaningful labels for NMI/Purity
LABELED_DATASETS = {
    "BBC_News",
    "20Newsgroups",
    "Poliblogs",
    "AG_News",
    "Arxiv",
    "DBpedia",
    "Yahoo_Answers",
    "IMDB",
    "Spotify",
}


def _compute_clustering_metrics(model, model_name, dataset):
    """Compute NMI, Purity, and Perplexity from model outputs."""
    from sklearn.metrics import normalized_mutual_info_score

    results = {}

    # --- Perplexity (model-agnostic via beta and theta) ---
    try:
        beta = model.get_beta()
        if hasattr(beta, "detach"):
            beta = beta.detach().cpu().numpy()
        beta = np.array(beta, dtype=np.float64)
        n_topics_actual = beta.shape[0]
        # Auto-transpose if needed
        if hasattr(model, "topic_dict") and len(model.topic_dict) != n_topics_actual:
            if beta.shape[1] == len(model.topic_dict):
                beta = beta.T

        theta = model.get_theta()
        if hasattr(theta, "detach"):
            theta = theta.detach().cpu().numpy()
        theta = np.array(theta, dtype=np.float64)

        bow = dataset.bow
        if bow is None:
            bow, _ = dataset.get_bow()
        bow = np.array(bow, dtype=np.float64)

        # Normalize beta to get p(w|t)
        beta_norm = beta / (beta.sum(axis=1, keepdims=True) + 1e-30)
        # Normalize theta to get p(t|d)
        theta_norm = theta / (theta.sum(axis=1, keepdims=True) + 1e-30)

        # p(w|d) = theta @ beta
        recon = theta_norm @ beta_norm
        recon = np.clip(recon, 1e-30, None)

        # Perplexity = exp(-1/N * sum(bow * log(recon)))
        total_words = bow.sum()
        if total_words > 0:
            log_likelihood = (bow * np.log(recon)).sum()
            perplexity = float(np.exp(-log_likelihood / total_words))
            results["Perplexity"] = round(perplexity, 3)
        else:
            results["Perplexity"] = None
    except Exception:
        results["Perplexity"] = None

    # --- NMI and Purity ---
    if dataset.name not in LABELED_DATASETS:
        results["NMI"] = None
        results["Purity"] = None
        return results

    labels = dataset.labels
    if labels is None or all(l is None for l in labels):
        results["NMI"] = None
        results["Purity"] = None
        return results

    try:
        theta = model.get_theta()
        if hasattr(theta, "detach"):
            theta = theta.detach().cpu().numpy()
        theta = np.array(theta)
        pred = np.argmax(theta, axis=1)
    except Exception:
        results["NMI"] = None
        results["Purity"] = None
        return results

    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    try:
        true = le.fit_transform(labels)
    except Exception:
        results["NMI"] = None
        results["Purity"] = None
        return results

    if len(true) != len(pred):
        results["NMI"] = None
        results["Purity"] = None
        return results

    nmi = float(np.around(normalized_mutual_info_score(true, pred), 5))
    results["NMI"] = nmi

    # Purity
    total = len(true)
    purity = 0.0
    for cluster_id in np.unique(pred):
        mask = pred == cluster_id
        if mask.sum() == 0:
            continue
        counts = np.bincount(true[mask])
        purity += counts.max()
    results["Purity"] = float(np.around(purity / total, 5))

    return results


def evaluate_model(model, model_name, topics, dataset, raw_dataset, eval_embedder=None):
    """Compute all metrics at multiple top-k cutoffs. Keys: '{metric}@{k}'."""
    from sentence_transformers import SentenceTransformer

    if eval_embedder is None:
        eval_embedder = SentenceTransformer(EVAL_EMBEDDING_MODEL)

    results = {}
    eval_times = {}

    # Get beta (handle transposed shapes)
    try:
        beta = model.get_beta()
        if hasattr(beta, "detach"):
            beta = beta.detach().cpu().numpy()
        beta = np.array(beta)
        n_topics_actual = len(topics)
        if beta.shape[0] != n_topics_actual and beta.shape[1] == n_topics_actual:
            beta = beta.T
    except Exception:
        beta = None

    def _eval(name, fn):
        t0 = time.time()
        try:
            results[name] = fn()
        except Exception as e:
            results[name] = None
            logger.warning(f"{name} failed: {e}")
        eval_times[f"eval_time_{name}"] = round(time.time() - t0, 3)

    # Pre-embed all topic words once and build shared TopwordEmbeddings cache
    from stream_topic.metrics.TopwordEmbeddings import TopwordEmbeddings

    shared_tw = TopwordEmbeddings(
        word_embedding_model=eval_embedder, create_new_file=False
    )
    shared_tw.embed_topwords(topics, n_topwords_to_use=min(20, len(topics[0])))

    # Build metric instances once with shared embedder (reuses embedding_dict)
    isim = ISIM(n_words=20, metric_embedder=eval_embedder)
    isim.topword_embeddings = shared_tw
    intm = INT(n_words=20, metric_embedder=eval_embedder)
    intm.topword_embeddings = shared_tw
    ish = ISH(n_words=20, metric_embedder=eval_embedder)
    ish.topword_embeddings = shared_tw
    emb_coh = Embedding_Coherence(n_words=20, metric_embedder=eval_embedder)
    emb_coh.topword_embeddings = shared_tw
    emb_td = Embedding_Topic_Diversity(n_words=20, metric_embedder=eval_embedder)
    emb_td.topword_embeddings = shared_tw

    for k in EVAL_TOP_K:
        topics_k = [t[:k] for t in topics]

        _eval(
            f"CV@{k}", lambda _t=topics_k, _k=k: CV(raw_dataset, n_words=_k).score(_t)
        )
        _eval(
            f"CV_train@{k}", lambda _t=topics_k, _k=k: CV(dataset, n_words=_k).score(_t)
        )
        _eval(f"NPMI@{k}", lambda _t=topics_k: NPMI(raw_dataset).score(_t))
        _eval(f"TD@{k}", lambda _t=topics_k, _k=k: TopicDiversity(n_words=_k).score(_t))

        # Embedding metrics: update n_words per k, reuse cached embeddings
        isim.n_words = k
        intm.n_words = k
        ish.n_words = k
        emb_coh.n_words = k
        emb_td.n_words = k

        _eval(f"ISIM@{k}", lambda _t=topics_k: isim.score(_t))
        _eval(f"INT@{k}", lambda _t=topics_k: intm.score(_t))
        _eval(f"ISH@{k}", lambda _t=topics_k: ish.score(_t))
        _eval(f"Emb_Coherence@{k}", lambda _t=topics_k: emb_coh.score(_t))

        if beta is not None:
            _eval(f"Emb_TD@{k}", lambda _t=topics_k, _b=beta: emb_td.score(_t, _b))
        else:
            results[f"Emb_TD@{k}"] = None
            eval_times[f"eval_time_Emb_TD@{k}"] = 0.0

    eval_times["eval_time_total"] = round(sum(eval_times.values()), 3)
    results.update(eval_times)
    return results


# ---------------------------------------------------------------------------
# Single run helper
# ---------------------------------------------------------------------------
def _get_model_stats(model, model_name):
    """Extract training stats: n_params, epochs trained, peak GPU memory."""
    import torch

    stats = {}

    # Number of parameters (neural models)
    if hasattr(model, "model") and hasattr(model.model, "parameters"):
        n_params = sum(p.numel() for p in model.model.parameters())
        n_trainable = sum(
            p.numel() for p in model.model.parameters() if p.requires_grad
        )
        stats["n_params"] = n_params
        stats["n_trainable_params"] = n_trainable
    else:
        stats["n_params"] = None
        stats["n_trainable_params"] = None

    # Epochs actually trained (Lightning models)
    if hasattr(model, "trainer") and model.trainer is not None:
        stats["epochs_trained"] = model.trainer.current_epoch + 1
    else:
        stats["epochs_trained"] = None

    # Peak GPU memory
    if torch.cuda.is_available():
        stats["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        torch.cuda.reset_peak_memory_stats()
    else:
        stats["peak_gpu_mb"] = None

    return stats


# Cache eval embedder across calls
_eval_embedder = None


def _get_eval_embedder():
    global _eval_embedder
    if _eval_embedder is None:
        from sentence_transformers import SentenceTransformer

        _eval_embedder = SentenceTransformer(EVAL_EMBEDDING_MODEL)
    return _eval_embedder


def run_single(
    dataset_name,
    model_name,
    model_cls,
    dataset,
    raw_dataset,
    n_topics,
    seed=42,
    version="default",
    hparams_override=None,
):
    """Train one model, evaluate, save to S3. Returns result dict or None."""
    logger.info(f"  [{dataset_name}] {model_name} seed={seed}")

    try:
        model = make_model(
            model_name, model_cls, dataset, hparams_override=hparams_override
        )
        start = time.time()
        fit_model(model, model_name, dataset, n_topics, seed=seed)
        train_time = time.time() - start

        topics = model.get_topics(n_words=20)
        model_stats = _get_model_stats(model, model_name)
        clustering_metrics = _compute_clustering_metrics(model, model_name, dataset)
        metrics = evaluate_model(
            model,
            model_name,
            topics,
            dataset,
            raw_dataset,
            eval_embedder=_get_eval_embedder(),
        )

        result = {
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "n_topics": n_topics,
            "n_topics_actual": len(topics),
            "train_time": round(train_time, 3),
            **model_stats,
            **clustering_metrics,
            **metrics,
        }

        # Save topics to S3
        topics_df = pd.DataFrame({f"topic_{i}": t for i, t in enumerate(topics)})
        s3_put_csv(
            topics_df,
            s3_key(dataset_name, version, model_name, f"topics_seed{seed}.csv"),
        )

        # Save metrics to S3
        s3_put_json(
            result,
            s3_key(dataset_name, version, model_name, f"metrics_seed{seed}.json"),
        )

        logger.info(
            f"    OK in {train_time:.1f}s (eval {metrics.get('eval_time_total', 0):.1f}s) "
            f"| CV@10={metrics.get('CV@10')} NPMI@10={metrics.get('NPMI@10')} TD@10={metrics.get('TD@10')}"
        )
        return result

    except Exception as e:
        logger.error(f"    FAILED: {e}")
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "model": model_name,
            "seed": seed,
            "n_topics": n_topics,
            "error": str(e),
        }


# ---------------------------------------------------------------------------
# HPO helper
# ---------------------------------------------------------------------------
def run_hpo(
    dataset_name,
    model_name,
    model_cls,
    dataset,
    raw_dataset,
    n_topics,
    version,
    hpo_timeout=3600,
    criterion="val_loss",
    custom_metric=None,
):
    """Run HPO for one model, save study history + best params to S3.

    Returns (best_hparams_dict, hpo_result_dict) or (None, None) on failure.
    """
    logger.info(
        f"  HPO: {model_name} on {dataset_name} "
        f"(budget={hpo_timeout}s, K={n_topics}, criterion={criterion})"
    )

    try:
        model = make_model(model_name, model_cls, dataset)

        if not hasattr(model, "optimize_and_fit"):
            logger.warning(f"    {model_name} has no HPO support, skipping")
            return None, None

        start = time.time()

        if model_name in AIC_HPO_MODELS:
            # Classical models: use their native criterion (aic/recon) or custom
            if criterion == "custom" and custom_metric is not None:
                # LDA uses 'metric' kwarg; others use 'custom_metric'
                if model_name == "LDA":
                    model.optimize_and_fit(
                        dataset,
                        metric=custom_metric,
                        min_topics=n_topics,
                        max_topics=n_topics,
                        criterion="custom",
                        n_trials=1000,
                        timeout=hpo_timeout,
                    )
                else:
                    model.optimize_and_fit(
                        dataset,
                        min_topics=n_topics,
                        max_topics=n_topics,
                        criterion="custom",
                        custom_metric=custom_metric,
                        n_trials=1000,
                        timeout=hpo_timeout,
                    )
            else:
                # Each model uses its own default criterion:
                # LDA -> aic (log-likelihood), NMF -> recon, KmeansTM/BERTopicTM/CEDC -> aic
                model.optimize_and_fit(
                    dataset,
                    min_topics=n_topics,
                    max_topics=n_topics,
                    n_trials=1000,
                    timeout=hpo_timeout,
                )
        else:
            # Neural models: val_loss or custom
            model.optimize_and_fit(
                dataset,
                min_topics=n_topics,
                max_topics=n_topics,
                criterion=criterion,
                custom_metric=custom_metric,
                n_trials=1000,
                timeout=hpo_timeout,
            )

        hpo_time = time.time() - start
        best_hparams = model.get_hyperparameters()

        hpo_result = {
            "dataset": dataset_name,
            "model": model_name,
            "criterion": criterion,
            "hpo_time": round(hpo_time, 1),
            "best_hparams": best_hparams,
        }

        # Save best hparams to S3
        s3_put_json(
            hpo_result,
            s3_key(dataset_name, version, model_name, "hpo_best_params.json"),
        )

        logger.info(f"    HPO done in {hpo_time:.0f}s")
        return best_hparams, hpo_result

    except Exception as e:
        logger.error(f"    HPO FAILED: {e}")
        traceback.print_exc()
        return None, None
