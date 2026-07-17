"""
SageMaker entry point for TopicArena HPO benchmark (v5_hpo_cv_5seed).

Receives datasets to process via hyperparameters. Runs HPO with CV coherence
objective for each (dataset, model) combo, then evaluates with 5 seeds.
"""

import os
import sys
import argparse

# Make the shipped benchmark config importable (bucket/prefix/region from env).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark")
)
from config import S3_BUCKET, S3_PREFIX, boto_session  # noqa: E402

# Download models from S3 and set cache
MODELS_DIR = "/tmp/st_cache_proper"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import tarfile
import boto3

if not os.path.exists(
    os.path.join(MODELS_DIR, "models--sentence-transformers--all-MiniLM-L6-v2")
):
    print("[ENTRY] downloading models from S3...", flush=True)
    s3 = boto_session().client("s3")
    tar_path = "/tmp/st_models.tar.gz"
    s3.download_file(
        S3_BUCKET, f"{S3_PREFIX}/_models/st_models.tar.gz", tar_path
    )
    print("[ENTRY] extracting...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("/tmp")
    os.remove(tar_path)
    print(f"[ENTRY] models ready at {MODELS_DIR}", flush=True)
else:
    print("[ENTRY] models already cached", flush=True)


# Download dataset files from S3
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stream_topic", "stream_topic_data"
)
if not os.path.exists(os.path.join(DATA_DIR, "preprocessed_datasets", "BBC_News")):
    print("[ENTRY] downloading datasets from S3...", flush=True)
    s3 = boto_session().client("s3")
    tar_path = "/tmp/st_data.tar.gz"
    s3.download_file(
        S3_BUCKET, f"{S3_PREFIX}/_models/st_data.tar.gz", tar_path
    )
    print("[ENTRY] extracting datasets...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_topic")
        )
    os.remove(tar_path)
    print("[ENTRY] datasets ready", flush=True)
else:
    print("[ENTRY] datasets already present", flush=True)

# stream_topic is shipped as source in the container
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from loguru import logger


def main():
    # Download NLTK resources before anything else
    import nltk

    for resource in [
        "stopwords",
        "wordnet",
        "punkt_tab",
        "brown",
        "averaged_perceptron_tagger",
    ]:
        nltk.download(resource, quiet=True)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets", required=True, help="Comma-separated dataset names"
    )
    parser.add_argument(
        "--models", default=None, help="Comma-separated model names (default: all)"
    )
    parser.add_argument("--version", default="v8_hpo_cv_5h")
    parser.add_argument("--hpo_timeout", type=int, default=5 * 60 * 60)
    parser.add_argument("--worker_id", default="0")
    args = parser.parse_args()

    from stream_topic.utils import TMDataset
    from stream_topic.metrics import CV

    # bench_common is shipped at scripts/benchmark/ in the source dir
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark"
        ),
    )
    from bench_common import (
        DATASET_TOPICS,
        ALL_MODELS,
        SEEDS,
        load_and_preprocess,
        precompute_embeddings,
        run_single,
        run_hpo,
        s3_put_csv,
        obj_exists,
        S3_PREFIX,
        S3_BUCKET,
    )

    SKIP_HPO = {
        ("PubMed", "TNTM"),
        ("AG_News", "TNTM"),
        ("AG_News", "BERTopicTM"),
        ("AG_News", "NSTM"),
        ("ACL", "TNTM"),
        ("DBpedia", "TNTM"),
        ("WikiText", "TNTM"),
        ("NYT", "TNTM"),
        ("Reuters", "TNTM"),
        ("ACL", "BERTopicTM"),
        ("WikiText", "BERTopicTM"),
        ("AG_News", "FASTopic"),
        ("Yahoo_Answers", "TNTM"),
        ("IMDB", "BERTopicTM"),
        ("NeurIPS", "BERTopicTM"),
        ("DBpedia", "NSTM"),
        ("Spotify", "TNTM"),
        ("NeurIPS", "TNTM"),
        ("Poliblogs", "TNTM"),
        ("Yahoo_Answers", "BERTopicTM"),
        ("Arxiv", "TNTM"),
        ("NeurIPS", "ECRTM"),
        ("PubMed", "BERTopicTM"),
        ("Yahoo_Answers", "NSTM"),
        ("20Newsgroups", "TNTM"),
        ("UN", "BERTopicTM"),
        ("DBpedia", "BERTopicTM"),
        ("AG_News", "CTM"),
        ("Yahoo_Answers", "FASTopic"),
        ("DBpedia", "FASTopic"),
        ("AG_News", "SawETM"),
        ("WikiText", "FASTopic"),
    }

    datasets = args.datasets.split(",")
    logger.info(
        f"TopicArena HPO worker {args.worker_id}: {len(datasets)} datasets: {datasets}"
    )

    # Filter models if specified
    if args.models:
        model_filter = set(args.models.split(","))
        models_to_run = [
            (name, cls) for name, cls in ALL_MODELS if name in model_filter
        ]
        logger.info(
            f"  Running {len(models_to_run)} models: {[m for m,_ in models_to_run]}"
        )
    else:
        models_to_run = ALL_MODELS

    class CVMetricWrapper:
        def __init__(self, raw_dataset, n_words=10):
            self._cv = CV(raw_dataset, n_words=n_words)

        def score(self, topics):
            return self._cv.score(topics)

    all_results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        n_topics = DATASET_TOPICS[dataset_name]

        dataset = load_and_preprocess(dataset_name)
        precompute_embeddings(dataset)

        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)
        cv_metric = CVMetricWrapper(raw_dataset)

        for model_name, model_cls in models_to_run:
            # Skip if already completed (all 5 seeds exist in storage)
            skip = True
            for seed in SEEDS:
                key = f"{S3_PREFIX}/{dataset_name}/{args.version}/{model_name}/metrics_seed{seed}.json"
                if not obj_exists(key):
                    skip = False
                    break
            if skip:
                logger.info(f"  SKIP {model_name} (already complete)")
                continue

            # Skip HPO for slow combos (default train > 30min)
            if (dataset_name, model_name) in SKIP_HPO:
                logger.info(
                    f"  Skipping HPO for {model_name} (too slow), using defaults"
                )
                best_hparams, hpo_result = None, None
            else:
                best_hparams, hpo_result = run_hpo(
                    dataset_name,
                    model_name,
                    model_cls,
                    dataset,
                    raw_dataset,
                    n_topics,
                    version=args.version,
                    hpo_timeout=args.hpo_timeout,
                    criterion="custom",
                    custom_metric=cv_metric,
                )

            for seed in SEEDS:
                result = run_single(
                    dataset_name,
                    model_name,
                    model_cls,
                    dataset,
                    raw_dataset,
                    n_topics,
                    seed=seed,
                    version=args.version,
                    hparams_override=best_hparams,
                )
                if hpo_result is not None:
                    result["hpo_time"] = hpo_result["hpo_time"]
                    result["hpo_criterion"] = "cv_coherence"
                all_results.append(result)

            df = pd.DataFrame(all_results)
            s3_put_csv(
                df,
                f"{S3_PREFIX}/_results/{args.version}_all_worker{args.worker_id}.csv",
            )

    df = pd.DataFrame(all_results)
    s3_put_csv(
        df, f"{S3_PREFIX}/_results/{args.version}_all_worker{args.worker_id}.csv"
    )
    logger.info(f"Done! {len(all_results)} runs completed.")


if __name__ == "__main__":
    main()
