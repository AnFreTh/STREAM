"""
SageMaker entry point for the TopicArena DEFAULT benchmark (v2_default_5seed).

Mirrors entry_point.py but with NO hyperparameter optimization: each
(dataset, model) combo is evaluated with the model's default hyperparameters
across 5 random seeds. This is the main results run for the paper.

Receives datasets (and optionally a model subset) via hyperparameters.
"""

import os
import sys
import argparse

# Make the shipped benchmark config importable (bucket/prefix/region from env).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark")
)
from config import S3_BUCKET, S3_PREFIX, boto_session  # noqa: E402

# Model / HF cache
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
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/_models/st_models.tar.gz", tar_path)
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
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/_models/st_data.tar.gz", tar_path)
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
    # NLTK resources
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
    parser.add_argument("--version", default="v2_default_5seed")
    parser.add_argument("--worker_id", default="0")
    parser.add_argument(
        "--fresh",
        type=int,
        default=0,
        help="If 1, ignore existing S3 results and recompute every combo "
        "(use for a corrected rerun; default 0 = skip already-complete combos).",
    )
    args = parser.parse_args()

    from stream_topic.utils import TMDataset

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
        s3_put_csv,
        obj_exists,
        S3_PREFIX,
        S3_BUCKET,
    )

    fresh = bool(args.fresh)
    datasets = args.datasets.split(",")
    logger.info(
        f"TopicArena DEFAULT worker {args.worker_id}: {len(datasets)} datasets: {datasets} "
        f"(fresh={fresh})"
    )

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

    all_results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        n_topics = DATASET_TOPICS[dataset_name]

        dataset = load_and_preprocess(dataset_name)
        precompute_embeddings(dataset)

        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)

        for model_name, model_cls in models_to_run:
            # Skip only when NOT a fresh rerun and all 5 seeds already exist.
            if not fresh:
                done = all(
                    obj_exists(
                        f"{S3_PREFIX}/{dataset_name}/{args.version}/{model_name}/metrics_seed{seed}.json"
                    )
                    for seed in SEEDS
                )
                if done:
                    logger.info(f"  SKIP {model_name} (already complete)")
                    continue

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
                )
                all_results.append(result)

            # Incremental save after each model
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
