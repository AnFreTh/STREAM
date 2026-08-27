"""
SageMaker entry point for the TopicArena EMBEDDING-SWAP ablation.

Same as entry_point_default.py (default hyperparameters, 5 seeds, no HPO) but the
document-modeling encoder is swapped for a bigger sentence-transformers model,
passed via the --embedding_model hyperparameter. The evaluation encoder
(all-mpnet-base-v2) is held fixed so embedding metrics keep a common reference.

Two differences from the default entry point:
  * HF_HUB_OFFLINE is NOT set, so SentenceTransformer can DOWNLOAD the requested
    encoder (the shipped tarball only contains all-MiniLM-L6-v2 + all-mpnet).
    The cached MiniLM/mpnet from the tarball are still used from cache.
  * STREAM_MODELING_EMBEDDING_MODEL is exported from --embedding_model BEFORE
    bench_common is imported, so its module global and precompute_embeddings'
    default arg both bind to the swapped encoder.
"""

import os
import sys
import argparse

# Make the shipped benchmark config importable (bucket/prefix/region from env).
sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark")
)
from config import S3_BUCKET, S3_PREFIX, boto_session  # noqa: E402

# Model / HF cache. NOTE: HF_HUB_OFFLINE is deliberately left unset here so the
# swapped encoder can be pulled from the Hub; the tarball-cached MiniLM + mpnet
# are still served from cache.
MODELS_DIR = "/tmp/st_cache_proper"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import tarfile
import boto3

if not os.path.exists(
    os.path.join(MODELS_DIR, "models--sentence-transformers--all-MiniLM-L6-v2")
):
    print("[ENTRY] downloading base models from S3...", flush=True)
    s3 = boto_session().client("s3")
    tar_path = "/tmp/st_models.tar.gz"
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/_models/st_models.tar.gz", tar_path)
    print("[ENTRY] extracting...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall("/tmp")
    os.remove(tar_path)
    print(f"[ENTRY] base models ready at {MODELS_DIR}", flush=True)
else:
    print("[ENTRY] base models already cached", flush=True)

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
        "--models",
        default="KmeansTM,BERTopicTM,CTM",
        help="Comma-separated model names (default: the 3 embedding-swap models).",
    )
    parser.add_argument("--version", required=True)
    parser.add_argument("--worker_id", default="0")
    parser.add_argument(
        "--embedding_model",
        required=True,
        help="sentence-transformers modeling encoder to swap in (e.g. "
        "mixedbread-ai/mxbai-embed-large-v1).",
    )
    parser.add_argument(
        "--fresh",
        type=int,
        default=0,
        help="If 1, ignore existing S3 results and recompute every combo.",
    )
    args = parser.parse_args()

    # CRITICAL: export the encoder BEFORE importing bench_common, whose module
    # global MODELING_EMBEDDING_MODEL and precompute_embeddings default arg bind
    # it at import time.
    os.environ["STREAM_MODELING_EMBEDDING_MODEL"] = args.embedding_model
    print(f"[ENTRY] modeling encoder: {args.embedding_model}", flush=True)

    from stream_topic.utils import TMDataset

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
        MODELING_EMBEDDING_MODEL,
        EVAL_EMBEDDING_MODEL,
        load_and_preprocess,
        precompute_embeddings,
        run_single,
        s3_put_csv,
        obj_exists,
        S3_PREFIX,
        S3_BUCKET,
    )

    assert MODELING_EMBEDDING_MODEL == args.embedding_model, (
        f"encoder override failed: bench_common has {MODELING_EMBEDDING_MODEL!r} "
        f"but {args.embedding_model!r} was requested"
    )
    logger.info(
        f"Modeling encoder {MODELING_EMBEDDING_MODEL} | eval encoder (fixed) "
        f"{EVAL_EMBEDDING_MODEL}"
    )

    fresh = bool(args.fresh)
    datasets = args.datasets.split(",")
    model_filter = set(args.models.split(","))
    models_to_run = [(name, cls) for name, cls in ALL_MODELS if name in model_filter]
    logger.info(
        f"TopicArena EMBEDDING worker {args.worker_id}: {len(datasets)} datasets, "
        f"{len(models_to_run)} models {[m for m, _ in models_to_run]} (fresh={fresh})"
    )

    all_results = []

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        n_topics = DATASET_TOPICS[dataset_name]

        dataset = load_and_preprocess(dataset_name)
        # Uses the swapped encoder via the env-overridden default arg.
        precompute_embeddings(dataset)

        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)

        for model_name, model_cls in models_to_run:
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
                result["embedding_model"] = args.embedding_model
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
