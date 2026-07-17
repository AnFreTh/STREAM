"""
Script 1: Default benchmark — single run, default hyperparameters.

Sanity check baseline. One run per model per dataset.
Results saved to S3 under version="v1_default".
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
from loguru import logger
from bench_common import (
    DATASETS, DATASET_TOPICS, ALL_MODELS, SEEDS,
    load_and_preprocess, precompute_embeddings,
    run_single, s3_put_csv, S3_BUCKET, S3_PREFIX,
)
from stream_topic.utils import TMDataset

VERSION = "v1_default"


def main():
    # Support multi-GPU: override dataset list from env
    dataset_list = os.environ.get("BENCHMARK_DATASETS")
    datasets = dataset_list.split(",") if dataset_list else DATASETS
    gpu_id = os.environ.get("BENCHMARK_GPU_ID", "")
    if gpu_id:
        logger.info(f"Worker GPU {gpu_id}: processing {datasets}")

    all_results = []
    suffix = f"_gpu{gpu_id}" if gpu_id else ""

    for dataset_name in datasets:
        logger.info(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
        n_topics = DATASET_TOPICS[dataset_name]

        # Load and preprocess once
        dataset = load_and_preprocess(dataset_name)
        precompute_embeddings(dataset)

        # Raw dataset for NPMI/CV
        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)

        for model_name, model_cls in ALL_MODELS:
            result = run_single(
                dataset_name, model_name, model_cls,
                dataset, raw_dataset, n_topics,
                seed=SEEDS[0], version=VERSION,
            )
            all_results.append(result)

            # Save incremental results
            df = pd.DataFrame(all_results)
            s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")

    # Final summary
    df = pd.DataFrame(all_results)
    s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")
    logger.info(f"\nDone! {len(all_results)} runs saved.")
    print(df[["dataset", "model", "CV@10", "NPMI@10", "TD@10", "train_time"]].to_string())


if __name__ == "__main__":
    main()
