"""
Script 3: HPO benchmark — 1 hour per model per dataset, then 5-seed eval.

For each (dataset, model):
  1. Run Optuna HPO for 1 hour with fixed n_topics, optimizing val_loss/aic
  2. Save best hyperparameters to S3
  3. Re-train 5 times with best params + different seeds, evaluate all metrics

Results saved to S3 under version="v3_hpo_5seed".
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pandas as pd
from loguru import logger
from bench_common import (
    DATASETS,
    DATASET_TOPICS,
    ALL_MODELS,
    SEEDS,
    load_and_preprocess,
    precompute_embeddings,
    run_single,
    run_hpo,
    s3_put_csv,
    S3_PREFIX,
)
from stream_topic.utils import TMDataset

VERSION = "v7_hpo_native_5h"
HPO_TIMEOUT = 5 * 60 * 60  # 5 hours per model per dataset


def main():
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

        dataset = load_and_preprocess(dataset_name)
        precompute_embeddings(dataset)

        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)

        for model_name, model_cls in ALL_MODELS:
            # Phase 1: HPO
            best_hparams, hpo_result = run_hpo(
                dataset_name,
                model_name,
                model_cls,
                dataset,
                raw_dataset,
                n_topics,
                version=VERSION,
                hpo_timeout=HPO_TIMEOUT,
            )

            # Phase 2: 5-seed evaluation with best params
            for seed in SEEDS:
                result = run_single(
                    dataset_name,
                    model_name,
                    model_cls,
                    dataset,
                    raw_dataset,
                    n_topics,
                    seed=seed,
                    version=VERSION,
                    hparams_override=best_hparams,
                )
                if hpo_result is not None:
                    result["hpo_time"] = hpo_result["hpo_time"]
                    result["hpo_criterion"] = hpo_result.get("criterion", "native")
                all_results.append(result)

            # Save incremental results
            df = pd.DataFrame(all_results)
            s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")

    # Final
    df = pd.DataFrame(all_results)
    s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")

    metrics = ["NMI", "Purity"] + [
        f"{m}@{k}"
        for m in ["CV", "NPMI", "TD", "ISIM", "INT", "ISH", "Emb_Coherence", "Emb_TD"]
        for k in [5, 10, 15, 20]
    ]
    summary = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    s3_put_csv(
        summary.reset_index(), f"{S3_PREFIX}/_results/{VERSION}_summary{suffix}.csv"
    )

    logger.info(f"\nDone! {len(all_results)} runs")


if __name__ == "__main__":
    main()
