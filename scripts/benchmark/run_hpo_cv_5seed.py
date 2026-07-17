"""
Script 4: HPO benchmark optimizing CV coherence — 1 hour per model per dataset, then 5-seed eval.

Same as script 3 but uses CV coherence as the HPO objective instead of val_loss/aic.
Results saved to S3 under version="v4_hpo_cv_5seed".
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
from stream_topic.metrics import CV

VERSION = "v8_hpo_cv_5h"
HPO_TIMEOUT = 5 * 60 * 60  # 5 hours per model per dataset

# Model+dataset combos where default train > 30min — HPO would be useless
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


class CVMetricWrapper:
    """Wrapper so CV metric works with optimize_and_fit(custom_metric=...)."""

    def __init__(self, raw_dataset, n_words=10):
        self._cv = CV(raw_dataset, n_words=n_words)

    def score(self, topics):
        return self._cv.score(topics)


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

        cv_metric = CVMetricWrapper(raw_dataset)

        for model_name, model_cls in ALL_MODELS:
            # Phase 1: HPO with CV coherence (skip slow combos)
            if (dataset_name, model_name) in SKIP_HPO:
                logger.info(
                    f"  Skipping HPO for {model_name} on {dataset_name} (default train > 30min)"
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
                    version=VERSION,
                    hpo_timeout=HPO_TIMEOUT,
                    criterion="custom",
                    custom_metric=cv_metric,
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
                    result["hpo_criterion"] = "cv_coherence"
                all_results.append(result)

            # Save incremental results
            df = pd.DataFrame(all_results)
            s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")

    # Final
    df = pd.DataFrame(all_results)
    s3_put_csv(df, f"{S3_PREFIX}/_results/{VERSION}_all{suffix}.csv")

    metrics = ["NMI", "Purity"] + [
        f"{m}@{k}"
        for m in [
            "CV",
            "CV_train",
            "NPMI",
            "TD",
            "ISIM",
            "INT",
            "ISH",
            "Emb_Coherence",
            "Emb_TD",
        ]
        for k in [5, 10, 15, 20]
    ]
    summary = df.groupby(["dataset", "model"])[metrics].agg(["mean", "std"])
    s3_put_csv(
        summary.reset_index(), f"{S3_PREFIX}/_results/{VERSION}_summary{suffix}.csv"
    )

    logger.info(f"\nDone! {len(all_results)} runs")


if __name__ == "__main__":
    main()
