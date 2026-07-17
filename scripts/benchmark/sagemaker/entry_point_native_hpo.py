"""
SageMaker entry point for TopicArena HPO with native objectives.
DEBUG VERSION - prints on every line.
"""

import os
import sys
import argparse

print("[ENTRY] starting entry_point.py", flush=True)

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
print("[ENTRY] sys.path set, starting main", flush=True)


def main():
    print("[ENTRY] main() called", flush=True)

    print("[ENTRY] downloading nltk...", flush=True)
    import nltk

    for resource in [
        "stopwords",
        "wordnet",
        "punkt_tab",
        "brown",
        "averaged_perceptron_tagger",
    ]:
        print(f"[ENTRY] nltk downloading {resource}...", flush=True)
        nltk.download(resource, quiet=True)
    print("[ENTRY] nltk done", flush=True)

    print("[ENTRY] parsing args...", flush=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", required=True)
    parser.add_argument("--models", default=None)
    parser.add_argument("--version", default="v7_hpo_native_5h")
    parser.add_argument("--hpo_timeout", type=int, default=5 * 60 * 60)
    parser.add_argument("--worker_id", default="0")
    args = parser.parse_args()
    print(
        f"[ENTRY] args: datasets={args.datasets} models={args.models} version={args.version} worker={args.worker_id}",
        flush=True,
    )

    print("[ENTRY] importing TMDataset...", flush=True)
    from stream_topic.utils import TMDataset

    print("[ENTRY] TMDataset done", flush=True)

    print("[ENTRY] importing loguru...", flush=True)
    from loguru import logger
    import pandas as pd

    print("[ENTRY] loguru+pandas done", flush=True)

    print("[ENTRY] setting bench path...", flush=True)
    sys.path.insert(
        0,
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark"
        ),
    )
    print("[ENTRY] importing bench_common...", flush=True)
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

    print("[ENTRY] bench_common done", flush=True)

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
    print(f"[ENTRY] datasets: {datasets}", flush=True)

    if args.models:
        model_filter = set(args.models.split(","))
        models_to_run = [
            (name, cls) for name, cls in ALL_MODELS if name in model_filter
        ]
    else:
        models_to_run = ALL_MODELS
    print(f"[ENTRY] models_to_run: {[m for m,_ in models_to_run]}", flush=True)

    all_results = []

    for dataset_name in datasets:
        print(f"[ENTRY] === DATASET: {dataset_name} ===", flush=True)
        n_topics = DATASET_TOPICS[dataset_name]
        print(f"[ENTRY] n_topics={n_topics}", flush=True)

        print(f"[ENTRY] load_and_preprocess({dataset_name})...", flush=True)
        dataset = load_and_preprocess(dataset_name)
        print(f"[ENTRY] load_and_preprocess done", flush=True)

        print(f"[ENTRY] precompute_embeddings...", flush=True)
        precompute_embeddings(dataset)
        print(f"[ENTRY] precompute_embeddings done", flush=True)

        print(f"[ENTRY] fetching raw_dataset...", flush=True)
        raw_dataset = TMDataset()
        raw_dataset.fetch_dataset(dataset_name)
        print(f"[ENTRY] raw_dataset done", flush=True)

        for model_name, model_cls in models_to_run:
            print(f"[ENTRY] --- MODEL: {model_name} ---", flush=True)

            # Skip if already completed
            print(f"[ENTRY] checking storage for existing results...", flush=True)
            skip = True
            for seed in SEEDS:
                key = f"{S3_PREFIX}/{dataset_name}/{args.version}/{model_name}/metrics_seed{seed}.json"
                if not obj_exists(key):
                    skip = False
                    break
            if skip:
                print(f"[ENTRY] SKIP {model_name} (already complete)", flush=True)
                continue
            print(f"[ENTRY] not complete, will run", flush=True)

            # Skip HPO for slow combos
            if (dataset_name, model_name) in SKIP_HPO:
                print(f"[ENTRY] skipping HPO (too slow), using defaults", flush=True)
                best_hparams, hpo_result = None, None
            else:
                print(f"[ENTRY] running HPO...", flush=True)
                best_hparams, hpo_result = run_hpo(
                    dataset_name,
                    model_name,
                    model_cls,
                    dataset,
                    raw_dataset,
                    n_topics,
                    version=args.version,
                    hpo_timeout=args.hpo_timeout,
                )
                print(f"[ENTRY] HPO done", flush=True)

            # 5-seed evaluation
            for seed in SEEDS:
                print(f"[ENTRY] run_single seed={seed}...", flush=True)
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
                    result["hpo_criterion"] = "native"
                all_results.append(result)
                print(f"[ENTRY] run_single seed={seed} done", flush=True)

            print(f"[ENTRY] saving incremental CSV...", flush=True)
            df = pd.DataFrame(all_results)
            s3_put_csv(
                df,
                f"{S3_PREFIX}/_results/{args.version}_all_worker{args.worker_id}.csv",
            )
            print(f"[ENTRY] saved", flush=True)

    print(f"[ENTRY] saving final CSV...", flush=True)
    df = pd.DataFrame(all_results)
    s3_put_csv(
        df, f"{S3_PREFIX}/_results/{args.version}_all_worker{args.worker_id}.csv"
    )
    print(f"[ENTRY] DONE! {len(all_results)} runs completed.", flush=True)


if __name__ == "__main__":
    main()
