"""
SageMaker entry point for the K-sensitivity sweep (rebuttal).
DEFAULT hyperparameters (no HPO), 5 seeds, at a specified n_topics K.
Results saved under version 'ksweep_k{K}' so each K is separable and skippable.
"""
import os, sys, argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts", "benchmark"))
from config import S3_BUCKET, S3_PREFIX, boto_session  # noqa: E402

MODELS_DIR = "/tmp/st_cache_proper"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = MODELS_DIR
os.environ["HF_HOME"] = MODELS_DIR
os.environ["TRANSFORMERS_CACHE"] = MODELS_DIR
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"

import tarfile
if not os.path.exists(os.path.join(MODELS_DIR, "models--sentence-transformers--all-MiniLM-L6-v2")):
    s3 = boto_session().client("s3"); tar_path="/tmp/st_models.tar.gz"
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/_models/st_models.tar.gz", tar_path)
    with tarfile.open(tar_path,"r:gz") as t: t.extractall("/tmp")
    os.remove(tar_path)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stream_topic", "stream_topic_data")
if not os.path.exists(os.path.join(DATA_DIR,"preprocessed_datasets","BBC_News")):
    s3 = boto_session().client("s3"); tar_path="/tmp/st_data.tar.gz"
    s3.download_file(S3_BUCKET, f"{S3_PREFIX}/_models/st_data.tar.gz", tar_path)
    with tarfile.open(tar_path,"r:gz") as t:
        t.extractall(os.path.join(os.path.dirname(os.path.abspath(__file__)),"stream_topic"))
    os.remove(tar_path)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
from loguru import logger


def main():
    import nltk
    for r in ["stopwords","wordnet","punkt_tab","brown","averaged_perceptron_tagger"]:
        nltk.download(r, quiet=True)

    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--n_topics", type=int, required=True)   # the swept K
    ap.add_argument("--models", default=None)
    ap.add_argument("--seeds", default=None,
                    help="Comma-separated seed subset (default: all 5). Lets a slow "
                         "big-corpus model be parallelized one seed per job.")
    ap.add_argument("--worker_id", default="0")
    args = ap.parse_args()
    K = args.n_topics
    # A capped rerun (STREAM_MAX_EPOCHS set) writes to its OWN namespace so it
    # never overwrites the uncapped jobs already running for the same K.
    _me = os.environ.get("STREAM_MAX_EPOCHS", "").strip()
    version = f"ksweep_k{K}_e{_me}" if _me else f"ksweep_k{K}"

    from stream_topic.utils import TMDataset
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"scripts","benchmark"))
    from bench_common import (ALL_MODELS, SEEDS, load_and_preprocess,
                              precompute_embeddings, run_single, s3_put_csv,
                              S3_PREFIX as PFX, S3_BUCKET as BKT, obj_exists)

    datasets = args.datasets.split(",")
    if args.models:
        mf=set(args.models.split(",")); models=[(n,c) for n,c in ALL_MODELS if n in mf]
    else:
        models=ALL_MODELS
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else SEEDS

    all_results=[]
    for ds in datasets:
        logger.info(f"=== {ds} @ K={K} ===")
        dataset = load_and_preprocess(ds)
        precompute_embeddings(dataset)
        raw = TMDataset(); raw.fetch_dataset(ds)
        for mname, mcls in models:
            # skip if the requested seeds are already done for this K
            done=all(obj_exists(f"{PFX}/{ds}/{version}/{mname}/metrics_seed{s}.json") for s in seeds)
            if done:
                logger.info(f"  SKIP {mname} (complete)"); continue
            for seed in seeds:
                if obj_exists(f"{PFX}/{ds}/{version}/{mname}/metrics_seed{seed}.json"):
                    continue   # per-seed skip: only compute missing seeds
                try:
                    r = run_single(ds, mname, mcls, dataset, raw, K, seed=seed, version=version)
                    all_results.append(r)
                except Exception as e:
                    logger.error(f"  {mname} seed{seed} K{K} FAILED: {e}")
            df=pd.DataFrame(all_results)
            s3_put_csv(df, f"{PFX}/_results/{version}_all_worker{args.worker_id}.csv")
    logger.info(f"DONE K={K}: {len(all_results)} runs")


if __name__ == "__main__":
    main()
