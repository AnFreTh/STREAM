"""
One-off: submit SageMaker jobs for the v8 CV combos still missing from S3.

Computes the missing (dataset, model) set by diffing S3 against the full plan,
then submits ONE job per missing combo (max parallelism; slow models like TNTM
get their own instance so they aren't starved). The entry point re-checks S3 and
skips already-complete combos, so this is safe to re-run.

Usage:
    python resume_missing_cv.py --dry-run
    python resume_missing_cv.py
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path

import sagemaker
from sagemaker.pytorch import PyTorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# Self-contained: avoid importing bench_common (pulls heavy optional deps).
from config import S3_BUCKET, S3_PREFIX, boto_session, require  # noqa: E402

# Region is fixed to where the data + quota live; do NOT inherit a stray shell
# AWS_REGION (an exported us-west-2 caused a prior misfire).
AWS_REGION = "us-east-1"
VERSION = "v8_hpo_cv_5h"
SM_ROLE_ARN = require("SM_ROLE_ARN")
INSTANCE_TYPE = "ml.g5.8xlarge"
MAX_RUNTIME = 5 * 24 * 60 * 60
VOLUME_SIZE_GB = 100
HPO_TIMEOUT = 5 * 60 * 60

DATASETS = [
    "BBC_News", "20Newsgroups", "Poliblogs", "UN", "WHO", "NeurIPS", "ACL",
    "Reuters", "NYT", "Spotify", "Reddit_GME", "IMDB", "AG_News", "Arxiv",
    "WikiText", "PubMed", "DBpedia", "Yahoo_Answers",
]
MODELS = [
    "LDA", "NMFTM", "KmeansTM", "KmeansTM_PCA", "BERTopicTM", "ETM", "ProdLDA",
    "NeuralLDA", "CTM", "CTMNeg", "NSTM", "FASTopic", "ECRTM", "SawETM",
    "HyperMiner", "TNTM",
]
SEEDS = [42, 84, 126, 168, 210]

import boto3
_s3 = boto3.Session(region_name=AWS_REGION).client("s3")


def find_missing():
    """Return (dataset, model) combos lacking all 5 seed metrics on S3.

    Uses ONE paginated list_objects_v2 per version (seconds) instead of thousands
    of head_object calls, which throttle and hang.
    """
    import re
    seen = {}  # (ds, model) -> {seed: key}
    paginator = _s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=f"{S3_PREFIX}/"):
        for obj in page.get("Contents", []):
            m = re.search(
                rf"{re.escape(S3_PREFIX)}/([^/]+)/{re.escape(VERSION)}/([^/]+)/metrics_seed(\d+)\.json$",
                obj["Key"],
            )
            if m:
                seen.setdefault((m.group(1), m.group(2)), {})[m.group(3)] = obj["Key"]

    def _seed_succeeded(key):
        # A FAILED run still writes a metrics JSON with an "error" key
        # (bench_common.run_single). Treating that as complete would skip the
        # combo forever, silently leaving it failed. Count a seed done only if
        # its JSON has no "error" field.
        try:
            body = _s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
            return "error" not in json.loads(body)
        except Exception:
            return False

    missing = []
    for ds in DATASETS:
        for model in MODELS:
            files = seen.get((ds, model), {})
            if len(files) < len(SEEDS):
                missing.append((ds, model))
                continue
            # All 5 files present -> verify none is an error-stub. Only these
            # (few) apparently-complete combos incur a get_object per seed.
            if not all(_seed_succeeded(k) for k in files.values()):
                missing.append((ds, model))
    return missing


def build_source_dir():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tmp = Path(tempfile.mkdtemp(prefix="topicarena_resumecv_"))
    shutil.copytree(
        repo_root / "stream_topic",
        tmp / "stream_topic",
        ignore=shutil.ignore_patterns("*.pyc", "__pycache__", "*.egg-info", "stream_topic_data"),
    )
    shutil.copytree(
        repo_root / "scripts" / "benchmark",
        tmp / "scripts" / "benchmark",
        ignore=shutil.ignore_patterns(
            "*.pyc", "__pycache__", "*.log", "*.ckpt", ".env",
            "checkpoints", "embeddings", "lightning_logs", "models", "results",
        ),
    )
    (tmp / "scripts" / "__init__.py").touch()
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point.py",
        tmp / "entry_point.py",
    )
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "requirements.txt",
        tmp / "requirements.txt",
    )
    return str(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    missing = find_missing()
    print(f"Missing v8 CV combos: {len(missing)}")
    from collections import Counter
    by_ds = Counter(ds for ds, _ in missing)
    for ds, n in sorted(by_ds.items()):
        print(f"  {ds:14} {n} models")

    if not missing:
        print("Nothing to do.")
        return

    if args.dry_run:
        print(f"\n[DRY RUN] would submit {len(missing)} jobs (one per combo).")
        for ds, model in missing[:10]:
            print(f"  {ds} / {model}")
        print("  ...")
        return

    import boto3
    session = boto3.Session(region_name=AWS_REGION)  # us-east-1, ignore shell profile/region
    sm_session = sagemaker.Session(boto_session=session)
    source_dir = build_source_dir()
    print(f"\nSource dir: {source_dir}")

    submitted = []
    for i, (ds, model) in enumerate(missing):
        # short, valid base name: topicarena-cvfix-<i>
        estimator = PyTorch(
            entry_point="entry_point.py",
            source_dir=source_dir,
            role=SM_ROLE_ARN,
            instance_type=INSTANCE_TYPE,
            instance_count=1,
            framework_version="2.3.0",
            py_version="py311",
            volume_size=VOLUME_SIZE_GB,
            max_run=MAX_RUNTIME,
            base_job_name=f"topicarena-cvfix-{i}",
            sagemaker_session=sm_session,
            hyperparameters={
                "datasets": ds,
                "models": model,
                "version": VERSION,
                "hpo_timeout": HPO_TIMEOUT,
                "worker_id": f"cvfix_{i}",
            },
            environment={
                # Storage config passed via env (NOT via a shipped .env file —
                # the container authenticates with its instance role, so no
                # AWS_PROFILE/ARN is passed or needed).
                "TOPICARENA_STORAGE": "s3",
                "TOPICARENA_S3_BUCKET": S3_BUCKET,
                "TOPICARENA_S3_PREFIX": S3_PREFIX,
                "AWS_DEFAULT_REGION": AWS_REGION,
                "CUDA_MODULE_LOADING": "LAZY",
                "FI_EFA_FORK_SAFE": "1",
                "RDMAV_FORK_SAFE": "1",
                "TOKENIZERS_PARALLELISM": "false",
            },
            debugger_hook_config=False,
            disable_profiler=True,
        )
        estimator.fit(wait=False)
        name = estimator.latest_training_job.name
        submitted.append({"job": name, "dataset": ds, "model": model})
        print(f"  [{i+1}/{len(missing)}] {name}  ({ds}/{model})")

    shutil.rmtree(source_dir, ignore_errors=True)
    out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "cvfix_jobs.json"))
    with open(out, "w") as f:
        json.dump(submitted, f, indent=2)
    print(f"\nSubmitted {len(submitted)} jobs. Info -> {out}")


if __name__ == "__main__":
    main()
