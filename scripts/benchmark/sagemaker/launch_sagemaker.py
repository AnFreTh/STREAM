"""
Launch SageMaker training jobs to complete the v5_hpo_cv_5seed benchmark.

Determines which (dataset, model) combos are already done on S3,
splits remaining work across N workers, and submits SageMaker jobs.

Usage:
    python launch_sagemaker.py
    python launch_sagemaker.py --dry-run
    python launch_sagemaker.py --n-workers 18 --instance-type ml.g5.xlarge
"""

import os
import sys
import io
import json
import shutil
import tempfile
import argparse
from pathlib import Path

import boto3
import pandas as pd

try:
    # SageMaker SDK v2
    import sagemaker
    from sagemaker.pytorch import PyTorch
except (ImportError, ModuleNotFoundError):
    # SageMaker SDK v3
    import sagemaker
    from sagemaker.train.pytorch import PyTorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_common import DATASETS, ALL_MODELS
from config import S3_BUCKET, S3_PREFIX, AWS_REGION, boto_session, require

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERSION = "v8_hpo_cv_5h"

SM_ROLE_ARN = require("SM_ROLE_ARN")

INSTANCE_TYPE = "ml.g5.8xlarge"  # 1x A10G, 128GB RAM
MAX_RUNTIME = 5 * 24 * 60 * 60  # 5 days
VOLUME_SIZE_GB = 100
N_GROUPS = 3  # jobs (instances) per dataset; models split across groups by train time


# ---------------------------------------------------------------------------
# Determine remaining work
# ---------------------------------------------------------------------------
def get_completed_combos(session):
    """Check S3 for which (dataset, model) combos already have 5 seeds done."""
    s3 = session.client("s3")
    completed = set()
    total = len(DATASETS) * len(ALL_MODELS)
    checked = 0

    for dataset in DATASETS:
        ds_done = 0
        for model_name, _ in ALL_MODELS:
            checked += 1
            seeds_done = 0
            for seed in [42, 84, 126, 168, 210]:
                key = f"{S3_PREFIX}/{dataset}/{VERSION}/{model_name}/metrics_seed{seed}.json"
                try:
                    s3.head_object(Bucket=S3_BUCKET, Key=key)
                    seeds_done += 1
                except:
                    break
            if seeds_done >= 5:
                completed.add((dataset, model_name))
                ds_done += 1
            pct = 100 * checked / total
            print(
                f"\r  [{pct:5.1f}%] {dataset}: {ds_done}/16 models done",
                end="",
                flush=True,
            )
        print()

    return completed


def get_remaining_datasets(completed):
    """Group remaining work by dataset (since we process all models per dataset)."""
    all_combos = {(ds, m) for ds in DATASETS for m, _ in ALL_MODELS}
    remaining = all_combos - completed

    # Group by dataset
    remaining_datasets = set()
    for ds, _ in remaining:
        remaining_datasets.add(ds)

    return sorted(remaining_datasets)


def split_models_by_time(n_groups=4):
    """Split the 16 models into n_groups, balanced by COUNT with heavy models spread out.

    Under HPO the cost per job is dominated by the number of models (each capped
    at the HPO timeout), not their default train time. So we balance group sizes
    and round-robin the time-sorted list to spread the slowest models across groups.
    """
    # Approximate training times from v2 benchmark (seconds, averaged across datasets)
    model_times = [
        ("TNTM", 4987),
        ("BERTopicTM", 2410),
        ("NSTM", 1482),
        ("FASTopic", 1143),
        ("ECRTM", 692),
        ("CTMNeg", 469),
        ("CTM", 436),
        ("HyperMiner", 467),
        ("SawETM", 313),
        ("ProdLDA", 213),
        ("NeuralLDA", 186),
        ("ETM", 186),
        ("LDA", 115),
        ("KmeansTM", 13),
        ("NMFTM", 14),
        ("KmeansTM_PCA", 5),
    ]

    # Sort by train time desc, then round-robin into groups: equalizes counts
    # while spreading the heaviest models one-per-group.
    model_times.sort(key=lambda x: x[1], reverse=True)
    groups = [[] for _ in range(n_groups)]
    times = [0] * n_groups
    for i, (model, t) in enumerate(model_times):
        idx = i % n_groups
        groups[idx].append(model)
        times[idx] += t

    return groups, times


# ---------------------------------------------------------------------------
# Build source directory for SageMaker
# ---------------------------------------------------------------------------
def build_source_dir():
    """Create a temp directory with all source code needed for the job."""
    repo_root = (
        Path(__file__).resolve().parent.parent.parent.parent
    )  # sagemaker/ -> benchmark/ -> scripts/ -> STREAM/

    tmp_dir = tempfile.mkdtemp(prefix="topicarena_sm_")
    tmp_path = Path(tmp_dir)

    # Copy stream_topic package
    shutil.copytree(
        repo_root / "stream_topic",
        tmp_path / "stream_topic",
        ignore=shutil.ignore_patterns(
            "*.pyc", "__pycache__", "*.egg-info", "stream_topic_data"
        ),
    )

    # Copy scripts/benchmark
    shutil.copytree(
        repo_root / "scripts" / "benchmark",
        tmp_path / "scripts" / "benchmark",
        ignore=shutil.ignore_patterns(
            "*.pyc",
            "__pycache__",
            "*.log",
            "*.ckpt",
            "checkpoints",
            "embeddings",
            "lightning_logs",
            "models",
        ),
    )
    # Ensure scripts/__init__.py exists
    (tmp_path / "scripts" / "__init__.py").touch()

    # Copy entry point to root
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point.py",
        tmp_path / "entry_point.py",
    )

    # Copy requirements
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "requirements.txt",
        tmp_path / "requirements.txt",
    )

    return tmp_dir


# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------
def launch_jobs(remaining_datasets, n_workers, instance_type, dry_run=False):
    """Launch N_GROUPS jobs per dataset (models split by training time)."""
    session = boto_session()
    sm_session = sagemaker.Session(boto_session=session)

    groups, times = split_models_by_time(N_GROUPS)
    print(f"\nModel split ({N_GROUPS} groups):")
    for g, (grp, t) in enumerate(zip(groups, times)):
        print(f"  Group {chr(97+g)} ({t/3600:.1f}h): {grp}")

    # N_GROUPS jobs per dataset
    jobs_plan = []
    for ds in remaining_datasets:
        for g, grp_models in enumerate(groups):
            if grp_models:  # skip empty groups (fewer models than groups)
                jobs_plan.append((ds, grp_models, chr(97 + g)))

    print(
        f"\nLaunching {len(jobs_plan)} SageMaker jobs ({len(remaining_datasets)} datasets × {N_GROUPS} groups):"
    )
    for ds, models, grp in jobs_plan[:6]:
        print(f"  {ds} group-{grp}: {len(models)} models")
    if len(jobs_plan) > 6:
        print(f"  ... and {len(jobs_plan)-6} more")

    if dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        return

    # Build source directory once
    source_dir = build_source_dir()
    print(f"\nSource dir: {source_dir}")

    job_names = []
    for i, (ds, models, grp) in enumerate(jobs_plan):
        models_str = ",".join(models)

        estimator = PyTorch(
            entry_point="entry_point.py",
            source_dir=source_dir,
            role=SM_ROLE_ARN,
            instance_type=instance_type,
            instance_count=1,
            framework_version="2.3.0",
            py_version="py311",
            volume_size=VOLUME_SIZE_GB,
            max_run=MAX_RUNTIME,
            base_job_name=f"topicarena-{grp}-w{i}",
            sagemaker_session=sm_session,
            hyperparameters={
                "datasets": ds,
                "models": models_str,
                "version": VERSION,
                "hpo_timeout": 5 * 60 * 60,  # 5 hours per model per dataset
                "worker_id": f"{i}_{grp}",
            },
            environment={
                "CUDA_MODULE_LOADING": "LAZY",
                "FI_EFA_FORK_SAFE": "1",
                "RDMAV_FORK_SAFE": "1",
            },
            debugger_hook_config=False,
            disable_profiler=True,
        )

        estimator.fit(wait=False)
        job_name = estimator.latest_training_job.name
        job_names.append(job_name)
        print(f"  Submitted: {job_name}")

    # Cleanup
    shutil.rmtree(source_dir)

    # Save job info
    jobs_file = "sagemaker_jobs.json"
    with open(jobs_file, "w") as f:
        json.dump(
            {
                "version": VERSION,
                "jobs": job_names,
                "plan": [(ds, grp) for ds, _, grp in jobs_plan],
            },
            f,
            indent=2,
        )
    print(f"\nJob info saved to {jobs_file}")
    print(f"Monitor: aws sagemaker list-training-jobs --name-contains topicarena")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Number of workers (default: 1 per remaining dataset)",
    )
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"TopicArena SageMaker Launcher")
    print(f"Version: {VERSION}")
    print(f"Instance: {args.instance_type}")
    print()

    session = boto_session()

    print("Checking completed work on S3...")
    completed = get_completed_combos(session)
    total = len(DATASETS) * len(ALL_MODELS)
    print(f"  Completed: {len(completed)}/{total} ({100*len(completed)/total:.0f}%)")

    remaining_datasets = get_remaining_datasets(completed)
    print(f"  Remaining datasets: {len(remaining_datasets)}/{len(DATASETS)}")
    print(f"  Datasets: {remaining_datasets}")

    if not remaining_datasets:
        print("\nAll done!")
        return

    n_workers = args.n_workers or len(remaining_datasets)
    print(f"\n  Workers: {n_workers}")
    print(f"  Datasets per worker: ~{len(remaining_datasets)/n_workers:.1f}")
    print(
        f"  Jobs per dataset: {N_GROUPS} (each ~{16/N_GROUPS:.0f} models × up to 5h HPO)"
    )

    launch_jobs(remaining_datasets, n_workers, args.instance_type, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
