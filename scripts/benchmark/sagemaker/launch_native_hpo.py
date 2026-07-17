"""
Launch SageMaker jobs for v3 HPO with native objectives.

Same structure as v5 launcher: 36 jobs (18 datasets × 2 model groups),
but uses native objectives (val_loss, AIC) instead of CV coherence.

Usage:
    python launch_native_hpo.py --dry-run
    python launch_native_hpo.py
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from bench_common import DATASETS, ALL_MODELS
from config import S3_BUCKET, S3_PREFIX, AWS_REGION, boto_session, require

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERSION = "v7_hpo_native_5h"

SM_ROLE_ARN = require("SM_ROLE_ARN")

INSTANCE_TYPE = "ml.g5.8xlarge"  # 1x A10G, 128GB RAM
MAX_RUNTIME = 5 * 24 * 60 * 60  # 5 days
VOLUME_SIZE_GB = 100
N_GROUPS = 3  # jobs (instances) per dataset; models split across groups by train time


def split_models_by_time(n_groups=4):
    """Split the 16 models into n_groups, balanced by COUNT with heavy models spread out.

    Under HPO the cost per job is dominated by the number of models (each capped
    at the HPO timeout), not their default train time. So we balance group sizes
    and round-robin the time-sorted list to spread the slowest models across groups.
    """
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
    # Sort by train time desc, then round-robin: equalizes counts while
    # spreading the heaviest models one-per-group.
    model_times.sort(key=lambda x: x[1], reverse=True)
    groups = [[] for _ in range(n_groups)]
    times = [0] * n_groups
    for i, (model, t) in enumerate(model_times):
        idx = i % n_groups
        groups[idx].append(model)
        times[idx] += t
    return groups, times


def build_source_dir():
    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    tmp_dir = tempfile.mkdtemp(prefix="topicarena_v3_")
    tmp_path = Path(tmp_dir)

    shutil.copytree(
        repo_root / "stream_topic",
        tmp_path / "stream_topic",
        ignore=shutil.ignore_patterns(
            "*.pyc", "__pycache__", "*.egg-info", "stream_topic_data"
        ),
    )

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
    (tmp_path / "scripts" / "__init__.py").touch()

    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point_native_hpo.py",
        tmp_path / "entry_point.py",
    )

    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "requirements.txt",
        tmp_path / "requirements.txt",
    )

    return tmp_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    args = parser.parse_args()

    print(f"TopicArena v3 HPO (Native Objectives) Launcher")
    print(f"Version: {VERSION}")
    print(f"Instance: {args.instance_type}")

    groups, times = split_models_by_time(N_GROUPS)
    print(f"\nModel split ({N_GROUPS} groups):")
    for g, (grp, t) in enumerate(zip(groups, times)):
        print(f"  Group {chr(97+g)} ({t/3600:.1f}h): {grp}")

    jobs_plan = []
    for ds in DATASETS:
        for g, grp_models in enumerate(groups):
            if grp_models:  # skip empty groups (fewer models than groups)
                jobs_plan.append((ds, grp_models, chr(97 + g)))

    print(
        f"\nTotal jobs: {len(jobs_plan)} ({len(DATASETS)} datasets × {N_GROUPS} groups)"
    )

    if args.dry_run:
        print(f"\n[DRY RUN] Would launch {len(jobs_plan)} jobs.")
        for ds, models, grp in jobs_plan[:6]:
            print(f"  {ds} group-{grp}: {models}")
        print(f"  ...")
        return

    session = boto_session()
    sm_session = sagemaker.Session(boto_session=session)

    source_dir = build_source_dir()
    print(f"\nSource dir: {source_dir}")

    job_names = []
    for i, (ds, models, grp) in enumerate(jobs_plan):
        models_str = ",".join(models)

        estimator = PyTorch(
            entry_point="entry_point.py",
            source_dir=source_dir,
            role=SM_ROLE_ARN,
            instance_type=args.instance_type,
            instance_count=1,
            framework_version="2.3.0",
            py_version="py311",
            volume_size=VOLUME_SIZE_GB,
            max_run=MAX_RUNTIME,
            base_job_name=f"topicarena-v3-{grp}-w{i}",
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
                "TOKENIZERS_PARALLELISM": "false",
            },
            debugger_hook_config=False,
            disable_profiler=True,
        )

        estimator.fit(wait=False)
        job_name = estimator.latest_training_job.name
        job_names.append(job_name)
        print(f"  Submitted: {job_name}")

    shutil.rmtree(source_dir)

    jobs_file = "sagemaker_v3_jobs.json"
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
    print(f"Monitor: aws sagemaker list-training-jobs --name-contains topicarena-v3")


if __name__ == "__main__":
    main()
