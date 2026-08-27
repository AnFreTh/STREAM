"""
Launch SageMaker training jobs for the TopicArena DEFAULT benchmark
(v2_default_5seed): default hyperparameters, 5 seeds, no HPO.

Mirrors launch_sagemaker.py but targets entry_point_default.py and does not
pass an HPO timeout. Determines remaining work from S3 and splits models across
N_GROUPS jobs per dataset.

Usage:
    python launch_default.py --dry-run
    python launch_default.py                       # resume: skip completed combos
    python launch_default.py --fresh               # recompute everything (corrected rerun)
    python launch_default.py --n-workers 18 --instance-type ml.g5.8xlarge
"""

import os
import sys
import json
import shutil
import tempfile
import argparse
from pathlib import Path

import boto3
import pandas as pd

try:
    import sagemaker
    from sagemaker.pytorch import PyTorch
except (ImportError, ModuleNotFoundError):
    import sagemaker
    from sagemaker.train.pytorch import PyTorch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bench_common import DATASETS, ALL_MODELS
from config import S3_BUCKET, S3_PREFIX, AWS_REGION, boto_session, require

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VERSION = "v2_default_5seed"

SM_ROLE_ARN = require("SM_ROLE_ARN")

INSTANCE_TYPE = "ml.g5.4xlarge"  # 1x A10G, 64GB RAM (quota 30; same GPU as 8xlarge)
MAX_RUNTIME = 5 * 24 * 60 * 60  # 5 days
VOLUME_SIZE_GB = 100
N_GROUPS = 3  # jobs (instances) per dataset; models split across groups by train time
SEEDS = [42, 84, 126, 168, 210]

# Max concurrent training jobs to keep in flight. us-east-1 quota for the g5 GPU
# types is 200, comfortably above the 54-job plan, so all fire in one wave. The
# quota-aware loop still self-throttles (requeue + backoff) if the account is
# busier than expected. NOTE: submit in us-east-1 (AWS_REGION=us-east-1) -- a
# stray AWS_REGION=us-west-2 sends jobs to a region where the quota is only 1.
MAX_CONCURRENT_JOBS = 180


# ---------------------------------------------------------------------------
# Determine remaining work
# ---------------------------------------------------------------------------
def get_completed_combos(session, version):
    """Check S3 for which (dataset, model) combos already have all 5 seeds done."""
    s3 = session.client("s3")
    completed = set()
    total = len(DATASETS) * len(ALL_MODELS)
    checked = 0

    for dataset in DATASETS:
        ds_done = 0
        for model_name, _ in ALL_MODELS:
            checked += 1
            seeds_done = 0
            for seed in SEEDS:
                key = f"{S3_PREFIX}/{dataset}/{version}/{model_name}/metrics_seed{seed}.json"
                try:
                    s3.head_object(Bucket=S3_BUCKET, Key=key)
                    seeds_done += 1
                except Exception:
                    break
            if seeds_done >= len(SEEDS):
                completed.add((dataset, model_name))
                ds_done += 1
            pct = 100 * checked / total
            print(
                f"\r  [{pct:5.1f}%] {dataset}: {ds_done}/{len(ALL_MODELS)} models done",
                end="",
                flush=True,
            )
        print()

    return completed


def get_remaining_datasets(completed):
    """Group remaining work by dataset (all models processed per dataset)."""
    all_combos = {(ds, m) for ds in DATASETS for m, _ in ALL_MODELS}
    remaining = all_combos - completed
    return sorted({ds for ds, _ in remaining})


def split_models_by_time(n_groups=3):
    """Split the 16 models into n_groups, balanced by train time (round-robin on
    a time-sorted list spreads the heaviest models one-per-group)."""
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
    repo_root = Path(__file__).resolve().parent.parent.parent.parent

    tmp_dir = tempfile.mkdtemp(prefix="topicarena_default_sm_")
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
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point_default.py",
        tmp_path / "entry_point_default.py",
    )
    # Also ship the HPO entry point (used when --hpo is passed).
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point_native_hpo.py",
        tmp_path / "entry_point_native_hpo.py",
    )
    # And the embedding-swap entry point (used when --embedding-model is passed).
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "entry_point_embedding.py",
        tmp_path / "entry_point_embedding.py",
    )
    shutil.copy(
        repo_root / "scripts" / "benchmark" / "sagemaker" / "requirements.txt",
        tmp_path / "requirements.txt",
    )
    return tmp_dir


# ---------------------------------------------------------------------------
# Launch jobs
# ---------------------------------------------------------------------------
def launch_jobs(remaining_datasets, instance_type, version, fresh=False,
                dry_run=False, models_override=None, extra_env=None, s3_prefix=None,
                hpo=False, hpo_timeout=5 * 60 * 60, n_groups=N_GROUPS,
                embedding_model=None):
    """Launch jobs per dataset. By default the 16 models are split across n_groups
    jobs (balanced by train time). If ``models_override`` is given (e.g. a TNTM-only
    A/B run), a single job per dataset runs exactly those models. ``extra_env`` is
    merged into each job's environment. When ``hpo`` is True, jobs run the HPO entry
    point (entry_point_native_hpo.py) with the given per-combo ``hpo_timeout``."""
    session = boto_session()
    sm_session = sagemaker.Session(boto_session=session)

    if models_override:
        groups = [list(models_override)]
        times = [0]
        print(f"\nModel override: single group with {models_override}")
    else:
        groups, times = split_models_by_time(n_groups)
        print(f"\nModel split ({n_groups} groups):")
        for g, (grp, t) in enumerate(zip(groups, times)):
            print(f"  Group {chr(97+g)} ({t/3600:.1f}h train est.): {grp}")

    jobs_plan = []
    for ds in remaining_datasets:
        for g, grp_models in enumerate(groups):
            if grp_models:
                jobs_plan.append((ds, grp_models, chr(97 + g)))

    print(
        f"\nLaunching {len(jobs_plan)} SageMaker jobs "
        f"({len(remaining_datasets)} datasets x {len(groups)} groups)"
        f"{' [HPO ' + str(hpo_timeout//3600) + 'h/combo]' if hpo else ''}:"
    )
    for ds, models, grp in jobs_plan[:6]:
        print(f"  {ds} group-{grp}: {len(models)} models")
    if len(jobs_plan) > 6:
        print(f"  ... and {len(jobs_plan)-6} more")

    if dry_run:
        print("\n[DRY RUN] No jobs submitted.")
        return

    source_dir = build_source_dir()
    print(f"\nSource dir: {source_dir}")

    sm_client = session.client("sagemaker")

    def in_flight(job_names):
        """Count job_names still InProgress or Stopping. Cheap: 1 API call/name."""
        active = 0
        for n in job_names:
            try:
                s = sm_client.describe_training_job(TrainingJobName=n)["TrainingJobStatus"]
                if s in ("InProgress", "Stopping"):
                    active += 1
            except Exception:
                pass  # missing/DNE -> not active
        return active

    def submit_one(i, ds, models, grp):
        if embedding_model:
            entry = "entry_point_embedding.py"
            hp = {
                "datasets": ds,
                "models": ",".join(models),
                "version": version,
                "worker_id": f"{i}_{grp}",
                "embedding_model": embedding_model,
                "fresh": 1 if fresh else 0,
            }
            base_name = f"topicarena-emb-{grp}-w{i}"
        elif hpo:
            entry = "entry_point_native_hpo.py"
            hp = {
                "datasets": ds,
                "models": ",".join(models),
                "version": version,
                "worker_id": f"{i}_{grp}",
                "hpo_timeout": hpo_timeout,   # per (dataset, model) HPO budget
            }
            base_name = f"topicarena-hpo-{grp}-w{i}"
        else:
            entry = "entry_point_default.py"
            hp = {
                "datasets": ds,
                "models": ",".join(models),
                "version": version,
                "worker_id": f"{i}_{grp}",
                "fresh": 1 if fresh else 0,
            }
            base_name = f"topicarena-def-{grp}-w{i}"
        estimator = PyTorch(
            entry_point=entry,
            source_dir=source_dir,
            role=SM_ROLE_ARN,
            instance_type=instance_type,
            instance_count=1,
            framework_version="2.3.0",
            py_version="py311",
            volume_size=VOLUME_SIZE_GB,
            max_run=MAX_RUNTIME,
            base_job_name=base_name,
            sagemaker_session=sm_session,
            hyperparameters=hp,
            environment={
                "CUDA_MODULE_LOADING": "LAZY",
                "FI_EFA_FORK_SAFE": "1",
                "RDMAV_FORK_SAFE": "1",
                # Reduce CUDA allocator fragmentation so the 5-seed eval that
                # follows a long HPO search does not OOM-hard-kill the worker on
                # the big corpora (paired with the post-HPO empty_cache reclaim).
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                # Override the results/asset S3 prefix in the container (takes
                # precedence over the shipped .env). Both results and the
                # _models/_data tarball fetch use this prefix.
                **({"TOPICARENA_S3_PREFIX": s3_prefix} if s3_prefix else {}),
                **(extra_env or {}),
            },
            debugger_hook_config=False,
            disable_profiler=True,
        )
        estimator.fit(wait=False)
        return estimator.latest_training_job.name

    # Quota-aware submission: keep at most MAX_CONCURRENT_JOBS in flight so the
    # per-instance-type training-job limit is never exceeded mid-submission.
    # As jobs complete, more get submitted; total wall-clock ~= ceil(N/M) waves
    # of the slowest-in-wave.
    import time as _time
    print(f"\nSubmitting up to {MAX_CONCURRENT_JOBS} concurrent jobs (quota-aware)...")
    job_names = []
    pending = list(enumerate(jobs_plan))
    while pending:
        active = in_flight(job_names)
        slots = max(0, MAX_CONCURRENT_JOBS - active)
        if slots == 0:
            print(f"  {active}/{MAX_CONCURRENT_JOBS} in-flight, {len(pending)} pending. Waiting 2 min...")
            _time.sleep(120)
            continue
        for _ in range(min(slots, len(pending))):
            i, (ds, models, grp) = pending.pop(0)
            try:
                name = submit_one(i, ds, models, grp)
                job_names.append(name)
                print(f"  Submitted [{len(job_names)}/{len(jobs_plan)}] {name}")
            except Exception as e:
                # ResourceLimitExceeded means quota is tighter than we think --
                # requeue this item and wait; do NOT abort the whole run.
                msg = str(e)
                print(f"  SUBMIT FAILED (requeued): {msg[:200]}")
                pending.insert(0, (i, (ds, models, grp)))
                _time.sleep(120)
                break

    shutil.rmtree(source_dir)

    jobs_file = f"sagemaker_jobs_default_{version}.json"
    with open(jobs_file, "w") as f:
        json.dump(
            {
                "version": version,
                "fresh": fresh,
                "jobs": job_names,
                "plan": [(ds, grp) for ds, _, grp in jobs_plan],
            },
            f,
            indent=2,
        )
    print(f"\nJob info saved to {jobs_file}")
    print("Monitor: aws sagemaker list-training-jobs --name-contains topicarena-def")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-type", default=INSTANCE_TYPE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--version",
        default=VERSION,
        help=f"S3 version tag / results namespace (default: {VERSION}). "
        "Set a NEW value (e.g. v2_default_5seed_fixed) for the corrected rerun "
        "so the original paper results are not overwritten.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Recompute every combo, ignoring existing S3 results. Use this for "
        "the corrected rerun so stale (buggy) results are not reused.",
    )
    parser.add_argument(
        "--allow-overwrite-original",
        action="store_true",
        help=f"Permit --fresh to overwrite the original '{VERSION}' namespace. "
        "Without this, --fresh on the default version is refused to protect the "
        "paper's per-seed results.",
    )
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model subset to run (default: all 16). Use for a "
        "single-model run, e.g. --models TNTM.",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset subset to run (default: all 18). Use to "
        "backfill specific cells, e.g. --datasets AG_News.",
    )
    parser.add_argument(
        "--tntm-legacy",
        action="store_true",
        help="Set STREAM_TNTM_LEGACY=1 on the jobs so TNTM runs its ORIGINAL "
        "float64 + per-topic-loop + logsumexp numerics (for the old-vs-new TNTM "
        "A/B). Use a distinct --version so it lands in its own namespace.",
    )
    parser.add_argument(
        "--max-epochs",
        default=None,
        help="Override max_epochs for neural models via STREAM_MAX_EPOCHS (e.g. "
        "200 to run at a paper-native budget). Use a distinct --version.",
    )
    parser.add_argument(
        "--hpo",
        action="store_true",
        help="Run HPO (entry_point_native_hpo.py): --hpo-timeout budget per "
        "(dataset, model), then 5-seed eval on the best params. Uses the model's "
        "native objective (val_loss for neural, AIC/recon for classical).",
    )
    parser.add_argument("--hpo-timeout", type=int, default=5 * 60 * 60,
                        help="HPO budget per (dataset, model) in seconds (default 5h).")
    parser.add_argument("--hpo-criterion", default="native", choices=["native", "cv"],
                        help="HPO objective when --hpo is set: 'native' (each model's "
                        "own criterion: val_loss/AIC/recon) or 'cv' (C_V coherence via "
                        "a custom metric). Sets STREAM_HPO_CRITERION on the jobs.")
    parser.add_argument("--n-groups", type=int, default=N_GROUPS,
                        help="Model groups per dataset = jobs per dataset (parallelism).")
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="Run the embedding-swap ablation (entry_point_embedding.py): the "
        "default benchmark (no HPO, 5 seeds) with this sentence-transformers "
        "modeling encoder swapped in, e.g. 'mixedbread-ai/mxbai-embed-large-v1'. "
        "Defaults --models to KmeansTM,BERTopicTM,CTM and, unless --version is "
        "given, auto-namespaces results under 'emb_<sanitized-encoder>'.",
    )
    parser.add_argument("--no-skip-hpo", action="store_true",
                        help="Set STREAM_NO_SKIP_HPO=1 so no (dataset,model) combo is "
                        "skipped in HPO (use for fast variants like FASTopic @200).")
    parser.add_argument("--hpo-trial-cap-min", type=float, default=None,
                        help="Per-trial wall-clock cap in minutes for HPO "
                        "(STREAM_HPO_TRIAL_MAX_MIN). Optuna's --hpo-timeout is only "
                        "checked between trials and cannot stop a single long trial; "
                        "this bounds each trial so slow models (e.g. ECRTM on large "
                        "corpora) respect the overall HPO budget.")
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Override the S3 results/asset prefix in the jobs (e.g. "
        "TopicArena/Revised). Results land at <prefix>/<dataset>/<version>/<model>/; "
        "the entry point also fetches _models/_data tarballs from <prefix>/_models/, "
        "so those must exist there.",
    )
    args = parser.parse_args()

    version = args.version
    models_override = args.models.split(",") if args.models else None

    # Embedding-swap ablation: default to the 3 embedding-dependent models and, if
    # the user did not override --version, put results in their own per-encoder
    # namespace so they never touch the main 'default' results.
    if args.embedding_model:
        if models_override is None:
            models_override = ["KmeansTM", "BERTopicTM", "CTM"]
        if version == VERSION:
            sanitized = args.embedding_model.replace("/", "-")
            version = f"emb_{sanitized}"

    extra_env = {}
    if args.tntm_legacy:
        extra_env["STREAM_TNTM_LEGACY"] = "1"
    if args.max_epochs:
        extra_env["STREAM_MAX_EPOCHS"] = str(args.max_epochs)
    if args.no_skip_hpo:
        extra_env["STREAM_NO_SKIP_HPO"] = "1"
    if args.hpo_criterion == "cv":
        extra_env["STREAM_HPO_CRITERION"] = "cv"
    if args.hpo_trial_cap_min:
        extra_env["STREAM_HPO_TRIAL_MAX_MIN"] = str(args.hpo_trial_cap_min)
    extra_env = extra_env or None

    print("TopicArena SageMaker Launcher - DEFAULT RUN")
    print(f"Version:  {version}")
    print(f"Instance: {args.instance_type}")
    print(f"Fresh:    {args.fresh}")
    if args.s3_prefix:
        print(f"S3 prefix override: {args.s3_prefix}")
    if models_override:
        print(f"Models:   {models_override}")
    if extra_env:
        print(f"Env:      {extra_env}")
    # Guard: --fresh recomputes and OVERWRITES per-seed results in `version`.
    # Refuse to do that on the original namespace unless explicitly allowed, so a
    # forgotten --version cannot destroy the paper's default results.
    if version == VERSION and args.fresh and not args.allow_overwrite_original:
        raise SystemExit(
            f"\nREFUSING TO RUN: --fresh would overwrite the original results at "
            f"'{VERSION}'.\n"
            f"  For the corrected rerun, pass a NEW namespace, e.g.:\n"
            f"    python launch_default.py --fresh --version v2_default_5seed_fixed\n"
            f"  If you REALLY intend to overwrite '{VERSION}', add "
            f"--allow-overwrite-original.\n"
        )
    print()

    if args.fresh:
        # Corrected rerun: recompute everything, do not consult S3 completion.
        remaining_datasets = sorted(DATASETS)
        print(
            f"[FRESH] Recomputing all {len(remaining_datasets)} datasets "
            f"x {len(ALL_MODELS)} models x {len(SEEDS)} seeds."
        )
    else:
        session = boto_session()
        print("Checking completed work on S3...")
        completed = get_completed_combos(session, version)
        total = len(DATASETS) * len(ALL_MODELS)
        print(
            f"  Completed: {len(completed)}/{total} ({100*len(completed)/total:.0f}%)"
        )
        remaining_datasets = get_remaining_datasets(completed)
        print(f"  Remaining datasets: {len(remaining_datasets)}/{len(DATASETS)}")
        print(f"  Datasets: {remaining_datasets}")
        if not remaining_datasets:
            print("\nAll done!")
            return

    # Optional dataset subset (backfilling specific cells).
    if args.datasets:
        wanted = set(args.datasets.split(","))
        remaining_datasets = [d for d in remaining_datasets if d in wanted]
        print(f"  Dataset subset: {remaining_datasets}")

    njobs = 1 if models_override else args.n_groups
    print(f"\n  Jobs per dataset: {njobs}" + (f"  | HPO {args.hpo_timeout//3600}h/combo" if args.hpo else ""))
    launch_jobs(
        remaining_datasets,
        args.instance_type,
        version,
        fresh=args.fresh,
        dry_run=args.dry_run,
        models_override=models_override,
        extra_env=extra_env,
        s3_prefix=args.s3_prefix,
        hpo=args.hpo,
        hpo_timeout=args.hpo_timeout,
        n_groups=args.n_groups,
        embedding_model=args.embedding_model,
    )


if __name__ == "__main__":
    main()
