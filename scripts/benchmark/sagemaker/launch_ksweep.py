"""Launch K-sensitivity sweep: default 5-seed runs at 3 bracketing K per dataset."""
import os, sys, json, shutil, tempfile
from pathlib import Path
import sagemaker, boto3
from sagemaker.pytorch import PyTorch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import S3_BUCKET, S3_PREFIX, require  # noqa

AWS_REGION="us-east-1"; SM_ROLE_ARN=require("SM_ROLE_ARN")
REPO=Path(__file__).resolve().parent.parent.parent.parent
SWEEP=json.load(open(os.environ.get("TOPICARENA_KSWEEP_FILE","/tmp/k_sweep.json")))
MAX_EPOCHS=os.environ.get("STREAM_MAX_EPOCHS","").strip()  # capped rerun if set

def build_source_dir():
    tmp=Path(tempfile.mkdtemp(prefix="ta_ksweep_"))
    shutil.copytree(REPO/"stream_topic", tmp/"stream_topic",
        ignore=shutil.ignore_patterns("*.pyc","__pycache__","*.egg-info","stream_topic_data"))
    shutil.copytree(REPO/"scripts"/"benchmark", tmp/"scripts"/"benchmark",
        ignore=shutil.ignore_patterns("*.pyc","__pycache__","*.log","*.ckpt",".env",
            "checkpoints","embeddings","lightning_logs","models","results"))
    (tmp/"scripts"/"__init__.py").touch()
    shutil.copy(REPO/"scripts"/"benchmark"/"sagemaker"/"entry_point_ksweep.py", tmp/"entry_point.py")
    shutil.copy(REPO/"scripts"/"benchmark"/"sagemaker"/"requirements.txt", tmp/"requirements.txt")
    return str(tmp)

def main():
    dry = "--dry-run" in sys.argv
    jobs=[(ds,k) for ds,ks in SWEEP.items() for k in ks]
    print(f"K-sweep: {len(jobs)} jobs (dataset,K), 16 models x 5 seeds each, defaults")
    if dry:
        for ds,k in jobs[:6]: print(f"  {ds} K={k}")
        print("  ..."); return
    session=boto3.Session(region_name=AWS_REGION)
    sm=sagemaker.Session(boto_session=session)
    src=build_source_dir(); print("source:",src)
    submitted=[]
    for i,(ds,k) in enumerate(jobs):
        est=PyTorch(entry_point="entry_point.py", source_dir=src, role=SM_ROLE_ARN,
            instance_type="ml.g5.8xlarge", instance_count=1, framework_version="2.3.0",
            py_version="py311", volume_size=100, max_run=2*24*3600,
            base_job_name=f"ta-ksweep{'-cap' if MAX_EPOCHS else ''}-{i}", sagemaker_session=sm,
            hyperparameters={"datasets":ds,"n_topics":k,"worker_id":f"k{k}_{ds[:6]}"},
            environment={"TOPICARENA_STORAGE":"s3","TOPICARENA_S3_BUCKET":S3_BUCKET,
                "TOPICARENA_S3_PREFIX":S3_PREFIX,"AWS_DEFAULT_REGION":AWS_REGION,
                "CUDA_MODULE_LOADING":"LAZY","FI_EFA_FORK_SAFE":"1","RDMAV_FORK_SAFE":"1",
                "TOKENIZERS_PARALLELISM":"false",
                **({"STREAM_MAX_EPOCHS":MAX_EPOCHS} if MAX_EPOCHS else {})},
            debugger_hook_config=False, disable_profiler=True)
        est.fit(wait=False)
        submitted.append({"job":est.latest_training_job.name,"dataset":ds,"K":k})
        print(f"  [{i+1}/{len(jobs)}] {est.latest_training_job.name} ({ds} K={k})")
    shutil.rmtree(src, ignore_errors=True)
    json.dump(submitted, open(REPO/"ksweep_jobs.json","w"), indent=2)
    print(f"submitted {len(submitted)} K-sweep jobs")

if __name__=="__main__": main()
