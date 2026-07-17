"""
Step 2: Submit JSONL prompt files to Amazon Bedrock Batch Inference.

Splits each prompt JSONL into chunks of <= MAX_BATCH_SIZE records and creates
one Bedrock batch inference job per chunk. Supports cross-account S3 access via
s3BucketOwner (set TOPICARENA_S3_BUCKET_OWNER and TOPICARENA_BEDROCK_PROFILE when
Bedrock runs in a different account than the S3 bucket).

Usage:
    python submit_bedrock.py --version v2_default_5seed
    python submit_bedrock.py --version v2_default_5seed --tasks intruder rating fit
"""

import os
import sys
import json
import time
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    S3_BUCKET,
    S3_PREFIX,
    S3_BUCKET_OWNER,
    AWS_REGION,
    LLM_JUDGE_MODEL_ID,
    BEDROCK_ROLE_ARN,
    boto_session,
    bedrock_boto_session,
)

MAX_BATCH_SIZE = 50_000  # Bedrock hard limit per batch job


def chunk_jsonl_locally(local_path, max_size=MAX_BATCH_SIZE):
    """Split a JSONL file into chunks of <= max_size lines.

    Writes chunk files next to the input (suffixed _chunkN.jsonl) and returns the
    list of chunk file paths. If the file fits in one chunk, returns [local_path].
    """
    with open(local_path, "r") as f:
        lines = f.readlines()
    n = len(lines)
    if n <= max_size:
        return [local_path]

    base, ext = os.path.splitext(local_path)
    chunk_paths = []
    for i in range(0, n, max_size):
        chunk_idx = i // max_size
        chunk_path = f"{base}_chunk{chunk_idx}{ext}"
        with open(chunk_path, "w") as f:
            f.writelines(lines[i:i + max_size])
        chunk_paths.append(chunk_path)
    return chunk_paths


def upload_chunk(s3, local_path, s3_key):
    """Upload a JSONL chunk to S3 with bucket-owner-full-control ACL for cross-account."""
    s3.upload_file(
        Filename=local_path,
        Bucket=S3_BUCKET,
        Key=s3_key,
        ExtraArgs={"ACL": "bucket-owner-full-control"},
    )
    return f"s3://{S3_BUCKET}/{s3_key}"


def submit_batch_job(bedrock, input_s3_uri, output_s3_uri, model_id, job_name,
                     role_arn, bucket_owner=None):
    """Submit a Bedrock batch inference job (optionally cross-account S3)."""
    input_cfg = {"s3Uri": input_s3_uri, "s3InputFormat": "JSONL"}
    output_cfg = {"s3Uri": output_s3_uri}
    if bucket_owner:
        input_cfg["s3BucketOwner"] = bucket_owner
        output_cfg["s3BucketOwner"] = bucket_owner

    response = bedrock.create_model_invocation_job(
        jobName=job_name,
        modelId=model_id,
        roleArn=role_arn,
        inputDataConfig={"s3InputDataConfig": input_cfg},
        outputDataConfig={"s3OutputDataConfig": output_cfg},
    )
    return response["jobArn"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--model_id", default=LLM_JUDGE_MODEL_ID)
    parser.add_argument(
        "--role_arn",
        default=BEDROCK_ROLE_ARN,
        help="IAM role ARN for Bedrock batch (default: TOPICARENA_BEDROCK_ROLE_ARN)",
    )
    parser.add_argument("--region", default=AWS_REGION)
    parser.add_argument(
        "--bucket_owner",
        default=S3_BUCKET_OWNER or None,
        help="S3 bucket owner account ID for cross-account access "
             "(default: TOPICARENA_S3_BUCKET_OWNER; omit for single-account)",
    )
    parser.add_argument("--input_dir", default="llm_prompts",
                        help="Local directory containing {task}_prompts_{version}.jsonl")
    parser.add_argument("--tasks", nargs="+", default=["intruder", "rating", "fit"],
                        help="Tasks to submit (default: intruder rating fit)")
    parser.add_argument("--max_batch_size", type=int, default=MAX_BATCH_SIZE,
                        help="Max records per Bedrock batch job (default 50000)")
    parser.add_argument("--dry_run", action="store_true",
                        help="Chunk and upload but don't create jobs")
    args = parser.parse_args()

    if not args.role_arn:
        parser.error(
            "No Bedrock role ARN. Pass --role_arn or set TOPICARENA_BEDROCK_ROLE_ARN "
            "in your environment / .env file."
        )

    # Bedrock session (may be a different account than the bucket in cross-account
    # setups); S3 session uses the standard configured profile / chain.
    bedrock = bedrock_boto_session(region=args.region).client("bedrock")
    s3 = boto_session().client("s3")

    print(f"Model: {args.model_id}")
    print(f"Role ARN: {args.role_arn}")
    owner_note = f" (owner acct {args.bucket_owner})" if args.bucket_owner else ""
    print(f"S3 bucket: s3://{S3_BUCKET}{owner_note}")

    all_jobs = []

    for task in args.tasks:
        local_path = os.path.join(args.input_dir, f"{task}_prompts_{args.version}.jsonl")
        if not os.path.exists(local_path):
            print(f"\nSKIP task={task}: file not found: {local_path}")
            continue

        print(f"\n=== task: {task} ===")
        chunk_paths = chunk_jsonl_locally(local_path, max_size=args.max_batch_size)
        print(f"  {len(chunk_paths)} chunk(s) (max {args.max_batch_size}/chunk)")

        for chunk_idx, chunk_path in enumerate(chunk_paths):
            with open(chunk_path, "r") as f:
                n_records = sum(1 for _ in f)

            timestamp = int(time.time())
            # Chunked file is stored in S3 under a chunks/ prefix only if we split
            if len(chunk_paths) > 1:
                input_key = f"{S3_PREFIX}/_llm_eval/chunks/{task}_{args.version}_chunk{chunk_idx}.jsonl"
            else:
                input_key = f"{S3_PREFIX}/_llm_eval/{task}_prompts_{args.version}.jsonl"

            input_s3_uri = upload_chunk(s3, chunk_path, input_key)
            print(f"  [chunk {chunk_idx}] {n_records} records -> {input_s3_uri}")

            output_prefix = f"{S3_PREFIX}/_llm_eval/output/{task}_{args.version}_chunk{chunk_idx}/"
            output_s3_uri = f"s3://{S3_BUCKET}/{output_prefix}"

            # Bedrock job names: must be <= 63 chars, alphanumerics + hyphens
            job_name = f"ta-{task}-{args.version.replace('_','-')}-c{chunk_idx}-{timestamp}"
            job_name = job_name[:63]

            if args.dry_run:
                print(f"  [chunk {chunk_idx}] DRY-RUN would submit '{job_name}' -> {output_s3_uri}")
                continue

            try:
                job_arn = submit_batch_job(
                    bedrock,
                    input_s3_uri=input_s3_uri,
                    output_s3_uri=output_s3_uri,
                    model_id=args.model_id,
                    job_name=job_name,
                    role_arn=args.role_arn,
                    bucket_owner=args.bucket_owner,
                )
                print(f"  [chunk {chunk_idx}] Job ARN: {job_arn}")
                all_jobs.append({
                    "task": task,
                    "chunk_idx": chunk_idx,
                    "n_records": n_records,
                    "job_name": job_name,
                    "job_arn": job_arn,
                    "input_s3_uri": input_s3_uri,
                    "output_s3_uri": output_s3_uri,
                })
            except Exception as e:
                print(f"  [chunk {chunk_idx}] FAILED: {e}")
                all_jobs.append({
                    "task": task,
                    "chunk_idx": chunk_idx,
                    "n_records": n_records,
                    "job_name": job_name,
                    "error": str(e),
                    "input_s3_uri": input_s3_uri,
                    "output_s3_uri": output_s3_uri,
                })

    # Save job info
    jobs_file = f"llm_jobs_{args.version}.json"
    with open(jobs_file, "w") as f:
        json.dump(all_jobs, f, indent=2)
    print(f"\nJob info saved to {jobs_file}")
    print("Monitor with: aws bedrock get-model-invocation-job "
          f"--job-identifier <JOB_ARN> --region {args.region}")


if __name__ == "__main__":
    main()
