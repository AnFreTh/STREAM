# TopicArena Benchmark Harness

Reproduction code for **TopicArena: A Comprehensive Benchmark for Topic Models**.
It trains 16 topic models on 18 datasets with 5 seeds, runs hyperparameter
optimization, computes 11 quality metrics at 4 top-*k* cutoffs, and (optionally)
an LLM-as-judge evaluation.

The harness runs both on a single machine and on AWS SageMaker. By default it
writes results to the **local filesystem**, so the core benchmark reproduces
with no AWS account. All environment-specific values (storage backend, S3
bucket, AWS region/profile, IAM roles, LLM model ID) are read from **environment
variables** with safe public defaults — no infrastructure identifiers are
hard-coded.

## 1. Setup

```bash
pip install -e .                       # installs the stream_topic library
pip install -r scripts/benchmark/sagemaker/requirements.txt
```

## 2. Configuration

Copy the example env file and fill in your values:

```bash
cp scripts/benchmark/.env.example scripts/benchmark/.env
$EDITOR scripts/benchmark/.env
```

| Variable | Purpose | Needed for |
|---|---|---|
| `TOPICARENA_STORAGE` | `local` (default) or `s3` | all runs |
| `TOPICARENA_LOCAL_DIR` | Output dir for local backend (default `scripts/benchmark/results/`) | local storage |
| `TOPICARENA_S3_BUCKET` | Bucket where results are written | S3 storage |
| `TOPICARENA_S3_PREFIX` | Key prefix under the bucket (default `TopicArena`) | S3 storage |
| `AWS_REGION` | AWS region (default `us-east-1`) | S3 / SageMaker / Bedrock |
| `AWS_PROFILE` | Named credential profile (optional; else default chain) | S3 / SageMaker / Bedrock |
| `TOPICARENA_SM_ROLE_ARN` | SageMaker execution role ARN | SageMaker launchers |
| `TOPICARENA_BEDROCK_ROLE_ARN` | Bedrock batch-inference role ARN | LLM evaluation |
| `TOPICARENA_LLM_MODEL_ID` | Judge model (default `us.anthropic.claude-opus-4-6-v1`) | LLM evaluation |
| `TOPICARENA_BEDROCK_PROFILE` | Bedrock account profile (cross-account only) | LLM evaluation |
| `TOPICARENA_S3_BUCKET_OWNER` | S3 bucket owner account ID (cross-account only) | LLM evaluation |

Values exported in your shell always take precedence over the `.env` file. The
`.env` file is git-ignored and never published; `.env.example` is the template.

> **Storage:** with the default `TOPICARENA_STORAGE=local`, results are written
> under `TOPICARENA_LOCAL_DIR` mirroring the layout
> `<dataset>/<version>/<model>/metrics_seed<seed>.json`. Set `TOPICARENA_STORAGE=s3`
> to write to `s3://$TOPICARENA_S3_BUCKET/$TOPICARENA_S3_PREFIX/` instead. Both
> backends are idempotent — completed combinations are skipped on re-run.

## 3. Run the benchmark (single machine)

Each script processes all (dataset, model) combinations and writes per-seed
metrics to S3. They are idempotent — already-completed combinations are skipped.

```bash
cd scripts/benchmark

python run_default_5seed.py     # default hyperparameters, 5 seeds  (main table)
python run_hpo_5seed.py         # native-objective HPO + 5-seed eval
python run_hpo_cv_5seed.py      # C_V-objective HPO + 5-seed eval
```

`bench_common.py` holds the shared configuration: the model roster
(`ALL_MODELS`), datasets (`DATASETS`), topic counts (`DATASET_TOPICS`),
preprocessing (`BENCHMARK_PREPROCESS`), seeds, and the evaluation pipeline.

## 4. Run on AWS SageMaker

The `sagemaker/` launchers fan the work out across training jobs (one instance
per dataset × model-group). Set `TOPICARENA_SM_ROLE_ARN` first.

```bash
cd scripts/benchmark/sagemaker

python launch_sagemaker.py --dry-run     # C_V-objective HPO; preview the plan
python launch_sagemaker.py               # submit
python launch_native_hpo.py              # native-objective HPO
```

`entry_point*.py` are the in-container scripts SageMaker invokes — they pull
model/data tarballs from S3 and call the same `bench_common` pipeline, so local
and SageMaker runs produce identical results.

## 5. LLM-as-judge evaluation (optional)

Requires `TOPICARENA_BEDROCK_ROLE_ARN` and an S3 storage backend. Uses Amazon
Bedrock Batch Inference with three tasks — **intruder detection**, **topic
rating** (1–5 + label), and **word fit** (per-word 0/1). The pipeline reads
topics that a benchmark run wrote to S3, so run the benchmark with
`TOPICARENA_STORAGE=s3` first.

```bash
cd scripts/benchmark/llm_eval

python generate_prompts.py --version v2_default_5seed --seeds 42 84 126 168 210
python submit_bedrock.py   --version v2_default_5seed        # chunk + submit batch jobs
python parse_results.py    --version v2_default_5seed        # score + correlate
```

If Bedrock runs in a different AWS account than the S3 bucket, also set
`TOPICARENA_BEDROCK_PROFILE` and `TOPICARENA_S3_BUCKET_OWNER` (see the table
above); the scripts pass `s3BucketOwner` and use bucket-owner-full-control ACLs
automatically.

## 6. Figures

```bash
cd scripts/benchmark
# reads CSVs from $TOPICARENA_RESULTS_DIR, writes PDFs/PNGs to $TOPICARENA_FIGURES_DIR
TOPICARENA_RESULTS_DIR=./results TOPICARENA_FIGURES_DIR=./figures \
  python generate_elo_hpo_plot.py
```

## What is *not* published

Run artifacts and secrets are git-ignored: `.env`, model checkpoints (`*.ckpt`),
cached embedding-model weights (`sagemaker/models/`), logs, and result CSVs/JSON.
