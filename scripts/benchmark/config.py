"""Central configuration for the TopicArena benchmark harness.

All environment-specific values (AWS account, S3 bucket, IAM roles, region,
credential profile, LLM judge model) are read from environment variables so the
same code runs unmodified in any environment. Public defaults are safe
placeholders; set your own values via a local ``.env`` file (see ``.env.example``)
or by exporting the variables before running.

The ``.env`` file is git-ignored and never published. To reproduce the benchmark
on your own infrastructure, copy ``.env.example`` to ``.env`` and fill in your
values, or export the same variables in your shell.
"""

import os
from pathlib import Path


def _load_dotenv():
    """Minimal .env loader (no external dependency).

    Looks for a ``.env`` file next to this module and one at the repo root,
    and populates os.environ for any key not already set. Existing environment
    variables always take precedence, so exported values win over the file.
    """
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path(__file__).resolve().parents[2] / ".env",  # repo root
    ]
    for env_path in candidates:
        if not env_path.is_file():
            continue
        for raw in env_path.read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


def _get(name, default):
    """Read an env var, falling back to a public default."""
    value = os.environ.get(name)
    return value if value not in (None, "") else default


# ---------------------------------------------------------------------------
# Storage backend
# ---------------------------------------------------------------------------
# "local" writes results to the local filesystem (no AWS needed) — this is the
# default so a fresh clone reproduces the benchmark out of the box. Set to "s3"
# to write to S3 instead.
STORAGE_BACKEND = _get("TOPICARENA_STORAGE", "local").lower()

# Root directory for the local backend (results land under $LOCAL_RESULTS_DIR/).
LOCAL_RESULTS_DIR = _get(
    "TOPICARENA_LOCAL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "results"),
)

# ---------------------------------------------------------------------------
# AWS / storage configuration
# ---------------------------------------------------------------------------
# Public defaults are placeholders. Override via .env or environment variables.
S3_BUCKET = _get("TOPICARENA_S3_BUCKET", "your-bucket")
S3_PREFIX = _get("TOPICARENA_S3_PREFIX", "TopicArena")

# AWS region and credential profile. AWS_PROFILE may be empty to use the
# default credential chain (env vars, instance role, etc.).
AWS_REGION = _get("AWS_REGION", "us-east-1")
AWS_PROFILE = os.environ.get("AWS_PROFILE") or None

# SageMaker execution role ARN (required only for SageMaker launchers).
SM_ROLE_ARN = _get("TOPICARENA_SM_ROLE_ARN", "")

# Bedrock batch-inference role ARN (required only for LLM-as-judge submission).
BEDROCK_ROLE_ARN = _get("TOPICARENA_BEDROCK_ROLE_ARN", "")

# Cross-account Bedrock support (optional). If Bedrock runs in a different AWS
# account than the S3 bucket, set BEDROCK_PROFILE to the Bedrock account's
# credential profile and S3_BUCKET_OWNER to the bucket owner's account ID. Leave
# both empty for a single-account setup (Bedrock uses AWS_PROFILE / the default
# chain, and no bucket-owner is passed).
BEDROCK_PROFILE = os.environ.get("TOPICARENA_BEDROCK_PROFILE") or None
S3_BUCKET_OWNER = _get("TOPICARENA_S3_BUCKET_OWNER", "")

# ---------------------------------------------------------------------------
# LLM-as-judge configuration
# ---------------------------------------------------------------------------
LLM_JUDGE_MODEL_ID = _get(
    "TOPICARENA_LLM_MODEL_ID", "us.anthropic.claude-opus-4-6-v1"
)


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------
def boto_session():
    """Return a boto3 Session honoring the configured profile and region.

    Falls back to the default credential chain when no profile is set, or when
    the named profile is unavailable (e.g. inside a SageMaker container that
    authenticates via its instance role rather than a local profile).
    """
    import os as _os
    import boto3
    from botocore.exceptions import ProfileNotFound

    if AWS_PROFILE:
        try:
            return boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
        except ProfileNotFound:
            # Profile named but unavailable (e.g. inside a SageMaker container that
            # authenticates via an instance role). Drop it from the environment so
            # the default credential chain isn't poisoned by AWS_PROFILE, then fall
            # back to the instance role / default chain.
            _os.environ.pop("AWS_PROFILE", None)
            _os.environ.pop("AWS_DEFAULT_PROFILE", None)
    return boto3.Session(region_name=AWS_REGION)


def bedrock_boto_session(region=None):
    """Return a boto3 Session for calling Bedrock.

    Uses TOPICARENA_BEDROCK_PROFILE when set (cross-account setups where Bedrock
    runs in a different account than the S3 bucket), otherwise falls back to
    AWS_PROFILE / the default credential chain.
    """
    import boto3
    from botocore.exceptions import ProfileNotFound

    region = region or AWS_REGION
    for profile in (BEDROCK_PROFILE, AWS_PROFILE):
        if profile:
            try:
                return boto3.Session(profile_name=profile, region_name=region)
            except ProfileNotFound:
                continue
    return boto3.Session(region_name=region)


def require(name):
    """Fetch a required config value, raising a clear error if it is unset.

    Use for values that have no safe public default (role ARNs, bucket).
    """
    value = globals().get(name)
    if not value or value == "your-bucket":
        raise RuntimeError(
            f"Config '{name}' is not set. Copy scripts/benchmark/.env.example to "
            f"scripts/benchmark/.env and fill in your values, or export the "
            f"corresponding environment variable."
        )
    return value
