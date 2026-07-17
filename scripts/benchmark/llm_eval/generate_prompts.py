"""
Step 1: Generate JSONL prompt files for Bedrock Batch Inference.

Reads all topics from storage, generates JSONL prompt files for three tasks:
  - intruder_prompts_{version}.jsonl  (intruder detection task)
  - rating_prompts_{version}.jsonl    (topic coherence rating task)
  - fit_prompts_{version}.jsonl       (per-word binary fit task)

Each line is a Bedrock batch inference request (Anthropic Claude format) with a
unique recordId that encodes (dataset, version, model, seed, topic_idx, [extra]).

Usage:
    python generate_prompts.py --version v2_default_5seed
    python generate_prompts.py --version v2_default_5seed --seeds 42 84 126 168 210
    python generate_prompts.py --version v2_default_5seed --tasks intruder rating
"""

import io
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    S3_BUCKET,
    S3_PREFIX,
    LLM_JUDGE_MODEL_ID,
    boto_session,
)

DATASETS = [
    "BBC_News", "20Newsgroups", "Poliblogs", "UN", "WHO",
    "NeurIPS", "ACL", "Reuters", "NYT", "Spotify",
    "Reddit_GME", "IMDB", "AG_News", "Arxiv", "WikiText",
    "PubMed", "DBpedia", "Yahoo_Answers",
]

ALL_MODEL_NAMES = [
    "LDA", "NMFTM", "KmeansTM", "KmeansTM_PCA", "BERTopicTM",
    "ETM", "ProdLDA", "NeuralLDA", "CTM", "CTMNeg",
    "NSTM", "FASTopic", "ECRTM", "SawETM", "HyperMiner", "TNTM",
]

DEFAULT_SEEDS = [42, 84, 126, 168, 210]

# Number of top words to use for LLM evaluation
N_WORDS_INTRUDER = 10  # show 10 words + 1 intruder
N_WORDS_RATING = 10    # show 10 words for rating
N_INTRUDER_REPEATS = 5  # repeat intruder task 5 times per topic (different intruders)
N_WORDS_FIT_THEME = 10  # top-10 words establish the theme for fit task
N_WORDS_FIT_SCORE = 20  # score top-20 words as fit (1) / no-fit (0)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

INTRUDER_SYSTEM = (
    "You are an expert in evaluating topic models. Your task is to identify an "
    "intruder word that does not belong to a group of topically related words. "
    "You will be given a list of words where all but one belong to the same "
    "topic. Identify the single word that does not fit."
)

INTRUDER_USER = """Below is a list of {n_words} words. {n_words_minus_one} of these words belong to the same topic, and exactly ONE word is an intruder that was inserted from a different topic.

Words: {words_shuffled}

Which single word is the intruder? Respond with ONLY the intruder word, nothing else."""

RATING_SYSTEM = (
    "You are an expert in evaluating topic models. Your task is to assess the "
    "coherence of a topic — how well the words in the topic relate to each "
    "other and form a meaningful, interpretable theme."
)

RATING_USER = """Rate the coherence of the following topic on a scale of 1 to 5:

1 = Words are completely unrelated, no discernible theme
2 = Weak connection between some words, but mostly incoherent
3 = Moderate theme visible, but several words don't fit well
4 = Clear, interpretable theme with only minor outliers
5 = Highly coherent, all words clearly and strongly belong to one theme

Topic words: {words}

Respond with ONLY a JSON object in this exact format:
{{"rating": <integer 1-5>, "label": "<2-4 word topic label>", "reason": "<one sentence explanation>"}}"""


FIT_SYSTEM = (
    "You are an expert in evaluating topic models. Given a topic's theme "
    "(established by its top 10 most representative words), you will decide "
    "for each of a list of candidate words whether that word fits the topic "
    "or does not fit. A word fits if it is thematically related to the "
    "topic; a word does not fit if it is off-topic or unrelated."
)

FIT_USER = """Below is a topic and a list of {n_candidates} candidate words.

Topic theme (top {n_theme} words establishing the theme): {theme_words}

Candidate words to score (in rank order):
{numbered_candidates}

For each candidate word, decide whether it fits the topic's theme (1) or does not fit (0).

Respond with ONLY a JSON array of exactly {n_candidates} integers (each 0 or 1), in the same order as the candidate words.
Example for a list of 5 words: [1,1,0,1,0]"""


# ---------------------------------------------------------------------------
# Bedrock model-input builders
# ---------------------------------------------------------------------------
def build_model_input(model_id: str, system_text: str, user_text: str, max_tokens: int) -> dict:
    """Build a Bedrock batch inference modelInput payload.

    Anthropic Claude expects `system` as a top-level field and `messages[*].content`
    as a list of typed blocks. Nova/Amazon models use a different structure.
    """
    if "anthropic" in model_id:
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": 0,
            "system": system_text,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": user_text}]}
            ],
        }
    # Nova / Amazon format
    return {
        "messages": [
            {"role": "user", "content": [{"text": f"{system_text}\n\n{user_text}"}]}
        ],
        "inferenceConfig": {"max_new_tokens": max_tokens, "temperature": 0.0},
    }


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------
s3 = boto_session().client("s3")


def s3_read_csv(key):
    """Read a CSV from S3 into a list of lists."""
    import csv
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    content = obj["Body"].read().decode("utf-8")
    reader = csv.reader(io.StringIO(content))
    header = next(reader)
    rows = list(reader)
    return header, rows


def s3_list_keys(prefix):
    """List all keys under a prefix."""
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def load_topics_from_s3(dataset, version, model, seed):
    """Load topic words from S3. Returns list of lists of words."""
    key = f"{S3_PREFIX}/{dataset}/{version}/{model}/topics_seed{seed}.csv"
    try:
        header, rows = s3_read_csv(key)
        # Each column is a topic, each row is a word rank
        n_topics = len(header)
        topics = []
        for col_idx in range(n_topics):
            topic_words = [rows[row_idx][col_idx] for row_idx in range(len(rows))
                          if col_idx < len(rows[row_idx]) and rows[row_idx][col_idx]]
            topics.append(topic_words)
        return topics
    except Exception as e:
        print(f"  SKIP {key}: {e}")
        return None


# ---------------------------------------------------------------------------
# Prompt generation
# ---------------------------------------------------------------------------
def make_record_id(dataset, version, model, seed, topic_idx, extra=""):
    """Encode metadata into a unique record ID for Bedrock batch."""
    parts = [dataset, version, model, str(seed), str(topic_idx)]
    if extra:
        parts.append(extra)
    return "||".join(parts)


def parse_record_id(record_id):
    """Decode a record ID back into metadata."""
    parts = record_id.split("||")
    result = {
        "dataset": parts[0],
        "version": parts[1],
        "model": parts[2],
        "seed": int(parts[3]),
        "topic_idx": int(parts[4]),
    }
    if len(parts) > 5:
        result["extra"] = parts[5]
    return result


def generate_intruder_prompts(all_topics_by_key, model_id, rng):
    """Generate intruder detection prompts.

    For each topic, insert a random word from another topic and ask the LLM
    to identify it. Repeat N_INTRUDER_REPEATS times with different intruders.
    """
    prompts = []

    for (dataset, version, model, seed), topics in all_topics_by_key.items():
        if topics is None or len(topics) < 2:
            continue

        for topic_idx, topic_words in enumerate(topics):
            words = topic_words[:N_WORDS_INTRUDER]
            if len(words) < N_WORDS_INTRUDER:
                continue

            # Get candidate intruder words from other topics
            other_words = []
            for j, other_topic in enumerate(topics):
                if j != topic_idx:
                    other_words.extend(other_topic[:N_WORDS_INTRUDER])

            if not other_words:
                continue

            for repeat_idx in range(N_INTRUDER_REPEATS):
                intruder = rng.choice(other_words)

                # Insert intruder and shuffle
                words_with_intruder = words.copy() + [intruder]
                rng.shuffle(words_with_intruder)

                n_total = len(words_with_intruder)
                user_text = INTRUDER_USER.format(
                    n_words=n_total,
                    n_words_minus_one=n_total - 1,
                    words_shuffled=", ".join(words_with_intruder),
                )

                record_id = make_record_id(
                    dataset, version, model, seed, topic_idx,
                    extra=f"intruder_{repeat_idx}_{intruder}",
                )

                prompts.append({
                    "recordId": record_id,
                    "modelInput": build_model_input(
                        model_id, INTRUDER_SYSTEM, user_text, max_tokens=50
                    ),
                })

    return prompts


def generate_rating_prompts(all_topics_by_key, model_id):
    """Generate topic coherence rating prompts."""
    prompts = []

    for (dataset, version, model, seed), topics in all_topics_by_key.items():
        if topics is None:
            continue

        for topic_idx, topic_words in enumerate(topics):
            words = topic_words[:N_WORDS_RATING]
            if len(words) < 5:  # skip very short topics
                continue

            user_text = RATING_USER.format(words=", ".join(words))

            record_id = make_record_id(
                dataset, version, model, seed, topic_idx,
                extra="rating",
            )

            prompts.append({
                "recordId": record_id,
                "modelInput": build_model_input(
                    model_id, RATING_SYSTEM, user_text, max_tokens=200
                ),
            })

    return prompts


def generate_fit_prompts(all_topics_by_key, model_id):
    """Generate per-topic binary fit prompts.

    For each topic, show the top-10 words as 'theme', then ask the LLM to
    score each of the top-20 words 0/1 (does-not-fit / fits). One prompt per
    topic; the LLM returns a JSON array of N_WORDS_FIT_SCORE integers.
    """
    prompts = []

    for (dataset, version, model, seed), topics in all_topics_by_key.items():
        if topics is None:
            continue

        for topic_idx, topic_words in enumerate(topics):
            theme_words = topic_words[:N_WORDS_FIT_THEME]
            candidate_words = topic_words[:N_WORDS_FIT_SCORE]
            if len(theme_words) < N_WORDS_FIT_THEME or len(candidate_words) < 5:
                # Skip topics that don't have enough words for a meaningful fit score
                continue

            numbered = "\n".join(
                f"{i+1}. {w}" for i, w in enumerate(candidate_words)
            )
            user_text = FIT_USER.format(
                n_candidates=len(candidate_words),
                n_theme=len(theme_words),
                theme_words=", ".join(theme_words),
                numbered_candidates=numbered,
            )

            record_id = make_record_id(
                dataset, version, model, seed, topic_idx,
                extra="fit",
            )

            # max_tokens must cover a JSON array of 20 single-digit integers
            # with commas: "[1,1,0,1,...]" is ~42 chars ~= ~20 tokens. Leave
            # headroom for tokenizer variance and any trailing whitespace.
            prompts.append({
                "recordId": record_id,
                "modelInput": build_model_input(
                    model_id, FIT_SYSTEM, user_text, max_tokens=120
                ),
            })

    return prompts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="Benchmark version (e.g. v2_default_5seed)")
    parser.add_argument("--seeds", nargs="*", type=int, default=DEFAULT_SEEDS,
                        help="Seeds to evaluate")
    parser.add_argument("--model_id", default=LLM_JUDGE_MODEL_ID,
                        help="Bedrock model ID (cross-region inference profile, "
                             "e.g. us.anthropic.claude-opus-4-6-v1)")
    parser.add_argument("--output_dir", default="llm_prompts")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--tasks", nargs="+", default=["intruder", "rating", "fit"],
                        choices=["intruder", "rating", "fit"],
                        help="Which tasks to generate prompts for (default: all three)")
    parser.add_argument("--upload", action="store_true", default=True,
                        help="Upload JSONLs to S3 (default: True)")
    parser.add_argument("--no-upload", dest="upload", action="store_false",
                        help="Skip S3 upload (useful for dry runs)")
    args = parser.parse_args()

    rng = random.Random(args.random_seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load all topics from S3
    print(f"Loading topics for version={args.version}, seeds={args.seeds}")
    print(f"Model: {args.model_id}")
    print(f"Tasks: {args.tasks}")
    all_topics = {}
    for dataset in DATASETS:
        for model in ALL_MODEL_NAMES:
            for seed in args.seeds:
                key = (dataset, args.version, model, seed)
                topics = load_topics_from_s3(dataset, args.version, model, seed)
                if topics is not None:
                    all_topics[key] = topics

    print(f"Loaded topics for {len(all_topics)} (dataset, model, seed) combinations")

    outputs = []  # [(local_path, n)]

    if "intruder" in args.tasks:
        print("Generating intruder detection prompts...")
        intruder_prompts = generate_intruder_prompts(all_topics, args.model_id, rng)
        intruder_path = os.path.join(args.output_dir, f"intruder_prompts_{args.version}.jsonl")
        with open(intruder_path, "w") as f:
            for p in intruder_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"  {len(intruder_prompts)} intruder prompts -> {intruder_path}")
        outputs.append((intruder_path, len(intruder_prompts)))

    if "rating" in args.tasks:
        print("Generating topic rating prompts...")
        rating_prompts = generate_rating_prompts(all_topics, args.model_id)
        rating_path = os.path.join(args.output_dir, f"rating_prompts_{args.version}.jsonl")
        with open(rating_path, "w") as f:
            for p in rating_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"  {len(rating_prompts)} rating prompts -> {rating_path}")
        outputs.append((rating_path, len(rating_prompts)))

    if "fit" in args.tasks:
        print("Generating topic fit prompts...")
        fit_prompts = generate_fit_prompts(all_topics, args.model_id)
        fit_path = os.path.join(args.output_dir, f"fit_prompts_{args.version}.jsonl")
        with open(fit_path, "w") as f:
            for p in fit_prompts:
                f.write(json.dumps(p) + "\n")
        print(f"  {len(fit_prompts)} fit prompts -> {fit_path}")
        outputs.append((fit_path, len(fit_prompts)))

    # Upload to S3 for Bedrock
    if args.upload:
        for local_path, _ in outputs:
            s3_key = f"{S3_PREFIX}/_llm_eval/{os.path.basename(local_path)}"
            s3.upload_file(local_path, S3_BUCKET, s3_key)
            print(f"  Uploaded to s3://{S3_BUCKET}/{s3_key}")

    print(f"\nTotal prompts: {sum(n for _, n in outputs)}")


if __name__ == "__main__":
    main()
