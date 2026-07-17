"""
Step 3: Parse Bedrock Batch Inference output and compute LLM-based scores.

Reads the JSONL output from Bedrock for three tasks (intruder, rating, fit),
extracts per-topic judgments, aggregates to (dataset, model, seed) scores, and
computes correlations with the automated metrics.

Usage:
    python parse_results.py --version v2_default_5seed
    python parse_results.py --version v2_default_5seed --tasks intruder rating

Cross-account note: Bedrock-written output objects may be KMS-encrypted with a
key in the Bedrock account. Set TOPICARENA_BEDROCK_PROFILE so reads use those
credentials (this script reads output via bedrock_boto_session()).
"""

import io
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import (  # noqa: E402
    S3_BUCKET,
    S3_PREFIX,
    boto_session,
    bedrock_boto_session,
)

# Reads use the Bedrock-account session by default (needed for KMS-encrypted
# output in cross-account setups); falls back to the standard session otherwise.
s3 = bedrock_boto_session().client("s3")


def s3_list_keys(prefix):
    keys = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def s3_read_jsonl(key):
    """Read a JSONL file from S3."""
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    content = obj["Body"].read().decode("utf-8")
    return [json.loads(line) for line in content.strip().split("\n") if line.strip()]


def parse_record_id(record_id):
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


# ---------------------------------------------------------------------------
# Intruder parsing
# ---------------------------------------------------------------------------
def parse_intruder_results(output_records):
    """Parse intruder detection results.

    Matching rule: case-insensitive whole-word match. The prompt asks for ONLY the
    intruder word, but Claude sometimes responds with punctuation, quotes, or a
    short phrase; we tokenize the response and check whether the intruder word
    appears as any token, so those variants still count as correct.
    """
    import re

    def tokenize(text):
        return set(re.findall(r"[A-Za-z0-9_]+", text.lower()))

    rows = []
    for record in output_records:
        record_id = record.get("recordId", "")
        meta = parse_record_id(record_id)
        extra = meta.get("extra", "")

        if not extra.startswith("intruder_"):
            continue

        # Parse extra: "intruder_{repeat_idx}_{intruder_word}"
        parts = extra.split("_", 2)
        if len(parts) < 3:
            continue
        intruder_word = parts[2]

        # Extract LLM response
        try:
            output = record.get("modelOutput", {})
            if isinstance(output, str):
                output = json.loads(output)
            content = output.get("content", [{}])
            if isinstance(content, list):
                raw_text = content[0].get("text", "").strip()
            else:
                raw_text = str(content).strip()
        except Exception:
            raw_text = ""

        llm_answer = raw_text.lower()
        correct = intruder_word.lower() in tokenize(raw_text)

        rows.append({
            "dataset": meta["dataset"],
            "model": meta["model"],
            "seed": meta["seed"],
            "topic_idx": meta["topic_idx"],
            "intruder_word": intruder_word,
            "llm_answer": llm_answer,
            "raw_text": raw_text,
            "correct": correct,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Rating parsing
# ---------------------------------------------------------------------------
def parse_rating_results(output_records):
    """Parse topic rating results.

    Returns DataFrame with columns:
        dataset, model, seed, topic_idx, rating, label, reason
    """
    rows = []
    for record in output_records:
        record_id = record.get("recordId", "")
        meta = parse_record_id(record_id)
        extra = meta.get("extra", "")

        if extra != "rating":
            continue

        raw_text = ""
        try:
            output = record.get("modelOutput", {})
            if isinstance(output, str):
                output = json.loads(output)
            content = output.get("content", [{}])
            if isinstance(content, list):
                raw_text = content[0].get("text", "").strip()
            else:
                raw_text = str(content).strip()

            # Parse JSON response (handle markdown code-block wrapping)
            text = raw_text
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
                text = text.strip()

            parsed = json.loads(text)
            rating = int(parsed.get("rating", 0))
            label = parsed.get("label", "")
            reason = parsed.get("reason", "")
        except Exception:
            rating = None
            label = ""
            reason = ""

            # Regex fallback for malformed JSON (duplicate/typo keys). Claude
            # occasionally emits near-valid JSON that json.loads rejects.
            if raw_text:
                import re
                m_rating = re.search(r'"rating"\s*:\s*([1-5])', raw_text)
                if m_rating:
                    try:
                        rating = int(m_rating.group(1))
                    except (ValueError, TypeError):
                        rating = None
                m_label = re.search(r'"label"\s*:\s*"([^"]+)"', raw_text)
                if m_label:
                    label = m_label.group(1)
                m_reason = re.search(r'"reason"\s*:\s*"([^"]+)"', raw_text)
                if m_reason:
                    reason = m_reason.group(1)

        rows.append({
            "dataset": meta["dataset"],
            "model": meta["model"],
            "seed": meta["seed"],
            "topic_idx": meta["topic_idx"],
            "rating": rating,
            "label": label,
            "reason": reason,
            "raw_text": raw_text,
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fit parsing
# ---------------------------------------------------------------------------
def parse_fit_results(output_records):
    """Parse per-topic binary fit results.

    Each record's LLM output should be a JSON array of integers (0 or 1) of
    length N_WORDS_FIT_SCORE=20. Returns one row per topic with fit@{5,10,15,20}.
    Rows where parsing fails keep NaN fit_* columns but retain raw_text.
    """
    import re

    def parse_fit_array(text):
        """Extract a JSON array of 0/1 ints from possibly-noisy LLM output."""
        if not isinstance(text, str):
            return []
        t = text.strip()
        if "```" in t:
            parts = t.split("```")
            if len(parts) >= 2:
                t = parts[1]
                if t.startswith("json"):
                    t = t[4:]
                t = t.strip()
        try:
            obj = json.loads(t)
            if isinstance(obj, list):
                return [int(x) for x in obj if isinstance(x, (int, float))]
        except Exception:
            pass
        m = re.search(r"\[([^\[\]]*)\]", t)
        if not m:
            return []
        inside = m.group(1)
        nums = re.findall(r"-?\d+", inside)
        return [int(n) for n in nums]

    rows = []
    for record in output_records:
        record_id = record.get("recordId", "")
        meta = parse_record_id(record_id)
        extra = meta.get("extra", "")
        if extra != "fit":
            continue

        raw_text = ""
        try:
            output = record.get("modelOutput", {})
            if isinstance(output, str):
                output = json.loads(output)
            content = output.get("content", [{}])
            if isinstance(content, list):
                raw_text = content[0].get("text", "").strip()
            else:
                raw_text = str(content).strip()
        except Exception:
            raw_text = ""

        arr = parse_fit_array(raw_text)
        # Clamp stray values (e.g. -1 or 2) to {0,1} by treating non-1 as 0
        arr = [1 if v == 1 else 0 for v in arr]

        def _at(k):
            if len(arr) >= k:
                return sum(arr[:k]) / k
            return None

        rows.append({
            "dataset":     meta["dataset"],
            "model":       meta["model"],
            "seed":        meta["seed"],
            "topic_idx":   meta["topic_idx"],
            "fit_array":   arr,
            "n_parsed":    len(arr),
            "fit_at_5":    _at(5),
            "fit_at_10":   _at(10),
            "fit_at_15":   _at(15),
            "fit_at_20":   _at(20),
            "raw_text":    raw_text,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    # Deduplicate: if a topic was retried (e.g. after a first-pass truncation),
    # we'll see it twice. Prefer the row with the higher n_parsed (fully returned).
    df = df.sort_values("n_parsed", ascending=False)
    df = df.drop_duplicates(
        subset=["dataset", "model", "seed", "topic_idx"], keep="first"
    ).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def compute_intruder_scores(df_intruder):
    """Compute intruder detection accuracy per model per dataset."""
    if df_intruder.empty:
        return pd.DataFrame()

    return df_intruder.groupby(["dataset", "model", "seed"]).agg(
        llm_intruder_accuracy=("correct", "mean"),
        llm_intruder_n_trials=("correct", "count"),
    ).reset_index()


def compute_rating_scores(df_rating):
    """Compute average rating and label diversity per model per dataset."""
    if df_rating.empty:
        return pd.DataFrame()

    scores = df_rating.groupby(["dataset", "model", "seed"]).agg(
        llm_rating_mean=("rating", "mean"),
        llm_rating_std=("rating", "std"),
        llm_rating_n_topics=("rating", "count"),
    ).reset_index()

    label_div = df_rating.groupby(["dataset", "model", "seed"]).apply(
        lambda g: g["label"].nunique() / max(len(g), 1)
    ).reset_index(name="llm_label_diversity")

    return scores.merge(label_div, on=["dataset", "model", "seed"])


def compute_fit_scores(df_fit):
    """Aggregate per-topic fit scores up to (dataset, model, seed) level."""
    if df_fit.empty:
        return pd.DataFrame()

    return df_fit.groupby(["dataset", "model", "seed"]).agg(
        llm_fit_at_5=("fit_at_5", "mean"),
        llm_fit_at_10=("fit_at_10", "mean"),
        llm_fit_at_15=("fit_at_15", "mean"),
        llm_fit_at_20=("fit_at_20", "mean"),
        llm_fit_n_topics=("fit_at_20", "count"),
    ).reset_index()


def correlate_with_automated(llm_scores, version):
    """Load automated metrics and compute correlations with LLM scores."""
    try:
        keys = s3_list_keys(f"{S3_PREFIX}/_results/{version}_all")
        if not keys:
            print("No automated results found for correlation analysis")
            return None

        dfs = []
        for key in keys:
            if key.endswith(".csv"):
                obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
                dfs.append(pd.read_csv(io.BytesIO(obj["Body"].read())))

        if not dfs:
            return None

        df_auto = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"Could not load automated results: {e}")
        return None

    merge_keys = ["dataset", "model", "seed"]
    df_merged = llm_scores.merge(df_auto, on=merge_keys, how="inner")

    if df_merged.empty:
        print("No matching records for correlation")
        return None

    auto_metrics = [c for c in df_auto.columns if "@10" in c]
    llm_metrics = [c for c in llm_scores.columns if c.startswith("llm_")]

    correlations = {}
    for lm in llm_metrics:
        if lm.endswith("_n_trials") or lm.endswith("_n_topics"):
            continue
        for am in auto_metrics:
            valid = df_merged[[lm, am]].dropna()
            if len(valid) > 5:
                corr = valid[lm].corr(valid[am])
                correlations[f"{lm}_vs_{am}"] = round(corr, 4)

    return correlations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True)
    parser.add_argument("--output_dir", default="llm_results")
    parser.add_argument("--tasks", nargs="+",
                        default=["intruder", "rating", "fit"],
                        choices=["intruder", "rating", "fit"],
                        help="Which tasks to parse (default: all three)")
    parser.add_argument("--merge_existing", action="store_true", default=True,
                        help="Merge new results into the existing llm_scores CSV "
                             "on S3 instead of rewriting (default: True)")
    parser.add_argument("--no_merge_existing", dest="merge_existing",
                        action="store_false",
                        help="Overwrite the existing llm_scores CSV instead of merging")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Find and read Bedrock output. Bedrock writes per-chunk output directories
    # like "{task}_{version}_chunk{i}/"; we also match single-job and _retry layouts.
    print("Reading Bedrock output from S3...")

    def collect_task_records(task):
        records = []
        for subprefix in (
            f"{S3_PREFIX}/_llm_eval/output/{task}_{args.version}/",
            f"{S3_PREFIX}/_llm_eval/output/{task}_{args.version}_chunk",
            f"{S3_PREFIX}/_llm_eval/output/{task}_retry_{args.version}_chunk",
        ):
            for key in s3_list_keys(subprefix):
                if key.endswith(".jsonl") or key.endswith(".jsonl.out"):
                    records.extend(s3_read_jsonl(key))
        return records

    intruder_records = collect_task_records("intruder") if "intruder" in args.tasks else []
    print(f"  Intruder records: {len(intruder_records)}")
    rating_records = collect_task_records("rating") if "rating" in args.tasks else []
    print(f"  Rating records: {len(rating_records)}")
    fit_records = collect_task_records("fit") if "fit" in args.tasks else []
    print(f"  Fit records: {len(fit_records)}")

    print("Parsing results...")
    df_intruder = parse_intruder_results(intruder_records) if intruder_records else pd.DataFrame()
    df_rating   = parse_rating_results(rating_records)     if rating_records   else pd.DataFrame()
    df_fit      = parse_fit_results(fit_records)           if fit_records      else pd.DataFrame()

    print(f"  Intruder rows: {len(df_intruder)}")
    print(f"  Rating rows:   {len(df_rating)}")
    print(f"  Fit rows:      {len(df_fit)}")

    # Save raw parsed results
    if not df_intruder.empty:
        path = os.path.join(args.output_dir, f"intruder_parsed_{args.version}.csv")
        df_intruder.to_csv(path, index=False)
        print(f"  Saved: {path}")

    if not df_rating.empty:
        path = os.path.join(args.output_dir, f"rating_parsed_{args.version}.csv")
        df_rating.to_csv(path, index=False)
        print(f"  Saved: {path}")

        print("\n  Sample topic labels:")
        for _, row in df_rating.head(10).iterrows():
            print(f"    [{row['model']}] Topic {row['topic_idx']}: "
                  f"rating={row['rating']}, label=\"{row['label']}\"")

    if not df_fit.empty:
        path = os.path.join(args.output_dir, f"fit_parsed_{args.version}.csv")
        df_fit.to_csv(path, index=False)
        print(f"  Saved: {path}")
        ok = df_fit["fit_at_20"].notna().sum()
        if ok:
            print(f"  fit_at_20 valid on {ok}/{len(df_fit)} topics, "
                  f"mean={df_fit['fit_at_20'].mean():.3f}")

    # Aggregate
    print("\nComputing aggregated scores...")
    intruder_scores = compute_intruder_scores(df_intruder) if not df_intruder.empty else pd.DataFrame()
    rating_scores   = compute_rating_scores(df_rating)     if not df_rating.empty   else pd.DataFrame()
    fit_scores      = compute_fit_scores(df_fit)           if not df_fit.empty      else pd.DataFrame()

    merge_keys = ["dataset", "model", "seed"]
    new_scores = None
    for part in (intruder_scores, rating_scores, fit_scores):
        if part.empty:
            continue
        new_scores = part if new_scores is None else new_scores.merge(part, on=merge_keys, how="outer")

    if new_scores is None or new_scores.empty:
        print("No results to aggregate")
        return

    # Merge with the existing llm_scores CSV on S3 so single-task re-runs only
    # overwrite their own columns and keep the others intact.
    llm_scores = new_scores
    if args.merge_existing:
        s3_key = f"{S3_PREFIX}/_llm_eval/llm_scores_{args.version}.csv"
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)
            existing = pd.read_csv(io.BytesIO(obj["Body"].read()))
            print(f"  Found existing llm_scores on S3: {len(existing)} rows, "
                  f"{len(existing.columns)} cols")
            new_cols = [c for c in new_scores.columns if c not in merge_keys]
            existing_trim = existing.drop(columns=[c for c in new_cols if c in existing.columns])
            llm_scores = existing_trim.merge(new_scores, on=merge_keys, how="outer")
            print(f"  Merged -> {len(llm_scores)} rows, {len(llm_scores.columns)} cols")
        except s3.exceptions.NoSuchKey:
            print("  No existing llm_scores on S3 — writing fresh")
        except Exception as e:
            print(f"  WARNING: could not load existing llm_scores ({e}); writing fresh")

    path = os.path.join(args.output_dir, f"llm_scores_{args.version}.csv")
    llm_scores.to_csv(path, index=False)
    print(f"  Saved: {path}")

    # Upload artifacts to S3
    uploads = [
        f"intruder_parsed_{args.version}.csv",
        f"rating_parsed_{args.version}.csv",
        f"fit_parsed_{args.version}.csv",
        f"llm_scores_{args.version}.csv",
    ]
    for fname in uploads:
        local = os.path.join(args.output_dir, fname)
        if os.path.exists(local):
            s3_key = f"{S3_PREFIX}/_llm_eval/{fname}"
            s3.upload_file(local, S3_BUCKET, s3_key,
                           ExtraArgs={"ACL": "bucket-owner-full-control"})
            print(f"  Uploaded: s3://{S3_BUCKET}/{s3_key}")

    # Leaderboard
    print("\n" + "=" * 70)
    print("LLM-as-Judge Leaderboard (averaged across datasets)")
    print("=" * 70)

    if "llm_intruder_accuracy" in llm_scores.columns:
        print("\nIntruder Detection Accuracy (higher = more coherent):")
        lb = llm_scores.groupby("model")["llm_intruder_accuracy"].mean().sort_values(ascending=False)
        for model, score in lb.items():
            print(f"  {model:<20s} {score:.3f}")

    if "llm_rating_mean" in llm_scores.columns:
        print("\nTopic Coherence Rating (1-5, higher = better):")
        lb = llm_scores.groupby("model")["llm_rating_mean"].mean().sort_values(ascending=False)
        for model, score in lb.items():
            print(f"  {model:<20s} {score:.2f}")

    if "llm_label_diversity" in llm_scores.columns:
        print("\nLabel Diversity (higher = more distinct topics):")
        lb = llm_scores.groupby("model")["llm_label_diversity"].mean().sort_values(ascending=False)
        for model, score in lb.items():
            print(f"  {model:<20s} {score:.3f}")

    for k in (5, 10, 15, 20):
        col = f"llm_fit_at_{k}"
        if col in llm_scores.columns:
            print(f"\nTop-{k} fit fraction (higher = more words fit the topic theme):")
            lb = llm_scores.groupby("model")[col].mean().sort_values(ascending=False)
            for model, score in lb.items():
                print(f"  {model:<20s} {score:.3f}")

    # Correlation with automated metrics
    print("\n" + "=" * 70)
    print("Correlation with Automated Metrics (@10)")
    print("=" * 70)
    correlations = correlate_with_automated(llm_scores, args.version)
    if correlations:
        for lm in ["llm_intruder_accuracy", "llm_rating_mean", "llm_label_diversity"]:
            relevant = {k: v for k, v in correlations.items() if k.startswith(lm)}
            if relevant:
                print(f"\n{lm}:")
                for k, v in sorted(relevant.items(), key=lambda x: -abs(x[1])):
                    am = k.split("_vs_")[1]
                    print(f"  vs {am:<25s} r = {v:+.3f}")

        corr_path = os.path.join(args.output_dir, f"llm_correlations_{args.version}.json")
        with open(corr_path, "w") as f:
            json.dump(correlations, f, indent=2)
        print(f"\nCorrelations saved to {corr_path}")


if __name__ == "__main__":
    main()
