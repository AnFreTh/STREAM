"""
Prepare additional benchmark datasets for TopicArena.

Downloads datasets from Hugging Face and saves them in the STREAM format
(parquet with 'text' and 'labels' columns + info pkl).

Datasets:
- AG News: 120K news articles, 4 categories
- Arxiv Abstracts: scientific paper abstracts with subject categories
- Yelp Reviews: restaurant/business reviews with star ratings
- WikiText: Wikipedia articles (no labels)
- Europarl: European Parliament proceedings (no labels)
- PubMed: biomedical abstracts
"""

import os
import pickle
import pandas as pd
from datasets import load_dataset

BASE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "stream_topic", "stream_topic_data", "preprocessed_datasets"
)


def save_dataset(df: pd.DataFrame, name: str, language: str = "en"):
    """Save a dataset in STREAM format."""
    save_dir = os.path.join(BASE_DIR, name)
    os.makedirs(save_dir, exist_ok=True)

    assert "text" in df.columns and "labels" in df.columns

    # Drop empty/short texts
    df = df[df["text"].str.split().str.len() >= 10].reset_index(drop=True)

    df[["text", "labels"]].to_parquet(os.path.join(save_dir, f"{name}.parquet"), index=False)

    info = {
        "name": name,
        "language": language,
        "preprocessing_steps": {
            "remove_stopwords": False,
            "lowercase": False,
            "remove_punctuation": False,
            "remove_numbers": False,
            "lemmatize": False,
            "stem": False,
            "expand_contractions": False,
            "remove_html_tags": False,
            "remove_special_chars": False,
            "remove_accents": False,
            "custom_stopwords": set(),
            "detokenize": False,
        },
    }
    with open(os.path.join(save_dir, f"{name}_info.pkl"), "wb") as f:
        pickle.dump(info, f)

    print(f"✓ {name}: {len(df)} docs saved to {save_dir}")
    print(f"  Labels: {df['labels'].nunique()} unique")
    print(f"  Avg doc length: {df['text'].str.split().str.len().mean():.0f} words")
    print()


def prepare_ag_news(max_docs: int = 120000):
    """AG News: 4-class news classification dataset."""
    print("Downloading AG News...")
    ds = load_dataset("fancyzhx/ag_news", split="train")
    label_map = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}
    df = pd.DataFrame({
        "text": ds["text"],
        "labels": [label_map[l] for l in ds["label"]],
    })
    if len(df) > max_docs:
        df = df.sample(max_docs, random_state=42).reset_index(drop=True)
    save_dataset(df, "AG_News")


def prepare_arxiv(max_docs: int = 50000):
    """Arxiv abstracts with subject area labels."""
    print("Downloading Arxiv Abstracts...")
    ds = load_dataset("ccdv/arxiv-classification", split="train")
    df = pd.DataFrame({
        "text": ds["text"],
        "labels": ds["label"],
    })
    if df["labels"].dtype == "int64":
        label_names = ds.features["label"].names if hasattr(ds.features["label"], "names") else None
        if label_names:
            df["labels"] = df["labels"].map(lambda x: label_names[x])
    if len(df) > max_docs:
        df = df.sample(max_docs, random_state=42).reset_index(drop=True)
    save_dataset(df, "Arxiv")



def prepare_wikitext(max_docs: int = 30000):
    """WikiText-103: Wikipedia articles (paragraph-level, no labels)."""
    print("Downloading WikiText-103...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")

    texts = []
    current = []
    for line in ds["text"]:
        line = line.strip()
        if not line:
            if current:
                merged = " ".join(current)
                if len(merged.split()) >= 50:
                    texts.append(merged)
                current = []
        elif line.startswith("="):
            if current:
                merged = " ".join(current)
                if len(merged.split()) >= 50:
                    texts.append(merged)
                current = []
        else:
            current.append(line)

    if current:
        merged = " ".join(current)
        if len(merged.split()) >= 50:
            texts.append(merged)

    df = pd.DataFrame({
        "text": texts,
        "labels": ["wikipedia"] * len(texts),
    })
    if len(df) > max_docs:
        df = df.sample(max_docs, random_state=42).reset_index(drop=True)
    save_dataset(df, "WikiText")


def prepare_europarl(max_docs: int = 30000):
    """European Parliament proceedings (English)."""
    print("Downloading Europarl...")
    try:
        ds = load_dataset("europarl_bilingual", lang1="en", lang2="fr", split="train")
        texts = [item["en"] for item in ds["translation"]]
    except Exception:
        print("  Trying alternative Europarl source...")
        ds = load_dataset("Helsinki-NLP/europarl", "en", split="train", trust_remote_code=True)
        texts = ds["text"]

    # Merge short utterances into longer documents
    merged_texts = []
    buffer = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        buffer.append(t)
        if len(" ".join(buffer).split()) >= 100:
            merged_texts.append(" ".join(buffer))
            buffer = []

    if buffer and len(" ".join(buffer).split()) >= 50:
        merged_texts.append(" ".join(buffer))

    df = pd.DataFrame({
        "text": merged_texts,
        "labels": ["europarl"] * len(merged_texts),
    })
    if len(df) > max_docs:
        df = df.sample(max_docs, random_state=42).reset_index(drop=True)
    save_dataset(df, "Europarl")


def prepare_pubmed(max_docs: int = 50000):
    """PubMed abstracts."""
    print("Downloading PubMed Abstracts...")
    try:
        ds = load_dataset("ccdv/pubmed-summarization", "document", split="train")
        df = pd.DataFrame({
            "text": ds["article"],
            "labels": ["pubmed"] * len(ds),
        })
    except Exception:
        print("  Trying PubMed QA as fallback...")
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        texts = []
        for ctx in ds["context"]:
            if isinstance(ctx, dict) and "contexts" in ctx:
                texts.append(" ".join(ctx["contexts"]))
            elif isinstance(ctx, list):
                texts.append(" ".join(ctx))
            else:
                texts.append(str(ctx))
        df = pd.DataFrame({
            "text": texts,
            "labels": ["pubmed"] * len(texts),
        })

    if len(df) > max_docs:
        df = df.sample(max_docs, random_state=42).reset_index(drop=True)
    save_dataset(df, "PubMed")


if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True)

    print("=" * 60)
    print("Preparing benchmark datasets for TopicArena")
    print("=" * 60)
    print()

    prepare_ag_news()
    prepare_arxiv()
    prepare_wikitext()
    prepare_europarl()
    prepare_pubmed()

    print("=" * 60)
    print("Done! All datasets saved to:")
    print(f"  {BASE_DIR}")
    print()
    print("Available datasets:")
    for d in sorted(os.listdir(BASE_DIR)):
        path = os.path.join(BASE_DIR, d)
        if os.path.isdir(path):
            parquet = os.path.join(path, f"{d}.parquet")
            if os.path.exists(parquet):
                df = pd.read_parquet(parquet)
                print(f"  {d}: {len(df)} docs, {df['labels'].nunique()} labels")
