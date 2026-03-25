"""
Lab Session 1: From the Unstructured Web to Structured Entities
Course: Web Mining & Semantics

Pipeline:
  Phase 1 - Web Crawling & Cleaning  → crawler_output.jsonl
  Phase 2 - Information Extraction   → extracted_knowledge.csv

Usage:
  python pipeline.py                     # uses default seed URLs (AI Research domain)
  python pipeline.py --urls urls.txt     # load seed URLs from a text file (one per line)
  python pipeline.py --domain energy     # switch to a preset domain
"""

import json
import csv
import time
import argparse
import urllib.robotparser
from pathlib import Path
from urllib.parse import urlparse
from datetime import datetime

import httpx
import trafilatura
import spacy
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MIN_WORD_COUNT = 500          # pages below this are discarded
REQUEST_DELAY  = 1.5          # seconds between requests (be polite)
REQUEST_TIMEOUT = 15          # seconds before giving up on a URL
OUTPUT_JSONL   = Path("crawler_output.jsonl")
OUTPUT_CSV     = Path("extracted_knowledge.csv")

# Preset seed URLs by domain (edit / extend freely)
DOMAIN_SEEDS = {
    "ai_research": [
        "https://en.wikipedia.org/wiki/Large_language_model",
        "https://en.wikipedia.org/wiki/Transformer_(deep_learning_architecture)",
        "https://en.wikipedia.org/wiki/OpenAI",
        "https://en.wikipedia.org/wiki/DeepMind",
        "https://en.wikipedia.org/wiki/Reinforcement_learning_from_human_feedback",
        "https://en.wikipedia.org/wiki/GPT-4",
        "https://en.wikipedia.org/wiki/Gemini_(language_model)",
        "https://en.wikipedia.org/wiki/Anthropic",
    ],
    "energy": [
        "https://en.wikipedia.org/wiki/Solar_power",
        "https://en.wikipedia.org/wiki/Wind_power",
        "https://en.wikipedia.org/wiki/Offshore_wind_power",
        "https://en.wikipedia.org/wiki/Hydrogen_economy",
        "https://en.wikipedia.org/wiki/Electric_vehicle",
        "https://en.wikipedia.org/wiki/Tesla,_Inc.",
        "https://en.wikipedia.org/wiki/International_Energy_Agency",
    ],
    "local_history": [
        "https://en.wikipedia.org/wiki/History_of_Paris",
        "https://en.wikipedia.org/wiki/Notre-Dame_de_Paris",
        "https://en.wikipedia.org/wiki/Eiffel_Tower",
        "https://en.wikipedia.org/wiki/Louvre",
        "https://en.wikipedia.org/wiki/Sacré-Cœur,_Paris",
    ],
}

# ─────────────────────────────────────────────
# PHASE 1 – CRAWLING & CLEANING
# ─────────────────────────────────────────────

def can_fetch(url: str) -> bool:
    """Check robots.txt before fetching."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = urllib.robotparser.RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp.can_fetch("*", url)
    except Exception:
        return True   # if robots.txt is unreachable, proceed cautiously


def fetch_and_clean(url: str, client: httpx.Client) -> dict | None:
    """
    Fetch a URL, extract main text with trafilatura, and return a record.
    Returns None if the page is not useful (< MIN_WORD_COUNT words).
    """
    if not can_fetch(url):
        print(f"  [BLOCKED by robots.txt] {url}")
        return None

    try:
        response = client.get(url, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except Exception as e:
        print(f"  [FETCH ERROR] {url} → {e}")
        return None

    html = response.text

    # trafilatura strips boilerplate (navbars, footers, ads, etc.)
    text = trafilatura.extract(
        html,
        include_comments=False,
        include_tables=True,
        no_fallback=False,
    )

    if not text:
        print(f"  [NO CONTENT] {url}")
        return None

    word_count = len(text.split())
    if word_count < MIN_WORD_COUNT:
        print(f"  [TOO SHORT] {url}  ({word_count} words)")
        return None

    print(f"  [OK] {url}  ({word_count} words)")
    return {
        "url":        url,
        "fetched_at": datetime.utcnow().isoformat(),
        "word_count": word_count,
        "text":       text,
    }


def crawl(seed_urls: list[str]) -> list[dict]:
    """Crawl all seed URLs and write results to JSONL."""
    records = []
    with httpx.Client(headers={"User-Agent": "WebMiningLabBot/1.0"}) as client:
        for url in seed_urls:
            print(f"Fetching: {url}")
            record = fetch_and_clean(url, client)
            if record:
                records.append(record)
                with open(OUTPUT_JSONL, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            time.sleep(REQUEST_DELAY)

    print(f"\nPhase 1 done. {len(records)} useful pages saved → {OUTPUT_JSONL}\n")
    return records


# ─────────────────────────────────────────────
# PHASE 2 – INFORMATION EXTRACTION
# ─────────────────────────────────────────────

# Entity labels we care about
TARGET_LABELS = {"PERSON", "ORG", "GPE", "LOC", "DATE", "PRODUCT", "EVENT", "WORK_OF_ART"}


def extract_entities(doc, url: str) -> list[dict]:
    """Extract named entities from a spaCy doc."""
    entities = []
    for ent in doc.ents:
        if ent.label_ in TARGET_LABELS:
            entities.append({
                "text":       ent.text.strip(),
                "label":      ent.label_,
                "start_char": ent.start_char,
                "end_char":   ent.end_char,
                "source_url": url,
            })
    return entities


def extract_relations(doc, url: str) -> list[dict]:
    """
    Extract subject–verb–object triples using dependency parsing.
    Looks for sentences that contain two or more named entities
    connected through a root verb.

    Pattern: nsubj ←── ROOT ──→ dobj  (or attr / prep / pobj chains)
    """
    relations = []
    ent_map = {tok.i: ent for ent in doc.ents for tok in ent}  # token_idx → entity

    for sent in doc.sents:
        ents_in_sent = [e for e in doc.ents if e.start >= sent.start and e.end <= sent.end]
        if len(ents_in_sent) < 2:
            continue

        for token in sent:
            if token.dep_ not in ("nsubj", "nsubjpass"):
                continue
            subject_ent = ent_map.get(token.i)
            if not subject_ent:
                continue

            head = token.head   # the governing verb
            relation_text = head.lemma_

            # look for an object attached to the same head
            for child in head.children:
                if child.dep_ in ("dobj", "attr", "pobj", "nsubjpass", "appos"):
                    object_ent = ent_map.get(child.i)
                    if not object_ent or object_ent == subject_ent:
                        continue
                    if subject_ent.label_ in TARGET_LABELS and object_ent.label_ in TARGET_LABELS:
                        relations.append({
                            "subject":       subject_ent.text.strip(),
                            "subject_label": subject_ent.label_,
                            "relation":      relation_text,
                            "object":        object_ent.text.strip(),
                            "object_label":  object_ent.label_,
                            "sentence":      sent.text.strip()[:200],
                            "source_url":    url,
                        })

    return relations


def run_ner_pipeline(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Run spaCy NER + relation extraction over all crawled pages."""
    print("Loading spaCy model (en_core_web_trf) …")
    nlp = spacy.load("en_core_web_trf")

    all_entities = []
    all_relations = []

    for i, record in enumerate(records, 1):
        url = record["url"]
        print(f"  [{i}/{len(records)}] NER on {url}")

        # Process in chunks to avoid memory issues on long documents
        text = record["text"]
        chunks = [text[j:j+50000] for j in range(0, len(text), 50000)]

        for chunk in chunks:
            doc = nlp(chunk)
            all_entities.extend(extract_entities(doc, url))
            all_relations.extend(extract_relations(doc, url))

    return all_entities, all_relations


def save_csv(entities: list[dict], relations: list[dict]):
    """Save entities and relations to CSV."""
    # Entities sheet
    df_ent = pd.DataFrame(entities).drop_duplicates(subset=["text", "label", "source_url"])
    df_ent.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nEntities saved → {OUTPUT_CSV}  ({len(df_ent)} rows)")

    # Relations sheet (separate file)
    relations_path = Path("extracted_relations.csv")
    df_rel = pd.DataFrame(relations)
    df_rel.to_csv(relations_path, index=False, encoding="utf-8")
    print(f"Relations saved → {relations_path}  ({len(df_rel)} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Web Mining Lab 1 Pipeline")
    parser.add_argument("--domain", default="ai_research",
                        choices=list(DOMAIN_SEEDS.keys()),
                        help="Preset domain to crawl")
    parser.add_argument("--urls", type=str, default=None,
                        help="Path to a text file with one seed URL per line")
    args = parser.parse_args()

    # Clear output files
    OUTPUT_JSONL.unlink(missing_ok=True)

    # Decide seed URLs
    if args.urls:
        with open(args.urls) as f:
            seeds = [line.strip() for line in f if line.strip()]
    else:
        seeds = DOMAIN_SEEDS[args.domain]

    print(f"=== PHASE 1: Crawling {len(seeds)} seed URLs (domain: {args.domain}) ===\n")
    records = crawl(seeds)

    if not records:
        print("No usable pages found. Check your URLs or network connection.")
        return

    print("=== PHASE 2: Information Extraction ===\n")
    entities, relations = run_ner_pipeline(records)
    save_csv(entities, relations)

    print("\n=== DONE ===")
    print(f"  crawler_output.jsonl    → {len(records)} pages")
    print(f"  extracted_knowledge.csv → {len(set(e['text'] for e in entities))} unique entities")
    print(f"  extracted_relations.csv → {len(relations)} candidate triples")


if __name__ == "__main__":
    main()
