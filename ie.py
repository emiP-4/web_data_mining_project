"""
Lab Session 1: From the Unstructured Web to Structured Entities
Course: Web Mining & Semantics

Phase 2 - Information Extraction  → extracted_knowledge.csv
                                   → extracted_relations.csv

Reads crawler_output.jsonl produced by crawl.py, then runs spaCy NER
and dependency-based relation extraction over every crawled page.

Usage:
  python ie.py                          # reads crawler_output.jsonl (default)
  python ie.py --input my_crawl.jsonl   # use a different JSONL input file
"""

import json
import argparse
from pathlib import Path

import spacy
import pandas as pd

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_JSONL    = Path("crawler_output.jsonl")
OUTPUT_CSV     = Path("extracted_knowledge.csv")
RELATIONS_CSV  = Path("extracted_relations.csv")

# Entity labels we care about — tuned for Olympics content
TARGET_LABELS = {
    "PERSON",       # athletes, coaches, IOC officials
    "ORG",          # National Olympic Committees, sports federations
    "GPE",          # countries competing
    "LOC",          # host cities, stadiums, venues
    "DATE",         # editions, record dates
    "EVENT",        # specific Games editions and sporting events
    "PRODUCT",      # equipment, brands (e.g. timing systems)
    "WORK_OF_ART",  # ceremonies, songs, emblems
}

# ─────────────────────────────────────────────
# INFORMATION EXTRACTION
# ─────────────────────────────────────────────

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

    Olympics-relevant examples:
      Usain Bolt  ──won──>  gold medal  (at 2016 Summer Olympics)
      Paris       ──host──> 2024 Summer Olympics
      IOC         ──award── host city
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


def load_records(jsonl_path: Path) -> list[dict]:
    """Load crawled records from a JSONL file produced by crawl.py."""
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


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
    """Save entities and relations to separate CSV files."""
    df_ent = pd.DataFrame(entities).drop_duplicates(subset=["text", "label", "source_url"])
    df_ent.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"\nEntities saved  → {OUTPUT_CSV}  ({len(df_ent)} rows)")

    df_rel = pd.DataFrame(relations)
    df_rel.to_csv(RELATIONS_CSV, index=False, encoding="utf-8")
    print(f"Relations saved → {RELATIONS_CSV}  ({len(df_rel)} rows)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Olympics Web Mining — Phase 2: Information Extraction")
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_JSONL,
        help=f"JSONL file produced by crawl.py (default: {INPUT_JSONL})",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print("Run crawl.py first to generate the JSONL file.")
        return

    print(f"=== PHASE 2: Information Extraction — reading {args.input} ===\n")
    records = load_records(args.input)
    print(f"Loaded {len(records)} pages from {args.input}\n")

    entities, relations = run_ner_pipeline(records)
    save_csv(entities, relations)

    print("\n=== DONE ===")
    print(f"  {OUTPUT_CSV}   → {len(set(e['text'] for e in entities))} unique entities")
    print(f"  {RELATIONS_CSV} → {len(relations)} candidate triples")


if __name__ == "__main__":
    main()
