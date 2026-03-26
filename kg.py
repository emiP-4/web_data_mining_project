"""
step2b_csv_to_rdf.py
──────────────────────
Converts extracted entities and relations (CSVs) into an initial RDF graph.
Theme: Olympic Games Knowledge Graph

Outputs:
  kg_artifacts/initial_graph.ttl — The starting Knowledge Graph in Turtle format

Usage:
  pip install rdflib pandas
  python src/kg/step2b_csv_to_rdf.py
"""

import pandas as pd
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, XSD
from pathlib import Path
import urllib.parse
import os

# 1. Setup Namespaces
OLY = Namespace("http://olympics.kg/entity/")
PROP = Namespace("http://olympics.kg/property/")
WD = Namespace("http://www.wikidata.org/entity/")

# Define file paths based on the project structure
INPUT_ENTITIES = Path("./data/extracted_knowledge.csv")
INPUT_RELATIONS = Path("./data/extracted_relations.csv")
OUTPUT_DIR = Path("./data/kg_artifacts")
OUTPUT_GRAPH = OUTPUT_DIR / "initial_graph.ttl"

def clean_uri(text: str) -> str:
    """Cleans text to be used as a valid URI component."""
    if not isinstance(text, str):
        return "Unknown"
    # Replace spaces with underscores and strip weird characters
    cleaned = text.strip().replace(" ", "_").replace('"', '').replace("'", "")
    # URL encode to ensure it's a valid URI
    return urllib.parse.quote(cleaned)

def build_initial_rdf():
    g = Graph()
    g.bind("oly", OLY)
    g.bind("prop", PROP)
    g.bind("wd", WD)
    g.bind("rdfs", RDFS)

    print("Loading CSVs...")
    if not INPUT_ENTITIES.exists() or not INPUT_RELATIONS.exists():
        print(f"Error: Could not find CSV files in {INPUT_ENTITIES.parent}.")
        print("Please run the extraction pipeline (Phase 1 & 2) first and ensure outputs are in the data/ folder.")
        return

    df_ent = pd.read_csv(INPUT_ENTITIES)
    df_rel = pd.read_csv(INPUT_RELATIONS)

    # 2. Add Entities to Graph
    print(f"Adding {len(df_ent)} entities to the graph...")
    for _, row in df_ent.iterrows():
        ent_uri = OLY[clean_uri(row['text'])]
        
        # Add label
        g.add((ent_uri, RDFS.label, Literal(row['text'].strip(), lang="en")))
        
        # Add type based on NER label (e.g., PERSON, GPE, ORG, SPORT)
        ent_type = clean_uri(row['label'])
        g.add((ent_uri, RDF.type, OLY[ent_type]))

    # 3. Add Relations to Graph
    print(f"Adding {len(df_rel)} relations to the graph...")
    for _, row in df_rel.iterrows():
        subj_uri = OLY[clean_uri(row['subject'])]
        obj_uri = OLY[clean_uri(row['object'])]
        
        # Create a property URI based on the verb/relation extracted
        rel_uri = PROP[clean_uri(row['relation'])]
        
        g.add((subj_uri, rel_uri, obj_uri))
        
        # Optional: Store the original sentence as provenance/evidence
        if 'sentence' in row and pd.notna(row['sentence']):
            g.add((subj_uri, PROP["extractedFromContext"], Literal(row['sentence'].strip())))

    # 4. Save the Graph
    # Ensure the output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    g.serialize(destination=str(OUTPUT_GRAPH), format="turtle")
    print(f"\n✔ Initial RDF graph saved to {OUTPUT_GRAPH} ({len(g)} triples)")

if __name__ == "__main__":
    build_initial_rdf()