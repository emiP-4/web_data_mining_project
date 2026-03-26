"""
build_expanded_kg.py
──────────────────────
Unified pipeline to generate an Expanded Knowledge Graph, Ontology, and Alignments.

Outputs:
  1. kg_artifacts/expanded_kb.nt   — Full Knowledge Graph (Local + Wikidata expansion)
  2. kg_artifacts/ontology.ttl     — Formal classes and properties definition
  3. kg_artifacts/alignment.ttl    — owl:sameAs links between Local URIs and Wikidata
"""

import pandas as pd
import time
import urllib.parse
from pathlib import Path
from collections import defaultdict
from rdflib import Graph, URIRef, Literal, Namespace
from rdflib.namespace import RDF, RDFS, OWL, XSD
from SPARQLWrapper import SPARQLWrapper, JSON

# --- Configuration & Paths ---
INPUT_ENTITIES = Path("./data/samples/extracted_knowledge.csv")
INPUT_RELATIONS = Path("./data/samples/extracted_relations.csv")
OUTPUT_DIR = Path("./kg_artifacts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_KB = OUTPUT_DIR / "expanded_kb.nt"
OUT_ONTO = OUTPUT_DIR / "ontology.ttl"
OUT_ALIGN = OUTPUT_DIR / "alignment.ttl"

# --- Namespaces ---
OLY = Namespace("http://olympics.kg/entity/")
PROP = Namespace("http://olympics.kg/property/")
ONT = Namespace("http://olympics.kg/ontology#")
WD = Namespace("http://www.wikidata.org/entity/")

# --- Wikidata Settings ---
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
SPARQL_DELAY = 0.8
SKIP_PROPERTIES = {
    "P18", "P94", "P154", "P856", "P571", "P580", "P582", "P1036", 
    "P2671", "P3417", "P2397", "P8408", "P244", "P349", "P950", "P214", "P268", "P227"
}
EXPAND_PROPERTIES = ["P166", "P1344", "P641", "P1532", "P276", "P664", "P17", "P286", "P463"]

# --- Explicit Alignments (Local text to WD QID) ---
# This maps the clean local names to their WD equivalents
ALIGNMENT_MAP = {
    "International_Olympic_Committee": "Q23319",
    "Summer_Olympic_Games": "Q159821",
    "Winter_Olympic_Games": "Q82414",
    "Paris": "Q90",
    "Usain_Bolt": "Q1189",
    "Michael_Phelps": "Q32620",
    "Simone_Biles": "Q35600",
    "Mikaela_Shiffrin": "Q1484620",
    "Athletics": "Q542",
    "Swimming": "Q31920",
    "Gymnastics": "Q462",
    "United_States": "Q30",
    "Olympic_gold_medal": "Q152433"
}

sparql_client = SPARQLWrapper(WIKIDATA_SPARQL)
sparql_client.setReturnFormat(JSON)
sparql_client.addCustomHttpHeader("User-Agent", "OlympicsMiningLabBot/1.0")

def clean_uri(text: str) -> str:
    if not isinstance(text, str):
        return "Unknown"
    cleaned = text.strip().replace(" ", "_").replace('"', '').replace("'", "")
    return urllib.parse.quote(cleaned)

def run_sparql(query: str) -> list[tuple]:
    sparql_client.setQuery(query)
    triples = []
    try:
        results = sparql_client.query().convert()
        if not isinstance(results, dict):
            print(f"  [SPARQL ERROR] Unexpected results type: {type(results)}")
            return triples
        bindings = results.get("results", {}).get("bindings", [])
        for row in bindings:
            s = row.get("s", {}).get("value", "")
            p = row.get("p", {}).get("value", "")
            o_val  = row.get("o", {}).get("value", "")
            o_type = row.get("o", {}).get("type", "uri")
            
            if s and p and o_val:
                pid = p.split("/")[-1]
                if pid in SKIP_PROPERTIES:
                    continue
                triples.append((s, p, o_val, o_type))
    except Exception as e:
        print(f"  [SPARQL ERROR] {e}")
    return triples

def main():
    # Initialize our three graphs
    g_kb = Graph()
    g_onto = Graph()
    g_align = Graph()

    for g in [g_kb, g_onto, g_align]:
        g.bind("oly", OLY)
        g.bind("prop", PROP)
        g.bind("ont", ONT)
        g.bind("wd", WD)
        g.bind("owl", OWL)

    # ---------------------------------------------------------
    # 1. PROCESS LOCAL DATA & BUILD ONTOLOGY
    # ---------------------------------------------------------
    print("Loading local CSV data...")
    if INPUT_ENTITIES.exists() and INPUT_RELATIONS.exists():
        df_ent = pd.read_csv(INPUT_ENTITIES)
        df_rel = pd.read_csv(INPUT_RELATIONS)

        # Build local entities & dynamically populate Ontology classes
        for _, row in df_ent.iterrows():
            ent_uri = OLY[clean_uri(row['text'])]
            ent_type_uri = ONT[clean_uri(row['label'])]

            # Add to KB
            g_kb.add((ent_uri, RDFS.label, Literal(row['text'].strip(), lang="en")))
            g_kb.add((ent_uri, RDF.type, ent_type_uri))

            # Add to Ontology (Declare as OWL Class)
            g_onto.add((ent_type_uri, RDF.type, OWL.Class))
            g_onto.add((ent_type_uri, RDFS.label, Literal(row['label'].strip(), lang="en")))

        # Build local relations & dynamically populate Ontology properties
        for _, row in df_rel.iterrows():
            subj_uri = OLY[clean_uri(row['subject'])]
            obj_uri = OLY[clean_uri(row['object'])]
            rel_uri = PROP[clean_uri(row['relation'])]

            # Add to KB
            g_kb.add((subj_uri, rel_uri, obj_uri))

            # Add to Ontology (Declare as OWL ObjectProperty)
            g_onto.add((rel_uri, RDF.type, OWL.ObjectProperty))
            g_onto.add((rel_uri, RDFS.label, Literal(row['relation'].strip(), lang="en")))
    else:
        print(f"Warning: CSVs not found in {INPUT_ENTITIES.parent}. Proceeding with Wikidata only.")

    # ---------------------------------------------------------
    # 2. GENERATE ALIGNMENT FILE
    # ---------------------------------------------------------
    print("\nGenerating alignments (owl:sameAs)...")
    for local_name, qid in ALIGNMENT_MAP.items():
        local_uri = OLY[clean_uri(local_name)]
        wd_uri = WD[qid]
        g_align.add((local_uri, OWL.sameAs, wd_uri))
        g_kb.add((local_uri, OWL.sameAs, wd_uri)) # Optionally keep it in the main KB too

    # ---------------------------------------------------------
    # 3. EXPAND KB VIA WIKIDATA
    # ---------------------------------------------------------
    print("\nExpanding KB via Wikidata SPARQL...")
    qids_to_expand = list(ALIGNMENT_MAP.values())
    
    # 1-Hop Expansion
    for qid in qids_to_expand:
        print(f"  Fetching 1-hop for {qid}...")
        query = f"""SELECT ?s ?p ?o WHERE {{ BIND(wd:{qid} AS ?s) wd:{qid} ?p ?o . FILTER(STRSTARTS(STR(?p), "http://www.wikidata.org/prop/direct/")) }} LIMIT 500"""
        for s, p, o_val, o_type in run_sparql(query):
            obj = URIRef(o_val) if o_type == "uri" else Literal(o_val)
            g_kb.add((URIRef(s), URIRef(p), obj))
        time.sleep(SPARQL_DELAY)

    # Predicate-Controlled Expansion
    for prop in EXPAND_PROPERTIES:
        print(f"  Fetching predicate controlled for {prop}...")
        query = f"""SELECT ?s ?p ?o WHERE {{ ?s wdt:{prop} ?o . BIND(wdt:{prop} AS ?p) }} LIMIT 1000"""
        for s, p, o_val, o_type in run_sparql(query):
            obj = URIRef(o_val) if o_type == "uri" else Literal(o_val)
            g_kb.add((URIRef(s), URIRef(p), obj))
        time.sleep(SPARQL_DELAY)

    # ---------------------------------------------------------
    # 4. SAVE ARTIFACTS
    # ---------------------------------------------------------
    print("\nSaving artifacts...")
    
    # Save Expanded KB (N-Triples format)
    g_kb.serialize(destination=str(OUT_KB), format="nt")
    print(f"✔ Expanded KB saved to {OUT_KB} ({len(g_kb)} triples)")

    # Save Ontology (Turtle format)
    g_onto.serialize(destination=str(OUT_ONTO), format="turtle")
    print(f"✔ Ontology saved to {OUT_ONTO} ({len(g_onto)} triples)")

    # Save Alignment (Turtle format)
    g_align.serialize(destination=str(OUT_ALIGN), format="turtle")
    print(f"✔ Alignment saved to {OUT_ALIGN} ({len(g_align)} triples)")

if __name__ == "__main__":
    main()