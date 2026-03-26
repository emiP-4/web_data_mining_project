"""
src/rag/sparql_rag.py
---------------------
SPARQL-generation RAG pipeline:
  1. Load an RDF graph from a TTL/NT file.
  2. Build a compact schema summary (prefixes, predicates, classes, sample triples).
  3. Ask a local Ollama LLM to translate a natural-language question into SPARQL.
  4. Execute the query via rdflib; attempt one auto-repair on failure.
  5. (Baseline) Answer the same question without any KG, for comparison.

Project layout assumed
----------------------
  kg_artifacts/ontology.ttl   ← default knowledge graph
  src/rag/sparql_rag.py       ← this file
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import List, Tuple

import requests
from rdflib import Graph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

KG_FILE: Path = PROJECT_ROOT / "kg_artifacts" / "ontology.ttl"
OLLAMA_URL: str = "http://localhost:11434/api/generate"
OLLAMA_MODEL: str = "llama3:8b"

MAX_PREDICATES: int = 80
MAX_CLASSES: int = 40
SAMPLE_TRIPLES: int = 20

# ---------------------------------------------------------------------------
# 0) Utility – call local Ollama
# ---------------------------------------------------------------------------

def ask_local_llm(user_prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Send *user_prompt* to a local Ollama model and return the response text."""
    payload = {
        "model": model,
        "prompt": user_prompt,
        "stream": False,
    }
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=None)
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach Ollama at {OLLAMA_URL}. Is the server running?"
        ) from exc

    if resp.status_code != 200:
        raise RuntimeError(
            f"Ollama API error {resp.status_code}: {resp.text}"
        )
    return resp.json().get("response", "")


# ---------------------------------------------------------------------------
# 1) Load RDF graph
# ---------------------------------------------------------------------------

def load_graph(kg_file: Path = KG_FILE) -> Graph:
    """Parse *kg_file* (Turtle or N-Triples) and return an rdflib Graph."""
    if not kg_file.exists():
        raise FileNotFoundError(f"KG file not found: {kg_file}")

    fmt = "nt" if kg_file.suffix == ".nt" else "ttl"
    g = Graph()
    g.parse(str(kg_file), format=fmt)
    print(f"[graph] Loaded {len(g):,} triples from {kg_file.name}.")
    return g


# ---------------------------------------------------------------------------
# 2) Schema summary helpers
# ---------------------------------------------------------------------------

def _prefix_block(g: Graph) -> str:
    """Return PREFIX declarations for all namespaces in *g*, plus common defaults."""
    defaults = {
        "rdf":  "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
        "xsd":  "http://www.w3.org/2001/XMLSchema#",
        "owl":  "http://www.w3.org/2002/07/owl#",
    }
    ns_map: dict[str, str] = {p: str(ns) for p, ns in g.namespace_manager.namespaces()}
    for prefix, uri in defaults.items():
        ns_map.setdefault(prefix, uri)

    # Fix: build lines AFTER the loop, not inside it
    lines = [f"PREFIX {p}: <{ns}>" for p, ns in ns_map.items()]
    return "\n".join(sorted(lines))


def _distinct_predicates(g: Graph, limit: int = MAX_PREDICATES) -> List[str]:
    q = f"SELECT DISTINCT ?p WHERE {{ ?s ?p ?o . }} LIMIT {limit}"
    return [str(row[0]) for row in g.query(q)]


def _distinct_classes(g: Graph, limit: int = MAX_CLASSES) -> List[str]:
    q = f"SELECT DISTINCT ?cls WHERE {{ ?s a ?cls . }} LIMIT {limit}"
    return [str(row[0]) for row in g.query(q)]


def _sample_triples(g: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
    q = f"SELECT ?s ?p ?o WHERE {{ ?s ?p ?o . }} LIMIT {limit}"
    return [(str(r.s), str(r.p), str(r.o)) for r in g.query(q)]


def build_schema_summary(g: Graph) -> str:
    """Produce a compact text description of the graph's schema for the LLM."""
    prefixes   = _prefix_block(g)
    predicates = _distinct_predicates(g)
    classes    = _distinct_classes(g)
    samples    = _sample_triples(g)

    pred_block   = "\n".join(f"  - {p}" for p in predicates)
    class_block  = "\n".join(f"  - {c}" for c in classes)
    sample_block = "\n".join(f"  - {s}  {p}  {o}" for s, p, o in samples)

    return (
        f"{prefixes}\n\n"
        f"# Predicates (up to {MAX_PREDICATES})\n{pred_block}\n\n"
        f"# Classes / rdf:type (up to {MAX_CLASSES})\n{class_block}\n\n"
        f"# Sample triples (up to {SAMPLE_TRIPLES})\n{sample_block}"
    ).strip()


# ---------------------------------------------------------------------------
# 3) NL → SPARQL generation
# ---------------------------------------------------------------------------

_SPARQL_SYSTEM = """You are a SPARQL generator. Convert the user QUESTION into a valid SPARQL 1.1 SELECT query for the given RDF graph schema.

Rules:
- Use ONLY the IRIs/prefixes visible in SCHEMA SUMMARY.
- Prefer readable variable names in SELECT projections.
- Do NOT invent new predicates or classes.
- Return ONLY the SPARQL query inside a single fenced code block labeled ```sparql.
- No explanations or extra text outside the code block."""

_CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _make_generation_prompt(schema_summary: str, question: str) -> str:
    return (
        f"{_SPARQL_SYSTEM}\n\n"
        f"SCHEMA SUMMARY:\n{schema_summary}\n\n"
        f"QUESTION: {question}\n\n"
        "Return only the SPARQL query in a ```sparql code block."
    )


def _extract_sparql(raw_text: str) -> str:
    """Extract the first fenced code block; fall back to the whole text."""
    match = _CODE_BLOCK_RE.search(raw_text)
    return match.group(1).strip() if match else raw_text.strip()


def generate_sparql(question: str, schema_summary: str) -> str:
    """Ask the LLM to produce a SPARQL query for *question*."""
    raw = ask_local_llm(_make_generation_prompt(schema_summary, question))
    return _extract_sparql(raw)


# ---------------------------------------------------------------------------
# 4) SPARQL execution + self-repair
# ---------------------------------------------------------------------------

def run_sparql(g: Graph, query: str) -> Tuple[List[str], List[Tuple[str, ...]]]:
    """Execute *query* against *g*. Returns (variable names, list of row tuples)."""
    result = g.query(query)
    variables = [str(v) for v in result.vars]
    rows = [tuple(str(cell) for cell in row) for row in result]
    return variables, rows


_REPAIR_SYSTEM = """The previous SPARQL query failed. Using SCHEMA SUMMARY and ERROR MESSAGE, return a corrected SPARQL 1.1 SELECT query.

Rules:
- Use only known prefixes/IRIs from the schema.
- Keep the query as simple and robust as possible.
- Return only a single ```sparql code block with the corrected query."""


def _make_repair_prompt(
    schema_summary: str,
    question: str,
    bad_query: str,
    error_msg: str,
) -> str:
    return (
        f"{_REPAIR_SYSTEM}\n\n"
        f"SCHEMA SUMMARY:\n{schema_summary}\n\n"
        f"ORIGINAL QUESTION: {question}\n\n"
        f"BAD SPARQL:\n{bad_query}\n\n"
        f"ERROR MESSAGE:\n{error_msg}\n\n"
        "Return only the corrected SPARQL in a ```sparql code block."
    )


def repair_sparql(
    schema_summary: str,
    question: str,
    bad_query: str,
    error_msg: str,
) -> str:
    """Ask the LLM to fix a broken SPARQL query."""
    raw = ask_local_llm(_make_repair_prompt(schema_summary, question, bad_query, error_msg))
    return _extract_sparql(raw)


# ---------------------------------------------------------------------------
# 5) Orchestration
# ---------------------------------------------------------------------------

def answer_with_sparql_rag(
    g: Graph,
    schema_summary: str,
    question: str,
    try_repair: bool = True,
) -> dict:
    """
    Full RAG pipeline:
      - Generate SPARQL → execute → optionally repair on failure.

    Returns a dict with keys:
      query, vars, rows, repaired (bool), error (str | None)
    """
    query = generate_sparql(question, schema_summary)

    try:
        variables, rows = run_sparql(g, query)
        return {"query": query, "vars": variables, "rows": rows, "repaired": False, "error": None}
    except Exception as exc:
        first_error = str(exc)

    if not try_repair:
        return {"query": query, "vars": [], "rows": [], "repaired": False, "error": first_error}

    repaired_query = repair_sparql(schema_summary, question, query, first_error)
    try:
        variables, rows = run_sparql(g, repaired_query)
        return {"query": repaired_query, "vars": variables, "rows": rows, "repaired": True, "error": None}
    except Exception as exc2:
        return {"query": repaired_query, "vars": [], "rows": [], "repaired": True, "error": str(exc2)}


# ---------------------------------------------------------------------------
# 6) Baseline: direct LLM answer (no KG)
# ---------------------------------------------------------------------------

def answer_no_rag(question: str) -> str:
    """Answer *question* using only the LLM, without any knowledge-graph context."""
    user_prompt = f"Answer the following question as accurately as you can:\n\n{question}"
    return ask_local_llm(user_prompt)


# ---------------------------------------------------------------------------
# 7) CLI helpers
# ---------------------------------------------------------------------------

def pretty_print_result(result: dict) -> None:
    if result.get("error"):
        print(f"\n[Execution Error]\n{result['error']}")

    print(f"\n[SPARQL Query{'  ← repaired' if result['repaired'] else ''}]")
    print(result["query"])

    variables = result.get("vars", [])
    rows      = result.get("rows", [])

    if not rows:
        print("\n[No rows returned]")
        return

    print(f"\n[Results]  ({len(rows)} row{'s' if len(rows) != 1 else ''})")
    print(" | ".join(variables))
    print("-" * max(40, sum(len(v) + 3 for v in variables)))
    for row in rows[:20]:
        print(" | ".join(row))
    if len(rows) > 20:
        print(f"… (showing 20 of {len(rows)})")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Allow overriding the KG file path via CLI: python sparql_rag.py path/to/graph.ttl
    kg_path = Path(sys.argv[1]) if len(sys.argv) > 1 else KG_FILE

    graph  = load_graph(kg_path)
    schema = build_schema_summary(graph)

    while True:
        try:
            question = input("\nQuestion (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue

        print("\n--- Baseline (No RAG) ---")
        print(answer_no_rag(question))

        print("\n--- SPARQL-generation RAG ---")
        rag_result = answer_with_sparql_rag(graph, schema, question, try_repair=True)
        pretty_print_result(rag_result)