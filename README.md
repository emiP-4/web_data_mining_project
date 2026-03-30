# Olympics Knowledge Graph: From Unstructured Web to Semantic Reasoning

This project is a complete, end-to-end Web Mining and Semantics pipeline. It extracts unstructured data about the Olympic Games from the web, structures it into an RDF Knowledge Graph, aligns it with Linked Open Data (Wikidata), applies logical reasoning (SWRL), trains Knowledge Graph Embeddings (KGE), and enables natural language querying via a local LLM (RAG).

**Course:** Web Mining & Semantics  
**Author:** Emilie Poupat  
**License:** MIT  

---

## 🏗️ Pipeline Architecture

The project is divided into 6 distinct phases:

1. **Web Crawling & Cleaning:** Fetches Olympic-related Wikipedia pages using `httpx` and extracts clean text using `trafilatura`.
2. **Information Extraction (IE):** Uses spaCy's transformer model (`en_core_web_trf`) for Named Entity Recognition (NER) and dependency parsing to extract subject-predicate-object triples.
3. **KB Construction & Expansion:** Converts extracted triples into an RDF graph (`rdflib`), links entities to Wikidata (`owl:sameAs`), and expands the graph using SPARQL queries against the Wikidata endpoint.
4. **Semantic Reasoning:** Uses `owlready2` and the Pellet reasoner to infer new knowledge (e.g., classifying athletes into a `GoldMedalist` class) based on SWRL rules.
5. **Knowledge Graph Embeddings (KGE):** Prunes the graph and trains machine learning models (**TransE** and **RotatE**) using `pykeen` to learn low-dimensional vector representations of entities and relations.
6. **RAG (NL → SPARQL):** Uses a local LLM (`llama3:8b` via Ollama) to translate natural language questions into executable SPARQL 1.1 queries against the generated graph.

---

## ⚙️ Prerequisites & Installation

- **Python:** 3.9+ recommended
- **Ollama:** Required for the RAG pipeline. Install [Ollama](https://ollama.com/) and pull the Llama 3 model (`ollama run llama3:8b`).

1. **Clone the repository:**
   ```bash
   git clone (https://github.com/emiP-4/WDM_Project.git)
   cd WDM_Project
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Download the spaCy transformer model:**
   ```bash
   python -m spacy download en_core_web_trf
   ```

---

## 🚀 Usage Guide

Run the pipeline in the following order to ensure all artifacts are generated correctly.

### Phase 1: Crawling
Extracts raw text and saves it to `data/samples/crawler_output.jsonl`.

```bash
python crawl.py --domain summer
```

### Phase 2: Information Extraction
Runs NER and relation extraction, outputting `extracted_knowledge.csv` and `extracted_relations.csv`.

```bash
python ie.py
```

### Phase 3: Graph Building & Expansion
Constructs the RDF graph, aligns to Wikidata, and generates `expanded_kb.nt`, `ontology.ttl`, and `alignment.ttl`.

```bash
python build_expanded_kg.py
```

### Phase 4: SWRL Reasoning
Runs the Pellet reasoner to infer new triples based on predefined logical rules.

```bash
python src/reasoning/olympics_reasoner.py
python src/reasoning/family_reasoner.py
```

### Phase 5: Knowledge Graph Embeddings
Trains TransE and RotatE models, evaluates them (MRR, Hits@K), and generates a 2D t-SNE visualization of the entity embeddings.

```bash
python src/kge/step6_kge_training.py
python src/kge/visualise.py
```

> Check the `kge_results/` folder for `comparison.json` and `tsne_embeddings.png`.

### Phase 6: RAG Querying
Chat with the knowledge graph using natural language.

```bash
python src/rag/sparql_rag.py kg_artifacts/expanded_kb.nt
```
