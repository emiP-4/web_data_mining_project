"""
step6_kge_training.py
──────────────────────
Step 6: KGE prep, training, and evaluation.
Theme: Olympic Games Knowledge Graph

This script uses PyKEEN to:
  1. Load the expanded_kb.nt file.
  2. Split triples into Train / Test / Validation sets.
  3. Train a TransE model.
  4. Evaluate and save the model/embeddings.

Usage:
  pip install pykeen torch
  python step6_kge_training.py
"""

import json
from pathlib import Path
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import rdflib
import numpy as np

INPUT_KB = "./kg_artifacts/expanded_kb.nt"
OUTPUT_DIR = "olympics_kge_model"

from collections import Counter

from pykeen.triples import TriplesFactory

def run_kge():
    print(f"Loading triples from {INPUT_KB}...")
    try:
        g = rdflib.Graph()
        g.parse(INPUT_KB, format="nt")
        
        # 1. Convert to a 2D numpy array
        triples_array = np.array([[str(s), str(p), str(o)] for s, p, o in g])
        print(f"Initial raw triples: {len(triples_array)}")
        
        # 2. Iteratively prune the graph until it stabilizes
        iteration = 1
        while True:
            # Count current entities and relations
            all_entities = np.concatenate((triples_array[:, 0], triples_array[:, 2]))
            entity_counts = Counter(all_entities)
            relation_counts = Counter(triples_array[:, 1])
            
            # Keep only triples where both entities AND the relation appear at least twice
            valid_triples = [
                [s, p, o] for s, p, o in triples_array 
                if entity_counts[s] > 1 and entity_counts[o] > 1 and relation_counts[p] > 1
            ]
            
            # If no triples were removed this round, the graph is stable!
            if len(valid_triples) == len(triples_array):
                print(f"Graph stabilized after {iteration} iterations.")
                break
                
            triples_array = np.array(valid_triples)
            iteration += 1
            
        # 3. Pass the fully stabilized array to PyKEEN
        tf = TriplesFactory.from_labeled_triples(triples_array)
        
        print(f"Final usable triples: {tf.num_triples}")
        print(f"Entities: {tf.num_entities} | Relations: {tf.num_relations}")
        
    except FileNotFoundError:
        print(f"Error: {INPUT_KB} not found. Run step4_kb_expansion.py first.")
        return

    # 4. Split the data safely
    print("Splitting data into train/test/val...")
    # Because the graph is now dense, a standard 80/10/10 split will work perfectly.
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)

    # 3. Train the Model (TransE)
    print("\nStarting KGE Training (TransE)...")
    # Note: epochs=50 is low for quick lab execution. For better results, increase to 200+.
    result = pipeline(
        training=training,
        testing=testing,
        validation=validation,
        model="TransE",
        model_kwargs={"embedding_dim": 100},
        training_kwargs={"num_epochs": 50},  
        evaluation_kwargs={"batch_size": 256},
        random_seed=42,
        device="cpu" # Change to "cuda" if you have a GPU
    )

    # 4. Evaluation Metrics
    metrics = result.metric_results.to_dict()
    mrr = metrics["both"]["realistic"]["inverse_harmonic_mean_rank"]
    hits_at_10 = metrics["both"]["realistic"]["hits_at_10"]

    print("\n=== KGE Evaluation Results ===")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    print(f"Hits@10: {hits_at_10:.4f}")

    # 5. Save Model and Stats
    result.save_to_directory(OUTPUT_DIR)
    
    stats = {"MRR": mrr, "Hits@10": hits_at_10}
    with open(Path(OUTPUT_DIR) / "evaluation_metrics.json", "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n✔ Model, embeddings, and evaluation saved to directory: '{OUTPUT_DIR}'")

if __name__ == "__main__":
    run_kge()