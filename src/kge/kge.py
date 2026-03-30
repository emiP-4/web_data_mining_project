"""
Step : KGE prep, training, and evaluation.
Theme: Olympic Games Knowledge Graph

This script uses PyKEEN to:
  1. Load the expanded_kb.nt file.
  2. Iteratively prune the graph to remove isolated nodes.
  3. Split triples into Train (80%) / Test (10%) / Validation (10%).
  4. Train BOTH TransE and RotatE models for comparison.
  5. Extract MRR, Hits@1, Hits@3, and Hits@10.
  6. Save models and a comparison JSON file.

Usage:
  pip install pykeen torch rdflib numpy
"""

import json
from pathlib import Path
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
import rdflib
import numpy as np
from collections import Counter

# --- Configuration ---
INPUT_KB = "./kg_artifacts/expanded_kb.nt"
OUTPUT_DIR = Path("./kge_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_and_prune_graph() -> TriplesFactory:
    print(f"Loading triples from {INPUT_KB}...")
    g = rdflib.Graph()
    try:
        g.parse(INPUT_KB, format="nt")
    except FileNotFoundError:
        print(f"[ERROR] {INPUT_KB} not found. Please run the KB expansion script first.")
        exit(1)
        
    # Convert to a 2D numpy array
    triples_array = np.array([[str(s), str(p), str(o)] for s, p, o in g])
    print(f"Initial raw triples: {len(triples_array)}")
    
    # Iteratively prune the graph until it stabilizes
    iteration = 1
    while True:
        all_entities = np.concatenate((triples_array[:, 0], triples_array[:, 2]))
        entity_counts = Counter(all_entities)
        relation_counts = Counter(triples_array[:, 1])
        
        # Keep only triples where both entities AND the relation appear at least twice
        valid_triples = [
            [s, p, o] for s, p, o in triples_array 
            if entity_counts[s] > 1 and entity_counts[o] > 1 and relation_counts[p] > 1
        ]
        
        if len(valid_triples) == len(triples_array):
            print(f"Graph stabilized after {iteration} iterations.")
            break
            
        triples_array = np.array(valid_triples)
        iteration += 1
        
    tf = TriplesFactory.from_labeled_triples(triples_array)
    print(f"Final usable triples: {tf.num_triples}")
    print(f"Entities: {tf.num_entities} | Relations: {tf.num_relations}")
    
    return tf

def main():
    # 1. Prepare Data
    tf = load_and_prune_graph()
    
    print("\nSplitting data into train/test/val (80/10/10)...")
    training, testing, validation = tf.split([0.8, 0.1, 0.1], random_state=42)
    
    results = {}
    models_to_train = ["TransE", "RotatE"]
    
    # 2. Train and Evaluate Models
    for model_name in models_to_train:
        print(f"\n========================================")
        print(f" Starting KGE Training: {model_name}")
        print(f"========================================")
        
        # Note: num_epochs is set to 50 for relatively quick execution. 
        # For your final report numbers, you might want to increase this to 200.
        result = pipeline(
            training=training,
            testing=testing,
            validation=validation,
            model=model_name,
            model_kwargs={"embedding_dim": 100},
            training_kwargs={"num_epochs": 50},  
            evaluation_kwargs={"batch_size": 256},
            random_seed=42,
            device="cpu" # Change to "cuda" if you have an Nvidia GPU
        )
        
        # 3. Extract Specific Metrics for the Rubric
        metrics = result.metric_results.to_dict()["both"]["realistic"]
        mrr = metrics["inverse_harmonic_mean_rank"]
        hits_1 = metrics["hits_at_1"]
        hits_3 = metrics["hits_at_3"]
        hits_10 = metrics["hits_at_10"]
        
        print(f"\n[{model_name} Results]")
        print(f"MRR:     {mrr:.4f}")
        print(f"Hits@1:  {hits_1:.4f}")
        print(f"Hits@3:  {hits_3:.4f}")
        print(f"Hits@10: {hits_10:.4f}")
        
        # 4. Save Model Artifacts
        model_out_dir = OUTPUT_DIR / model_name
        result.save_to_directory(str(model_out_dir))
        print(f"✔ Saved {model_name} artifacts to {model_out_dir}")
        
        # Store for final comparison JSON
        results[model_name] = {
            "MRR": mrr,
            "Hits@1": hits_1,
            "Hits@3": hits_3,
            "Hits@10": hits_10
        }
        
    # 5. Save Final Comparison JSON
    comparison_file = OUTPUT_DIR / "comparison.json"
    with open(comparison_file, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n✔ Full evaluation comparison saved to {comparison_file}")
    print("DONE! You can now copy the metrics from comparison.json into your report table.")

if __name__ == "__main__":
    main()