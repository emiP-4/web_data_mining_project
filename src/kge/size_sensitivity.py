"""
Size Sensitivity Analysis for KGE
Runs TransE and RotatE against three graph sizes: Unfiltered, Filtered, and a 20k Subset.
"""

import numpy as np
import rdflib
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory
from collections import Counter

INPUT_KB = "./kg_artifacts/expanded_kb.nt"

def get_triples():
    g = rdflib.Graph()
    g.parse(INPUT_KB, format="nt")
    return np.array([[str(s), str(p), str(o)] for s, p, o in g])

def prune_graph(triples_array):
    iteration = 1
    while True:
        all_entities = np.concatenate((triples_array[:, 0], triples_array[:, 2]))
        entity_counts = Counter(all_entities)
        relation_counts = Counter(triples_array[:, 1])
        
        valid_triples = [
            [s, p, o] for s, p, o in triples_array 
            if entity_counts[s] > 1 and entity_counts[o] > 1 and relation_counts[p] > 1
        ]
        
        if len(valid_triples) == len(triples_array):
            break
        triples_array = np.array(valid_triples)
        iteration += 1
    return triples_array

def run_evaluation(triples_array, dataset_name):
    print(f"\n--- Running evaluation for: {dataset_name} ({len(triples_array)} triples) ---")
    tf = TriplesFactory.from_labeled_triples(triples_array)
    
    try:
        # PyKEEN will crash here if the graph has entities with a degree of 1
        train, test, val = tf.split([0.8, 0.1, 0.1], random_state=42)
    except ValueError as e:
        print(f"[!] Splitting failed for {dataset_name}.")
        print(f"[!] Reason: The graph is too sparse and contains isolated nodes.")
        print("[!] Marking MRR as N/A (Failed to train).\n")
        return # Skip training for this broken dataset
    
    for model in ["TransE", "RotatE"]:
        print(f"  Training {model}...")
        result = pipeline(
            training=train, testing=test, validation=val,
            model=model,
            model_kwargs={"embedding_dim": 100},
            training_kwargs={"num_epochs": 50},  
            evaluation_kwargs={"batch_size": 256},
            random_seed=42, device="cpu"
        )
        mrr = result.metric_results.to_dict()["both"]["realistic"]["inverse_harmonic_mean_rank"]
        print(f"  -> {dataset_name} | {model} MRR: {mrr:.4f}")

def main():
    print("Loading and pruning graph...")
    all_triples = get_triples()
    filtered_triples = prune_graph(all_triples)
    
    actual_len = len(filtered_triples)
    subset_size = actual_len // 2  # Take exactly half of your real triples
    
    print(f"\n[INFO] Total Unfiltered Triples: {len(all_triples)}")
    print(f"[INFO] Total Filtered Triples:   {actual_len}")
    print(f"[INFO] Creating a Subset of:     {subset_size}")
    
    # Create the dynamic subset
    np.random.seed(42)
    indices_subset = np.random.choice(actual_len, subset_size, replace=False)
    subset_triples = filtered_triples[indices_subset]
    
    # Run the evaluations
    run_evaluation(all_triples, f"Unfiltered ({len(all_triples)})")
    run_evaluation(subset_triples, f"50% Subset ({subset_size})")
    run_evaluation(filtered_triples, f"Filtered ({actual_len})")

if __name__ == "__main__":
    main()

if __name__ == "__main__":
    main()