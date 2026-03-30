"""
Embedding Visualisation (t-SNE)
Loads the trained RotatE model, extracts entity embeddings, 
reduces them to 2D using t-SNE, and plots them.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from pathlib import Path

# Paths
MODEL_PATH = Path("./kge_results/RotatE/trained_model.pkl")
OUTPUT_PLOT = Path("./kge_results/tsne_embeddings.png")

def main():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at {MODEL_PATH}.")
        print("Make sure you successfully trained the RotatE model first!")
        return

    print("Loading trained RotatE model...")
    # Load the PyTorch model
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    
    # Extract the raw 100-dimensional entity embeddings
    # model.entity_representations[0] holds the entity embedding layer
    print("Extracting entity embeddings...")
    embeddings = model.entity_representations[0](indices=None).detach().cpu().numpy()
    
    print(f"Loaded {embeddings.shape[0]} entities with dimension {embeddings.shape[1]}.")
    
    # Configure and run t-SNE
    print("Running t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(
            n_components=2, 
            perplexity=30, 
            max_iter=1000,    # <--- New parameter name
            random_state=42,
            init='pca',       
            learning_rate='auto'
        )
    embeddings_2d = tsne.fit_transform(embeddings)
    
    # Plotting
    print("Generating scatter plot...")
    plt.figure(figsize=(12, 10))
    
    #Create a basic scatter plot (alpha=0.6 makes overlapping points readable)
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6, s=15, c='dodgerblue', edgecolors='none')
    
    plt.title("t-SNE Projection of Olympics Knowledge Graph Entities (RotatE)", fontsize=16)
    plt.xlabel("t-SNE Dimension 1", fontsize=12)
    plt.ylabel("t-SNE Dimension 2", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Save the figure
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    print(f"✔ Success! Visualization saved to: {OUTPUT_PLOT}")

if __name__ == "__main__":
    main()