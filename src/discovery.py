import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import pandas as pd

class ChemicalDiscovery:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def get_embeddings(self, dataloader):
        """Extract embeddings from the projector head."""
        from tqdm import tqdm
        all_embeddings = []
        all_metadata = []
        
        with torch.no_grad():
            for inputs, _ in tqdm(dataloader, desc="Extracting embeddings"):
                inputs = inputs.to(self.device)
                _, embs = self.model(inputs)
                all_embeddings.append(embs.cpu().numpy())
                
        return np.concatenate(all_embeddings, axis=0)

    def perform_clustering(self, embeddings, n_clusters=5):
        """Scalable clustering for large datasets using MiniBatchKMeans."""
        from sklearn.cluster import MiniBatchKMeans
        
        # Check for NaN values
        nan_mask = np.isnan(embeddings).any(axis=1)
        nan_count = nan_mask.sum()
        
        if nan_count > 0:
            print(f"Warning: Found {nan_count} embeddings with NaN values")
            print(f"Removing NaN embeddings ({nan_count}/{len(embeddings)} samples)")
            
            # Filter out NaN embeddings
            valid_embeddings = embeddings[~nan_mask]
            valid_indices = np.where(~nan_mask)[0]
            
            print(f"Clustering with {len(valid_embeddings)} valid embeddings")
        else:
            valid_embeddings = embeddings
            valid_indices = np.arange(len(embeddings))
        
        # Check if we have enough samples
        if len(valid_embeddings) < n_clusters:
            print(f"Warning: Only {len(valid_embeddings)} valid samples, reducing clusters to {len(valid_embeddings)}")
            n_clusters = max(1, len(valid_embeddings))
        
        # Use MiniBatchKMeans for memory efficiency (perfect for large datasets)
        print(f"Using MiniBatchKMeans for efficient clustering of {len(valid_embeddings)} samples...")
        clustering = MiniBatchKMeans(
            n_clusters=n_clusters, 
            batch_size=1000,
            random_state=42,
            n_init=10,
            verbose=0
        )
        valid_clusters = clustering.fit_predict(valid_embeddings)
        
        # Create full cluster array (with -1 for NaN samples)
        clusters = np.full(len(embeddings), -1, dtype=int)
        clusters[valid_indices] = valid_clusters
        
        # For visualization, use a sample for dendrogram
        # (Hierarchical clustering on small sample, not full dataset)
        sample_size = min(1000, len(valid_embeddings))
        sample_idx = np.random.choice(len(valid_embeddings), sample_size, replace=False)
        Z = linkage(valid_embeddings[sample_idx], 'ward')
        
        print(f"Clustering complete! {n_clusters} clusters identified.")
        
        return clusters, Z

    def generate_saliency_map(self, input_spectrum, target_class=None):
        """
        Generate a saliency map to visualize key contributing peaks.
        input_spectrum: (2048,) or (1, 2048) or (1, 1, 2048)
        """
        self.model.eval()  # Use eval mode to avoid BatchNorm error with batch_size=1
        input_spectrum = torch.FloatTensor(input_spectrum).to(self.device)
        
        # Ensure proper shape: [batch, channels, length]
        if input_spectrum.dim() == 1:
            # (2048,) -> (1, 1, 2048)
            input_spectrum = input_spectrum.unsqueeze(0).unsqueeze(0)
        elif input_spectrum.dim() == 2:
            # (1, 2048) -> (1, 1, 2048)
            input_spectrum = input_spectrum.unsqueeze(1)
        
        input_spectrum.requires_grad_()
        
        # Enable gradients even in eval mode for saliency computation
        with torch.set_grad_enabled(True):
            logits, _ = self.model(input_spectrum)
            
            if target_class is None:
                target_class = logits.argmax().item()
                
            score = logits[0, target_class]
            score.backward()
        
        saliency = input_spectrum.grad.abs().squeeze().cpu().numpy()
        return saliency

    def plot_dendrogram(self, Z, labels=None):
        plt.figure(figsize=(12, 8))
        dendrogram(Z, labels=labels, leaf_rotation=90)
        plt.title('Hierarchical Clustering of Organic Families (Embeddings)')
        plt.xlabel('Species')
        plt.ylabel('Distance')
        plt.show()

if __name__ == "__main__":
    print("ChemicalDiscovery module ready.")
