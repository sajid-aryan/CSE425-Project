"""
EASY TASK: Basic VAE for Hybrid Language Music Clustering

This script implements the Easy Task requirements:
- Basic VAE for feature extraction from music data
- K-Means clustering on latent features
- Visualization using t-SNE or UMAP
- Compare with baseline (PCA + K-Means) 
- Evaluate using Silhouette Score and Calinski-Harabasz Index
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import umap
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.dataset import AudioDataset, create_data_loaders
from src.vae import BasicVAE, train_vae
from src.clustering import extract_vae_features, BaselineComparison
from src.evaluation import ClusteringEvaluator
from src.visualization import VisualizationPipeline

def run_easy_task():
    """Execute the Easy Task implementation."""
    print("🚀 EASY TASK: Basic VAE + K-Means Clustering")
    print("=" * 50)
    
    # Set device (force GPU)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available. Please ensure you have a GPU and CUDA installed.")
        return None
    
    # Load dataset
    print("\n📂 Loading audio dataset...")
    data_dir = "data"
    audio_dir = os.path.join(data_dir, "audio")
    lyrics_files = []
    
    if not os.path.exists(audio_dir):
        print(f"❌ Error: Audio directory not found at {audio_dir}")
        return None
    
    # Create data loaders
    # NOTE: Feature extraction with librosa is CPU-bound; caching prevents re-extracting
    # features every epoch and makes training fast even when using a GPU.
    train_loader, test_loader = create_data_loaders(
        audio_dir,
        lyrics_files,
        batch_size=32,
        dataset_type='features',
        cache_dir=os.path.join(data_dir, "cache", "audio_features"),
        include_tonnetz=False,
        duration=10.0,
        max_files=None,
        num_workers=0,
    )
    
    # Get input dimension
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    print(f"📊 Input dimension: {input_dim}")
    
    # Initialize Basic VAE
    print("\n🧠 Initializing Basic VAE...")
    basic_vae = BasicVAE(
        input_dim=input_dim,
        hidden_dim=256,
        latent_dim=32
    ).to(device)
    
    # Train Basic VAE
    print("\n🏃 Training Basic VAE...")
    optimizer = optim.Adam(basic_vae.parameters(), lr=1e-4)  # Lower learning rate
    
    training_history = train_vae(
        basic_vae, train_loader, optimizer, device, epochs=50
    )
    
    print("✅ Training completed!")
    
    # Extract latent features
    print("\n🔍 Extracting latent features...")
    latent_features, true_labels = extract_vae_features(
        basic_vae, test_loader, device
    )
    
    print(f"📊 Latent features shape: {latent_features.shape}")
    
    # Apply K-Means clustering
    print("\n🎯 Applying K-Means clustering...")
    n_clusters = len(np.unique(true_labels))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    vae_cluster_labels = kmeans.fit_predict(latent_features)
    
    # Baseline: PCA + K-Means
    print("\n📊 Creating PCA + K-Means baseline...")
    baseline = BaselineComparison(n_clusters=n_clusters)
    
    # Get original features for baseline
    all_features = []
    for data, _ in test_loader:
        all_features.append(data.numpy())
    original_features = np.vstack(all_features)
    
    pca_features, pca_cluster_labels = baseline.pca_kmeans(original_features)
    
    # Evaluate clustering performance
    print("\n📈 Evaluating clustering performance...")
    evaluator = ClusteringEvaluator()
    
    # VAE + K-Means metrics
    vae_metrics = evaluator.evaluate_clustering(
        latent_features, vae_cluster_labels, true_labels
    )
    
    # PCA + K-Means metrics
    pca_metrics = evaluator.evaluate_clustering(
        pca_features, pca_cluster_labels, true_labels
    )
    
    # Display results
    print("\n" + "="*50)
    print("📊 EASY TASK RESULTS")
    print("="*50)
    
    print("\n🔬 VAE + K-Means Performance:")
    print(f"  Silhouette Score: {vae_metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Index: {vae_metrics['calinski_harabasz_index']:.4f}")
    print(f"  Adjusted Rand Index: {vae_metrics['adjusted_rand_index']:.4f}")
    
    print("\n📐 PCA + K-Means (Baseline) Performance:")
    print(f"  Silhouette Score: {pca_metrics['silhouette_score']:.4f}")
    print(f"  Calinski-Harabasz Index: {pca_metrics['calinski_harabasz_index']:.4f}")
    print(f"  Adjusted Rand Index: {pca_metrics['adjusted_rand_index']:.4f}")
    
    # Create visualizations
    print("\n🎨 Creating visualizations...")
    viz = VisualizationPipeline()
    
    # t-SNE visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latent_tsne = tsne.fit_transform(latent_features)
    
    # UMAP visualization
    umap_reducer = umap.UMAP(n_components=2, random_state=42)
    latent_umap = umap_reducer.fit_transform(latent_features)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # t-SNE plots
    axes[0, 0].scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
    axes[0, 0].set_title('t-SNE: True Labels')
    axes[0, 0].set_xlabel('t-SNE 1')
    axes[0, 0].set_ylabel('t-SNE 2')
    
    axes[0, 1].scatter(latent_tsne[:, 0], latent_tsne[:, 1], c=vae_cluster_labels, cmap='tab10', alpha=0.7)
    axes[0, 1].set_title('t-SNE: VAE + K-Means Clusters')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    
    # UMAP plots
    axes[1, 0].scatter(latent_umap[:, 0], latent_umap[:, 1], c=true_labels, cmap='tab10', alpha=0.7)
    axes[1, 0].set_title('UMAP: True Labels')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    
    axes[1, 1].scatter(latent_umap[:, 0], latent_umap[:, 1], c=vae_cluster_labels, cmap='tab10', alpha=0.7)
    axes[1, 1].set_title('UMAP: VAE + K-Means Clusters')
    axes[1, 1].set_xlabel('UMAP 1')
    axes[1, 1].set_ylabel('UMAP 2')
    
    plt.tight_layout()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/easy_task_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save metrics to CSV
    results_df = pd.DataFrame({
        'Method': ['VAE + K-Means', 'PCA + K-Means'],
        'Silhouette_Score': [vae_metrics['silhouette_score'], pca_metrics['silhouette_score']],
        'Calinski_Harabasz': [vae_metrics['calinski_harabasz_index'], pca_metrics['calinski_harabasz_index']],
        'Adjusted_Rand_Index': [vae_metrics['adjusted_rand_index'], pca_metrics['adjusted_rand_index']]
    })
    
    results_df.to_csv("results/easy_task_results.csv", index=False)
    print("💾 Results saved to results/easy_task_results.csv")
    
    # Determine winner
    if vae_metrics['silhouette_score'] > pca_metrics['silhouette_score']:
        winner = "VAE + K-Means"
    else:
        winner = "PCA + K-Means"
    
    print(f"\n🏆 Best performing method: {winner}")
    print("\n✅ Easy Task completed successfully!")
    
    return {
        'vae_metrics': vae_metrics,
        'pca_metrics': pca_metrics,
        'latent_features': latent_features,
        'true_labels': true_labels,
        'cluster_labels': vae_cluster_labels,
        'training_history': training_history
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the Easy Task
    results = run_easy_task()
    
    if results:
        print("\n🎉 Easy Task execution completed!")
        print("📂 Check the 'results/' folder for outputs.")
    else:
        print("\n❌ Easy Task execution failed!")