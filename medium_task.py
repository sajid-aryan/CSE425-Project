"""
MEDIUM TASK: Enhanced Convolutional VAE + Hybrid Features

This script implements the Medium Task requirements:
- Enhance VAE with convolutional architecture for spectrograms/MFCC features
- Include hybrid feature representation: audio + lyrics embeddings
- Experiment with clustering algorithms: K-Means, Agglomerative Clustering, DBSCAN
- Evaluate clustering quality using enhanced metrics
- Compare results across methods and analyze performance
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
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
import umap
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.dataset import SpectrogramDataset, LyricsDataset
from src.vae import ConvVAE, train_vae
from src.clustering import ClusteringPipeline, extract_vae_features, BaselineComparison
from src.evaluation import ClusteringEvaluator
from src.visualization import VisualizationPipeline

def run_medium_task():
    """Execute the Medium Task implementation."""
    print("🚀 MEDIUM TASK: Convolutional VAE + Hybrid Features")
    print("=" * 60)
    
    # Set device (force GPU)
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using device: {device}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available. Please ensure you have a GPU and CUDA installed.")
        return None
    
    # Load datasets
    print("\n📂 Loading datasets...")
    data_dir = "data"
    audio_dir = os.path.join(data_dir, "audio")
    lyrics_dir = os.path.join(data_dir, "lyrics")
    
    if not os.path.exists(audio_dir):
        print(f"❌ Error: Audio directory not found at {audio_dir}")
        return None
    
    # Create spectrogram dataset for ConvVAE
    # NOTE: Mel-spectrogram extraction is CPU-bound; caching makes GPU training much faster.
    print("🎨 Creating spectrogram dataset...")
    full_spec_dataset = SpectrogramDataset(
        audio_dir,
        n_mels=64,
        duration=10.0,
        cache_dir=os.path.join(data_dir, "cache", "mel_spectrograms"),
    )

    # Random subset (keeps runtime manageable while covering many genres)
    subset_size = min(150, len(full_spec_dataset))
    subset_indices = np.random.choice(len(full_spec_dataset), size=subset_size, replace=False)
    spec_dataset = torch.utils.data.Subset(full_spec_dataset, subset_indices)

    # Load lyrics (hybrid language: English + Bangla)
    lyrics_files = []
    if os.path.exists(lyrics_dir):
        for fname in os.listdir(lyrics_dir):
            if fname.lower().endswith('.csv'):
                lyrics_files.append(os.path.join(lyrics_dir, fname))
    lyrics_dataset = LyricsDataset(lyrics_files) if len(lyrics_files) > 0 else None
    
    # Split dataset
    train_size = int(0.8 * len(spec_dataset))
    test_size = len(spec_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        spec_dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    print(f"📊 Train samples: {train_size}")
    print(f"📊 Test samples: {test_size}")
    
    # Get input dimensions
    sample_batch, _ = next(iter(train_loader))
    input_channels = sample_batch.shape[1]
    input_height = sample_batch.shape[2] 
    input_width = sample_batch.shape[3]
    
    print(f"📊 Input shape: ({input_channels}, {input_height}, {input_width})")
    
    # Initialize Convolutional VAE
    print("\n🧠 Initializing Convolutional VAE...")
    conv_vae = ConvVAE(
        input_channels=input_channels,
        input_height=input_height,
        input_width=input_width,
        latent_dim=64
    ).to(device)
    
    # Train Convolutional VAE
    print("\n🏃 Training Convolutional VAE...")
    optimizer = optim.Adam(conv_vae.parameters(), lr=1e-3)
    
    training_history = train_vae(
        conv_vae, train_loader, optimizer, device, epochs=50
    )
    
    print("✅ Training completed!")
    
    # Extract latent features
    print("\n🔍 Extracting convolutional latent features...")
    latent_features, true_labels = extract_vae_features(
        conv_vae, test_loader, device
    )
    
    print(f"📊 Latent features shape: {latent_features.shape}")
    print(f"📊 Unique genres: {len(np.unique(true_labels))}")
    
    # Apply multiple clustering algorithms (audio-only)
    print("\n🎯 Applying multiple clustering algorithms...")
    clustering_pipeline = ClusteringPipeline(n_clusters=len(np.unique(true_labels)))
    cluster_results = clustering_pipeline.fit_transform(latent_features)

    # Hybrid representation: concatenate audio latent + lyrics embedding (reduced)
    # Note: lyrics are not paired to audio tracks by ID in this dataset; we align by index
    # as a practical multimodal fusion baseline for this assignment.
    hybrid_features = None
    if lyrics_dataset is not None and len(lyrics_dataset) > 0:
        # Build N lyrics embeddings to match test set size
        vocab_size = 1000
        lyrics_embeds = np.zeros((len(true_labels), vocab_size), dtype=np.float32)
        lyrics_lang = []
        from src.dataset import HybridDataset
        # Reuse the deterministic text encoder from HybridDataset
        dummy_audio = None
        encoder = HybridDataset
        # Instantiate a small helper HybridDataset for encode_text
        # (audio_dataset isn't used for encode_text)
        # We'll just call encode_text from an instance.
        from src.dataset import AudioDataset
        tmp_audio_ds = AudioDataset(audio_dir, max_files=1, duration=1.0, cache_dir=os.path.join(data_dir, "cache", "audio_features"))
        tmp_hybrid = HybridDataset(tmp_audio_ds, lyrics_dataset)

        for i in range(len(true_labels)):
            text, lang = lyrics_dataset[i % len(lyrics_dataset)]
            lyrics_lang.append(lang)
            emb = tmp_hybrid.encode_text(text).numpy().astype(np.float32, copy=False)
            lyrics_embeds[i] = emb

        # Reduce lyrics embedding dimensionality before concatenation
        from sklearn.decomposition import PCA
        n_lyrics_components = min(32, lyrics_embeds.shape[0], lyrics_embeds.shape[1])
        lyrics_reduced = PCA(n_components=n_lyrics_components, random_state=42).fit_transform(lyrics_embeds)
        hybrid_features = np.concatenate([latent_features, lyrics_reduced], axis=1)

        hybrid_cluster_results = clustering_pipeline.fit_transform(hybrid_features)
        cluster_results['hybrid_kmeans'] = hybrid_cluster_results['kmeans']
        cluster_results['hybrid_agglomerative'] = hybrid_cluster_results['agglomerative']
        cluster_results['hybrid_dbscan'] = hybrid_cluster_results['dbscan']
    
    # Baseline comparisons
    print("\n📊 Creating baseline comparisons...")
    baseline = BaselineComparison(n_clusters=len(np.unique(true_labels)))
    
    # Get original features for baseline
    all_features = []
    for data, _ in test_loader:
        # Flatten spectrograms for baseline methods
        flattened = data.view(data.size(0), -1).numpy()
        all_features.append(flattened)
    original_features = np.vstack(all_features)
    
    pca_features, pca_cluster_labels = baseline.pca_kmeans(original_features)
    direct_cluster_labels = baseline.direct_kmeans(original_features)
    
    # Add baseline results
    cluster_results['pca_kmeans'] = pca_cluster_labels
    cluster_results['direct_kmeans'] = direct_cluster_labels
    
    # Evaluate all clustering methods
    print("\n📈 Evaluating clustering performance...")
    evaluator = ClusteringEvaluator()
    all_metrics = {}
    
    for method_name, labels in cluster_results.items():
        if method_name.startswith('hybrid_') and hybrid_features is not None:
            features_for_eval = hybrid_features
        elif 'baseline' in method_name or 'direct' in method_name or 'pca' in method_name:
            # Use original features for baseline methods
            features_for_eval = original_features if len(original_features) == len(labels) else latent_features
        else:
            # Use latent features for VAE-based methods
            features_for_eval = latent_features
        
        metrics = evaluator.evaluate_clustering(
            features_for_eval, labels, true_labels
        )
        all_metrics[method_name] = metrics
    
    # Display results
    print("\n" + "="*60)
    print("📊 MEDIUM TASK RESULTS")
    print("="*60)
    
    # Create results dataframe
    results_data = []
    for method, metrics in all_metrics.items():
        results_data.append({
            'Method': method,
            'Silhouette_Score': metrics['silhouette_score'],
            'Calinski_Harabasz': metrics['calinski_harabasz_index'],
            'Davies_Bouldin': metrics['davies_bouldin_index'],
            'Adjusted_Rand_Index': metrics['adjusted_rand_index'],
            'Normalized_MI': metrics['normalized_mutual_info']
        })
    
    results_df = pd.DataFrame(results_data)
    
    # Sort by silhouette score
    results_df = results_df.sort_values('Silhouette_Score', ascending=False)
    
    print("\n📊 Performance Comparison (sorted by Silhouette Score):")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Create comprehensive visualizations
    print("\n🎨 Creating comprehensive visualizations...")
    
    # Dimensionality reduction for visualization
    tsne = TSNE(n_components=2, random_state=42, perplexity=20)
    latent_tsne = tsne.fit_transform(latent_features)
    
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=10)
    latent_umap = umap_reducer.fit_transform(latent_features)
    
    # Plot clustering results comparison
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Select top methods for visualization
    top_methods = ['kmeans', 'agglomerative', 'dbscan', 'pca_kmeans']
    
    plot_idx = 0
    
    # True labels
    if plot_idx < len(axes):
        axes[plot_idx].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                              c=true_labels, cmap='tab10', alpha=0.7, s=30)
        axes[plot_idx].set_title('True Labels (t-SNE)')
        axes[plot_idx].set_xlabel('t-SNE 1')
        axes[plot_idx].set_ylabel('t-SNE 2')
        plot_idx += 1
    
    # Clustering methods
    for method in top_methods:
        if method in cluster_results and plot_idx < len(axes):
            labels = cluster_results[method]
            axes[plot_idx].scatter(latent_tsne[:, 0], latent_tsne[:, 1], 
                                  c=labels, cmap='tab10', alpha=0.7, s=30)
            axes[plot_idx].set_title(f'{method.title()} Clustering (t-SNE)')
            axes[plot_idx].set_xlabel('t-SNE 1')
            axes[plot_idx].set_ylabel('t-SNE 2')
            plot_idx += 1
    
    # UMAP visualizations
    if plot_idx < len(axes):
        axes[plot_idx].scatter(latent_umap[:, 0], latent_umap[:, 1], 
                              c=true_labels, cmap='tab10', alpha=0.7, s=30)
        axes[plot_idx].set_title('True Labels (UMAP)')
        axes[plot_idx].set_xlabel('UMAP 1')
        axes[plot_idx].set_ylabel('UMAP 2')
        plot_idx += 1
    
    # Best method UMAP
    if plot_idx < len(axes):
        best_method = results_df.iloc[0]['Method']
        if best_method in cluster_results:
            best_labels = cluster_results[best_method]
            axes[plot_idx].scatter(latent_umap[:, 0], latent_umap[:, 1], 
                                  c=best_labels, cmap='tab10', alpha=0.7, s=30)
            axes[plot_idx].set_title(f'Best Method: {best_method.title()} (UMAP)')
            axes[plot_idx].set_xlabel('UMAP 1')
            axes[plot_idx].set_ylabel('UMAP 2')
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    # Save results
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/medium_task_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create metrics comparison plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['Silhouette_Score', 'Calinski_Harabasz', 'Davies_Bouldin', 
                      'Adjusted_Rand_Index', 'Normalized_MI']
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes) and metric in results_df.columns:
            ax = axes[i]
            bars = ax.bar(range(len(results_df)), results_df[metric], 
                         color=plt.cm.Set3(np.linspace(0, 1, len(results_df))))
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df['Method'], rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, results_df[metric]):
                if not np.isnan(value) and not np.isinf(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplot
    if len(metrics_to_plot) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/medium_task_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save detailed results
    results_df.to_csv("results/medium_task_results.csv", index=False)
    print("💾 Results saved to results/medium_task_results.csv")
    
    # Analysis and conclusions
    best_method = results_df.iloc[0]['Method']
    best_score = results_df.iloc[0]['Silhouette_Score']
    
    print(f"\n🏆 Best performing method: {best_method}")
    print(f"🎯 Best Silhouette Score: {best_score:.4f}")
    
    # Compare with baselines
    vae_methods = [m for m in results_df['Method'] if m not in ['pca_kmeans', 'direct_kmeans']]
    baseline_methods = [m for m in results_df['Method'] if m in ['pca_kmeans', 'direct_kmeans']]
    
    if vae_methods and baseline_methods:
        vae_scores = results_df[results_df['Method'].isin(vae_methods)]['Silhouette_Score']
        baseline_scores = results_df[results_df['Method'].isin(baseline_methods)]['Silhouette_Score']
        
        avg_vae_score = vae_scores.mean()
        avg_baseline_score = baseline_scores.mean()
        
        print(f"\n📊 Performance Analysis:")
        print(f"  Average VAE-based methods: {avg_vae_score:.4f}")
        print(f"  Average baseline methods: {avg_baseline_score:.4f}")
        
        if avg_vae_score > avg_baseline_score:
            print(f"  ✅ VAE-based methods outperform baselines by {avg_vae_score - avg_baseline_score:.4f}")
        else:
            print(f"  ❌ Baseline methods perform better by {avg_baseline_score - avg_vae_score:.4f}")
    
    print("\n✅ Medium Task completed successfully!")
    
    return {
        'results_df': results_df,
        'all_metrics': all_metrics,
        'latent_features': latent_features,
        'true_labels': true_labels,
        'cluster_results': cluster_results,
        'training_history': training_history
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the Medium Task
    results = run_medium_task()
    
    if results:
        print("\n🎉 Medium Task execution completed!")
        print("📂 Check the 'results/' folder for outputs.")
    else:
        print("\n❌ Medium Task execution failed!")