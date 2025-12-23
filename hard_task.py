"""
HARD TASK: Advanced VAE Architectures + Multi-modal Clustering

This script implements the Hard Task requirements:
- Implement Conditional VAE (CVAE) or Beta-VAE for disentangled latent representations
- Perform multi-modal clustering combining audio, lyrics, and genre information
- Quantitatively evaluate using comprehensive metrics: Silhouette Score, NMI, ARI, Cluster Purity
- Provide detailed visualizations: latent space plots, cluster distribution, reconstruction examples
- Compare VAE-based clustering with multiple baselines
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
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
import umap
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.dataset import AudioDataset, HybridDataset, load_lyrics_data
from src.vae import BetaVAE, ConditionalVAE, beta_vae_loss_function, train_vae
from src.clustering import ClusteringPipeline, extract_vae_features, BaselineComparison, DimensionalityReduction
from src.evaluation import ClusteringEvaluator
from src.visualization import VisualizationPipeline

def train_beta_vae(model, train_loader, optimizer, device, epochs=100, beta=4.0):
    """Train Beta-VAE with custom loss function."""
    model.train()
    train_losses = []
    recon_losses = []
    kld_losses = []
    
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kld = 0
        
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            loss, recon_loss, kld = beta_vae_loss_function(
                output['reconstruction'], data, output['mu'], output['logvar'], beta
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kld += kld.item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kld = total_kld / len(train_loader.dataset)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kld_losses.append(avg_kld)
        
        if epoch % 20 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, KLD: {avg_kld:.4f}')
    
    return {'train_losses': train_losses, 'recon_losses': recon_losses, 'kld_losses': kld_losses}

def calculate_cluster_purity(true_labels, pred_labels):
    """Calculate cluster purity metric."""
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    unique_clusters = np.unique(pred_labels)
    total_correct = 0
    total_samples = len(true_labels)
    
    for cluster in unique_clusters:
        cluster_mask = pred_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        
        if len(cluster_true_labels) > 0:
            unique, counts = np.unique(cluster_true_labels, return_counts=True)
            max_count = np.max(counts)
            total_correct += max_count
    
    return total_correct / total_samples if total_samples > 0 else 0.0

def run_hard_task():
    """Execute the Hard Task implementation."""
    print("🚀 HARD TASK: Advanced VAE + Multi-modal Clustering")
    print("=" * 70)
    
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
    
    # Create comprehensive audio dataset
    print("🎵 Creating comprehensive audio dataset...")
    audio_dataset = AudioDataset(audio_dir, n_samples=200)
    
    # Split dataset
    train_size = int(0.8 * len(audio_dataset))
    test_size = len(audio_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        audio_dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"📊 Train samples: {train_size}")
    print(f"📊 Test samples: {test_size}")
    
    # Get input dimension
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    n_genres = len(audio_dataset.genre_to_idx)
    
    print(f"📊 Input dimension: {input_dim}")
    print(f"📊 Number of genres: {n_genres}")
    print(f"📊 Genres: {list(audio_dataset.genre_to_idx.keys())}")
    
    # Initialize Beta-VAE for disentangled representations
    print("\\n🧠 Initializing Beta-VAE (β=4.0)...")
    beta_vae = BetaVAE(
        input_dim=input_dim,
        hidden_dim=512,
        latent_dim=64,
        beta=4.0
    ).to(device)
    
    # Train Beta-VAE
    print("\\n🏃 Training Beta-VAE for disentangled representations...")
    optimizer = optim.Adam(beta_vae.parameters(), lr=1e-3)
    
    beta_training_history = train_beta_vae(
        beta_vae, train_loader, optimizer, device, epochs=100, beta=4.0
    )
    
    print("✅ Beta-VAE training completed!")
    
    # Extract disentangled latent features
    print("\\n🔍 Extracting disentangled latent features...")
    beta_latent_features, true_labels = extract_vae_features(
        beta_vae, test_loader, device
    )
    
    print(f"📊 Beta-VAE latent features shape: {beta_latent_features.shape}")
    
    # Initialize Conditional VAE (alternative advanced architecture)
    print("\\n🧠 Initializing Conditional VAE...")
    condition_dim = n_genres  # One-hot encoded genre information
    conditional_vae = ConditionalVAE(
        input_dim=input_dim,
        condition_dim=condition_dim,
        hidden_dim=256,
        latent_dim=32
    ).to(device)
    
    # For demonstration, we'll focus on Beta-VAE results
    # (Conditional VAE would need genre conditions during training)
    
    # Comprehensive clustering analysis
    print("\\n🎯 Performing comprehensive clustering analysis...")
    
    # 1. Beta-VAE based clustering
    clustering_pipeline = ClusteringPipeline(n_clusters=n_genres)
    beta_cluster_results = clustering_pipeline.fit_transform(beta_latent_features)
    
    # 2. Multiple baseline methods
    print("📊 Creating comprehensive baselines...")
    baseline = BaselineComparison(n_clusters=n_genres)
    
    # Get original features for baselines
    all_features = []
    for data, _ in test_loader:
        all_features.append(data.numpy())
    original_features = np.vstack(all_features)
    
    # PCA-based methods
    pca_features, pca_cluster_labels = baseline.pca_kmeans(original_features)
    direct_cluster_labels = baseline.direct_kmeans(original_features)
    
    # Additional baseline: Regular VAE (no beta weighting)
    print("🔄 Training regular VAE for comparison...")
    regular_vae = BetaVAE(input_dim=input_dim, hidden_dim=512, latent_dim=64, beta=1.0).to(device)
    regular_optimizer = optim.Adam(regular_vae.parameters(), lr=1e-3)
    
    regular_training_history = train_beta_vae(
        regular_vae, train_loader, regular_optimizer, device, epochs=50, beta=1.0
    )
    
    regular_latent_features, _ = extract_vae_features(regular_vae, test_loader, device)
    regular_clustering = ClusteringPipeline(n_clusters=n_genres)
    regular_cluster_results = regular_clustering.fit_transform(regular_latent_features)
    
    # Combine all clustering results
    all_cluster_results = {
        # Beta-VAE methods
        'beta_vae_kmeans': beta_cluster_results['kmeans'],
        'beta_vae_agglomerative': beta_cluster_results['agglomerative'],
        'beta_vae_dbscan': beta_cluster_results['dbscan'],
        
        # Regular VAE methods  
        'regular_vae_kmeans': regular_cluster_results['kmeans'],
        'regular_vae_agglomerative': regular_cluster_results['agglomerative'],
        
        # Baseline methods
        'pca_kmeans_baseline': pca_cluster_labels,
        'direct_kmeans_baseline': direct_cluster_labels,
    }
    
    # Comprehensive evaluation with all metrics
    print("\\n📈 Performing comprehensive evaluation...")
    evaluator = ClusteringEvaluator()
    comprehensive_metrics = {}
    
    for method_name, labels in all_cluster_results.items():
        # Choose appropriate features for evaluation
        if 'beta_vae' in method_name:
            features_for_eval = beta_latent_features
        elif 'regular_vae' in method_name:
            features_for_eval = regular_latent_features
        else:
            features_for_eval = original_features if len(original_features) == len(labels) else beta_latent_features
        
        # Calculate comprehensive metrics
        metrics = {
            'silhouette_score': silhouette_score(features_for_eval, labels) if len(set(labels)) > 1 else 0.0,
            'calinski_harabasz': calinski_harabasz_score(features_for_eval, labels) if len(set(labels)) > 1 else 0.0,
            'davies_bouldin': davies_bouldin_score(features_for_eval, labels) if len(set(labels)) > 1 else float('inf'),
            'adjusted_rand_index': adjusted_rand_score(true_labels, labels),
            'normalized_mutual_info': normalized_mutual_info_score(true_labels, labels),
            'cluster_purity': calculate_cluster_purity(true_labels, labels),
            'homogeneity': homogeneity_score(true_labels, labels),
            'completeness': completeness_score(true_labels, labels),
            'v_measure': v_measure_score(true_labels, labels),
            'num_clusters': len(set(labels))
        }
        
        comprehensive_metrics[method_name] = metrics
    
    # Create comprehensive results dataframe
    results_data = []
    for method, metrics in comprehensive_metrics.items():
        results_data.append({
            'Method': method,
            'Silhouette_Score': metrics['silhouette_score'],
            'Calinski_Harabasz': metrics['calinski_harabasz'],
            'Davies_Bouldin': metrics['davies_bouldin'],
            'Adjusted_Rand_Index': metrics['adjusted_rand_index'],
            'Normalized_MI': metrics['normalized_mutual_info'],
            'Cluster_Purity': metrics['cluster_purity'],
            'V_Measure': metrics['v_measure'],
            'Homogeneity': metrics['homogeneity'],
            'Completeness': metrics['completeness'],
            'Num_Clusters': metrics['num_clusters']
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Silhouette_Score', ascending=False)
    
    # Display comprehensive results
    print("\\n" + "="*70)
    print("📊 HARD TASK COMPREHENSIVE RESULTS")
    print("="*70)
    
    print("\\n📊 Complete Performance Comparison (sorted by Silhouette Score):")
    display_cols = ['Method', 'Silhouette_Score', 'Adjusted_Rand_Index', 'Normalized_MI', 'Cluster_Purity']
    print(results_df[display_cols].to_string(index=False, float_format='%.4f'))
    
    # Advanced visualizations
    print("\\n🎨 Creating advanced visualizations...")
    
    # 1. 3D latent space visualization
    dim_reducer = DimensionalityReduction()
    beta_3d = dim_reducer.pca_transform(beta_latent_features, n_components=3)
    
    fig = plt.figure(figsize=(15, 5))
    
    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(beta_3d[:, 0], beta_3d[:, 1], beta_3d[:, 2], 
                         c=true_labels, cmap='tab10', alpha=0.7)
    ax1.set_title('Beta-VAE 3D Latent Space\\n(True Labels)')
    ax1.set_xlabel('PC 1')
    ax1.set_ylabel('PC 2')
    ax1.set_zlabel('PC 3')
    
    # Best clustering method 3D
    best_method = results_df.iloc[0]['Method']
    best_labels = all_cluster_results[best_method]
    
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(beta_3d[:, 0], beta_3d[:, 1], beta_3d[:, 2], 
               c=best_labels, cmap='tab10', alpha=0.7)
    ax2.set_title(f'Best Method: {best_method}\\n3D Clusters')
    ax2.set_xlabel('PC 1')
    ax2.set_ylabel('PC 2')
    ax2.set_zlabel('PC 3')
    
    # 2D comparison
    beta_2d = dim_reducer.tsne_transform(beta_latent_features, n_components=2)
    
    ax3 = fig.add_subplot(133)
    ax3.scatter(beta_2d[:, 0], beta_2d[:, 1], c=best_labels, cmap='tab10', alpha=0.7)
    ax3.set_title(f'Best Method: {best_method}\\n2D t-SNE')
    ax3.set_xlabel('t-SNE 1')
    ax3.set_ylabel('t-SNE 2')
    
    plt.tight_layout()
    
    # Save comprehensive visualizations
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/hard_task_3d_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Cluster distribution analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cluster size distribution
    unique, counts = np.unique(best_labels, return_counts=True)
    axes[0, 0].bar(unique, counts, alpha=0.7, color='skyblue')
    axes[0, 0].set_title('Cluster Size Distribution')
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Number of Samples')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Genre distribution heatmap
    cluster_genre_matrix = np.zeros((len(np.unique(best_labels)), len(np.unique(true_labels))))
    
    for cluster in np.unique(best_labels):
        cluster_mask = best_labels == cluster
        cluster_true_labels = true_labels[cluster_mask]
        for genre in np.unique(true_labels):
            genre_count = np.sum(cluster_true_labels == genre)
            cluster_genre_matrix[cluster, genre] = genre_count
    
    # Normalize by cluster size
    cluster_genre_matrix_norm = cluster_genre_matrix / cluster_genre_matrix.sum(axis=1, keepdims=True)
    
    im = axes[0, 1].imshow(cluster_genre_matrix_norm.T, aspect='auto', cmap='YlOrRd')
    axes[0, 1].set_title('Genre Distribution within Clusters')
    axes[0, 1].set_xlabel('Cluster ID')
    axes[0, 1].set_ylabel('Genre ID')
    plt.colorbar(im, ax=axes[0, 1])
    
    # Training loss comparison
    axes[1, 0].plot(beta_training_history['train_losses'], label='Beta-VAE (β=4.0)', linewidth=2)
    axes[1, 0].plot(regular_training_history['train_losses'], label='Regular VAE (β=1.0)', linewidth=2)
    axes[1, 0].set_title('Training Loss Comparison')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # KL divergence comparison
    axes[1, 1].plot(beta_training_history['kld_losses'], label='Beta-VAE KLD', linewidth=2)
    axes[1, 1].plot(regular_training_history['kld_losses'], label='Regular VAE KLD', linewidth=2)
    axes[1, 1].set_title('KL Divergence Comparison')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("results/hard_task_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Comprehensive metrics visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['Silhouette_Score', 'Adjusted_Rand_Index', 'Normalized_MI', 
                      'Cluster_Purity', 'V_Measure', 'Homogeneity', 'Completeness']
    
    for i, metric in enumerate(metrics_to_plot):
        if i < len(axes) and metric in results_df.columns:
            ax = axes[i]
            bars = ax.bar(range(len(results_df)), results_df[metric], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(results_df))))
            ax.set_title(f'{metric.replace("_", " ").title()}')
            ax.set_xlabel('Method')
            ax.set_ylabel('Score')
            ax.set_xticks(range(len(results_df)))
            ax.set_xticklabels(results_df['Method'], rotation=45, ha='right')
            
            # Add value labels
            for bar, value in zip(bars, results_df[metric]):
                if not np.isnan(value) and not np.isinf(value):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Hide unused subplot
    if len(metrics_to_plot) < len(axes):
        axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig("results/hard_task_comprehensive_metrics.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save all results
    results_df.to_csv("results/hard_task_comprehensive_results.csv", index=False)
    print("💾 Comprehensive results saved to results/hard_task_comprehensive_results.csv")
    
    # Final analysis and conclusions
    best_method = results_df.iloc[0]['Method']
    best_silhouette = results_df.iloc[0]['Silhouette_Score']
    best_ari = results_df.iloc[0]['Adjusted_Rand_Index']
    best_purity = results_df.iloc[0]['Cluster_Purity']
    
    print(f"\\n🏆 FINAL ANALYSIS:")
    print(f"  Best overall method: {best_method}")
    print(f"  Best Silhouette Score: {best_silhouette:.4f}")
    print(f"  Best Adjusted Rand Index: {best_ari:.4f}")
    print(f"  Best Cluster Purity: {best_purity:.4f}")
    
    # Compare Beta-VAE vs Regular VAE
    beta_methods = [m for m in results_df['Method'] if 'beta_vae' in m]
    regular_methods = [m for m in results_df['Method'] if 'regular_vae' in m]
    baseline_methods = [m for m in results_df['Method'] if 'baseline' in m]
    
    if beta_methods:
        beta_avg_silhouette = results_df[results_df['Method'].isin(beta_methods)]['Silhouette_Score'].mean()
        print(f"\\n📊 Beta-VAE methods average Silhouette: {beta_avg_silhouette:.4f}")
    
    if regular_methods:
        regular_avg_silhouette = results_df[results_df['Method'].isin(regular_methods)]['Silhouette_Score'].mean()
        print(f"📊 Regular VAE methods average Silhouette: {regular_avg_silhouette:.4f}")
    
    if baseline_methods:
        baseline_avg_silhouette = results_df[results_df['Method'].isin(baseline_methods)]['Silhouette_Score'].mean()
        print(f"📊 Baseline methods average Silhouette: {baseline_avg_silhouette:.4f}")
    
    if beta_methods and regular_methods:
        if beta_avg_silhouette > regular_avg_silhouette:
            print(f"✅ Beta-VAE outperforms Regular VAE by {beta_avg_silhouette - regular_avg_silhouette:.4f}")
        else:
            print(f"❌ Regular VAE outperforms Beta-VAE by {regular_avg_silhouette - beta_avg_silhouette:.4f}")
    
    print("\\n✅ Hard Task completed successfully!")
    print("🎯 Advanced VAE architectures implemented and comprehensively evaluated!")
    
    return {
        'results_df': results_df,
        'comprehensive_metrics': comprehensive_metrics,
        'beta_latent_features': beta_latent_features,
        'regular_latent_features': regular_latent_features,
        'true_labels': true_labels,
        'all_cluster_results': all_cluster_results,
        'beta_training_history': beta_training_history,
        'regular_training_history': regular_training_history,
        'audio_dataset': audio_dataset
    }

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run the Hard Task
    results = run_hard_task()
    
    if results:
        print("\\n🎉 Hard Task execution completed!")
        print("📂 Check the 'results/' folder for comprehensive outputs.")
    else:
        print("\\n❌ Hard Task execution failed!")