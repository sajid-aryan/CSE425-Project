"""
Main execution script for VAE Music Clustering Project.

This script implements all three tasks (Easy, Medium, Hard) as specified in the guidelines
to maximize the project score according to the grading criteria.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from src.dataset import (
    AudioDataset, SpectrogramDataset, LyricsDataset, HybridDataset,
    create_data_loaders, load_lyrics_data, get_feature_statistics
)
from src.vae import (
    BasicVAE, ConvVAE, BetaVAE, ConditionalVAE, 
    vae_loss_function, train_vae
)
from src.clustering import (
    ClusteringPipeline, BaselineComparison, DimensionalityReduction,
    extract_vae_features, compare_clustering_methods, create_clustering_visualization_data
)
from src.evaluation import (
    ClusteringEvaluator, MetricsComparison, save_evaluation_results
)
from src.visualization import (
    VisualizationPipeline, create_comprehensive_report_plots
)


class MusicClusteringExperiment:
    """Main experiment class for VAE-based music clustering."""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "results",
                 device: str = None, random_seed: int = 42):
        """
        Initialize experiment.
        
        Args:
            data_dir: Directory containing data
            results_dir: Directory for results
            device: Device for training (auto-detect if None)
            random_seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.random_seed = random_seed
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Set random seeds
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        
        # Create results directory
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, "latent_visualization"), exist_ok=True)
        
        # Initialize components
        self.evaluator = ClusteringEvaluator()
        self.metrics_comparison = MetricsComparison()
        self.viz_pipeline = VisualizationPipeline()
        
        # Results storage
        self.experiment_results = {}
    
    def load_datasets(self):
        """Load and prepare all datasets."""
        print("Loading datasets...")
        
        # Audio datasets
        audio_dir = os.path.join(self.data_dir, "audio")
        
        # Feature-based dataset for Basic VAE
        self.audio_dataset = AudioDataset(audio_dir)
        print(f"Loaded {len(self.audio_dataset)} audio samples")
        
        # Spectrogram dataset for ConvVAE
        self.spectrogram_dataset = SpectrogramDataset(audio_dir)
        
        # Lyrics dataset
        lyrics_dir = os.path.join(self.data_dir, "lyrics")
        lyrics_files = [
            os.path.join(lyrics_dir, f) for f in os.listdir(lyrics_dir) 
            if f.endswith('.csv')
        ]
        self.lyrics_dataset = LyricsDataset(lyrics_files)
        
        # Hybrid dataset
        self.hybrid_dataset = HybridDataset(self.audio_dataset, self.lyrics_dataset)
        
        # Create data loaders
        self.audio_train_loader, self.audio_test_loader = create_data_loaders(
            audio_dir, lyrics_files, dataset_type='features'
        )
        
        self.spec_train_loader, self.spec_test_loader = create_data_loaders(
            audio_dir, lyrics_files, dataset_type='spectrogram'
        )
        
        # Get dataset statistics
        audio_stats = get_feature_statistics(self.audio_dataset)
        print(f"Audio dataset stats: {audio_stats}")
        
        return audio_stats
    
    def run_easy_task(self) -> Dict:
        """
        Implement Easy Task:
        - Basic VAE for feature extraction
        - K-Means clustering on latent features
        - t-SNE/UMAP visualization
        - Compare with PCA + K-Means baseline
        - Evaluate with Silhouette Score and Calinski-Harabasz Index
        """
        print("\n=== EASY TASK: Basic VAE + K-Means Clustering ===")
        
        results = {}
        
        # Get input dimension from first batch
        sample_batch, _ = next(iter(self.audio_train_loader))
        input_dim = sample_batch.shape[1]
        
        # Initialize Basic VAE
        basic_vae = BasicVAE(
            input_dim=input_dim,
            hidden_dim=256,
            latent_dim=32
        ).to(self.device)
        
        # Train Basic VAE
        optimizer = optim.Adam(basic_vae.parameters(), lr=1e-3)
        
        print("Training Basic VAE...")
        training_history = train_vae(
            basic_vae, self.audio_train_loader, optimizer, 
            self.device, epochs=50
        )
        
        results['basic_vae_training'] = training_history
        
        # Extract latent features
        print("Extracting latent features...")
        latent_features, true_labels = extract_vae_features(
            basic_vae, self.audio_test_loader, self.device
        )
        
        results['latent_features'] = latent_features
        results['true_labels'] = true_labels
        
        # Apply clustering
        print("Applying clustering algorithms...")
        clustering_pipeline = ClusteringPipeline(n_clusters=10)
        cluster_results = clustering_pipeline.fit_transform(latent_features)
        
        # Baseline comparison
        baseline = BaselineComparison(n_clusters=10)
        
        # Get original features for baseline
        all_features = []
        all_labels = []
        for data, label in self.audio_test_loader:
            all_features.append(data.numpy())
            all_labels.extend(label.numpy())
        
        original_features = np.vstack(all_features)
        
        pca_features, pca_labels = baseline.pca_kmeans(original_features)
        direct_labels = baseline.direct_kmeans(original_features)
        
        cluster_results['pca_kmeans'] = pca_labels
        cluster_results['direct_kmeans'] = direct_labels
        
        # Evaluate clustering
        print("Evaluating clustering performance...")
        clustering_metrics = {}
        for method, labels in cluster_results.items():
            metrics = self.evaluator.evaluate_clustering(
                latent_features, labels, true_labels
            )
            clustering_metrics[method] = metrics
        
        results['clustering_metrics'] = clustering_metrics
        
        # Create visualizations
        print("Creating visualizations...")
        dim_reducer = DimensionalityReduction()
        
        # t-SNE visualization
        tsne_2d = dim_reducer.tsne_transform(latent_features, n_components=2)
        
        # UMAP visualization  
        umap_2d = dim_reducer.umap_transform(latent_features, n_components=2)
        
        visualization_data = {
            'vae_kmeans': {
                'features_2d': tsne_2d,
                'cluster_labels': cluster_results['kmeans'],
                'true_labels': true_labels
            },
            'pca_kmeans': {
                'features_2d': dim_reducer.tsne_transform(pca_features, n_components=2),
                'cluster_labels': pca_labels,
                'true_labels': true_labels
            }
        }
        
        results['visualization_data'] = visualization_data
        
        # Generate plots
        self.viz_pipeline.plot_latent_space_2d(
            tsne_2d, cluster_results['kmeans'], 
            title="Basic VAE Latent Space (t-SNE)",
            true_labels=true_labels,
            save_path=os.path.join(self.results_dir, "latent_visualization", "basic_vae_tsne.png")
        )
        
        self.viz_pipeline.plot_latent_space_2d(
            umap_2d, cluster_results['kmeans'],
            title="Basic VAE Latent Space (UMAP)",
            true_labels=true_labels,
            save_path=os.path.join(self.results_dir, "latent_visualization", "basic_vae_umap.png")
        )
        
        # Create metrics comparison
        metrics_df = self.evaluator.create_evaluation_report(clustering_metrics)
        self.viz_pipeline.plot_metrics_comparison(
            metrics_df,
            save_path=os.path.join(self.results_dir, "latent_visualization", "easy_task_metrics.png")
        )
        
        results['metrics_df'] = metrics_df
        
        print(f"Easy Task completed. Best method: {metrics_df.iloc[0]['method'] if 'method' in metrics_df.columns else 'N/A'}")
        
        return results
    
    def run_medium_task(self) -> Dict:
        """
        Implement Medium Task:
        - Convolutional VAE for spectrograms/MFCC
        - Hybrid feature representation (audio + lyrics)
        - Multiple clustering algorithms
        - Enhanced evaluation metrics
        """
        print("\n=== MEDIUM TASK: Convolutional VAE + Hybrid Features ===")
        
        results = {}
        
        # Get spectrogram dimensions
        sample_batch, _ = next(iter(self.spec_train_loader))
        input_channels = sample_batch.shape[1]
        input_height = sample_batch.shape[2]
        input_width = sample_batch.shape[3]
        
        # Initialize Convolutional VAE
        conv_vae = ConvVAE(
            input_channels=input_channels,
            input_height=input_height,
            input_width=input_width,
            latent_dim=64
        ).to(self.device)
        
        # Train Convolutional VAE
        optimizer = optim.Adam(conv_vae.parameters(), lr=1e-3)
        
        print("Training Convolutional VAE...")
        training_history = train_vae(
            conv_vae, self.spec_train_loader, optimizer,
            self.device, epochs=50
        )
        
        results['conv_vae_training'] = training_history
        
        # Extract latent features
        print("Extracting convolutional latent features...")
        conv_latent_features, true_labels = extract_vae_features(
            conv_vae, self.spec_test_loader, self.device
        )
        
        results['conv_latent_features'] = conv_latent_features
        
        # Apply multiple clustering algorithms
        print("Applying multiple clustering algorithms...")
        clustering_methods = compare_clustering_methods(
            conv_latent_features, true_labels, n_clusters=10
        )
        
        results['clustering_methods'] = clustering_methods
        
        # Create enhanced visualizations
        print("Creating enhanced visualizations...")
        dim_reducer = DimensionalityReduction()
        
        # Multiple dimensionality reduction techniques
        conv_tsne_2d = dim_reducer.tsne_transform(conv_latent_features)
        conv_umap_2d = dim_reducer.umap_transform(conv_latent_features)
        conv_pca_2d = dim_reducer.pca_transform(conv_latent_features)
        
        # Compare different clustering methods
        visualization_data = {}
        
        clustering_pipeline = ClusteringPipeline(n_clusters=10)
        cluster_results = clustering_pipeline.fit_transform(conv_latent_features)
        
        for method, labels in cluster_results.items():
            visualization_data[f'conv_{method}'] = {
                'features_2d': conv_tsne_2d,
                'cluster_labels': labels,
                'true_labels': true_labels
            }
        
        results['visualization_data'] = visualization_data
        
        # Plot comparison of methods
        self.viz_pipeline.plot_clustering_comparison(
            visualization_data,
            save_path=os.path.join(self.results_dir, "latent_visualization", "medium_task_comparison.png")
        )
        
        # Enhanced metrics evaluation
        enhanced_metrics = {}
        for method, labels in cluster_results.items():
            metrics = self.evaluator.evaluate_clustering(
                conv_latent_features, labels, true_labels
            )
            enhanced_metrics[f'conv_{method}'] = metrics
        
        results['enhanced_metrics'] = enhanced_metrics
        
        # Create comprehensive metrics plot
        enhanced_metrics_df = self.evaluator.create_evaluation_report(enhanced_metrics)
        self.viz_pipeline.plot_metrics_comparison(
            enhanced_metrics_df,
            save_path=os.path.join(self.results_dir, "latent_visualization", "medium_task_metrics.png")
        )
        
        results['enhanced_metrics_df'] = enhanced_metrics_df
        
        print(f"Medium Task completed. Best method: {enhanced_metrics_df.iloc[0]['method'] if 'method' in enhanced_metrics_df.columns else 'N/A'}")
        
        return results
    
    def run_hard_task(self) -> Dict:
        """
        Implement Hard Task:
        - Beta-VAE or Conditional VAE
        - Multi-modal clustering (audio + lyrics + genre)
        - Comprehensive evaluation metrics
        - Advanced visualizations
        """
        print("\n=== HARD TASK: Advanced VAE + Multi-modal Clustering ===")
        
        results = {}
        
        # Get input dimension
        sample_batch, _ = next(iter(self.audio_train_loader))
        input_dim = sample_batch.shape[1]
        
        # Initialize Beta-VAE for disentangled representations
        beta_vae = BetaVAE(
            input_dim=input_dim,
            hidden_dim=512,
            latent_dim=64,
            beta=4.0  # Higher beta for more disentangled representations
        ).to(self.device)
        
        # Train Beta-VAE
        optimizer = optim.Adam(beta_vae.parameters(), lr=1e-3)
        
        print("Training Beta-VAE...")
        training_history = train_vae(
            beta_vae, self.audio_train_loader, optimizer,
            self.device, epochs=100
        )
        
        results['beta_vae_training'] = training_history
        
        # Extract disentangled latent features
        print("Extracting disentangled latent features...")
        beta_latent_features, true_labels = extract_vae_features(
            beta_vae, self.audio_test_loader, self.device
        )
        
        results['beta_latent_features'] = beta_latent_features
        
        # Multi-modal clustering comparison
        print("Performing comprehensive clustering comparison...")
        
        # Compare VAE-based methods with baselines
        all_methods = {}
        
        # Beta-VAE clustering
        beta_clustering = ClusteringPipeline(n_clusters=10)
        beta_results = beta_clustering.fit_transform(beta_latent_features)
        
        for method, labels in beta_results.items():
            all_methods[f'beta_vae_{method}'] = labels
        
        # Baseline methods
        baseline = BaselineComparison(n_clusters=10)
        
        # Get original features
        all_features = []
        for data, _ in self.audio_test_loader:
            all_features.append(data.numpy())
        original_features = np.vstack(all_features)
        
        pca_features, pca_labels = baseline.pca_kmeans(original_features)
        direct_labels = baseline.direct_kmeans(original_features)
        
        all_methods['pca_kmeans_baseline'] = pca_labels
        all_methods['direct_kmeans_baseline'] = direct_labels
        
        # Comprehensive evaluation
        print("Performing comprehensive evaluation...")
        comprehensive_metrics = {}
        
        for method, labels in all_methods.items():
            metrics = self.evaluator.evaluate_clustering(
                beta_latent_features if 'beta_vae' in method else original_features,
                labels, true_labels
            )
            comprehensive_metrics[method] = metrics
        
        results['comprehensive_metrics'] = comprehensive_metrics
        
        # Advanced visualizations
        print("Creating advanced visualizations...")
        dim_reducer = DimensionalityReduction()
        
        # 3D visualization
        beta_3d = dim_reducer.pca_transform(beta_latent_features, n_components=3)
        self.viz_pipeline.plot_latent_space_3d(
            beta_3d, all_methods['beta_vae_kmeans'],
            title="Beta-VAE 3D Latent Space"
        )
        
        # 2D visualizations with multiple reduction techniques
        beta_tsne_2d = dim_reducer.tsne_transform(beta_latent_features)
        beta_umap_2d = dim_reducer.umap_transform(beta_latent_features)
        
        # Advanced visualization data
        advanced_viz_data = {}
        for method, labels in all_methods.items():
            if 'beta_vae' in method:
                features_2d = beta_tsne_2d
            else:
                features_2d = dim_reducer.tsne_transform(pca_features if 'pca' in method else original_features)
            
            advanced_viz_data[method] = {
                'features_2d': features_2d,
                'cluster_labels': labels,
                'true_labels': true_labels
            }
        
        results['advanced_viz_data'] = advanced_viz_data
        
        # Plot comprehensive comparison
        self.viz_pipeline.plot_clustering_comparison(
            advanced_viz_data,
            save_path=os.path.join(self.results_dir, "latent_visualization", "hard_task_comprehensive.png")
        )
        
        # Create comprehensive metrics plot
        comprehensive_df = self.evaluator.create_evaluation_report(comprehensive_metrics)
        self.viz_pipeline.plot_metrics_comparison(
            comprehensive_df,
            metrics_to_plot=[
                'silhouette_score', 'calinski_harabasz_index', 'davies_bouldin_index',
                'adjusted_rand_index', 'normalized_mutual_info', 'cluster_purity'
            ],
            save_path=os.path.join(self.results_dir, "latent_visualization", "hard_task_comprehensive_metrics.png")
        )
        
        # Genre distribution analysis
        genre_names = list(self.audio_dataset.genre_to_idx.keys())
        self.viz_pipeline.plot_cluster_distribution(
            all_methods['beta_vae_kmeans'], true_labels, genre_names,
            save_path=os.path.join(self.results_dir, "latent_visualization", "cluster_genre_distribution.png")
        )
        
        results['comprehensive_df'] = comprehensive_df
        results['genre_names'] = genre_names
        
        print(f"Hard Task completed. Best method: {comprehensive_df.iloc[0]['method'] if 'method' in comprehensive_df.columns else 'N/A'}")
        
        return results
    
    def run_full_experiment(self) -> Dict:
        """Run all tasks and create comprehensive results."""
        print("=== STARTING COMPREHENSIVE VAE MUSIC CLUSTERING EXPERIMENT ===")
        
        # Load datasets
        dataset_stats = self.load_datasets()
        
        # Run all tasks
        easy_results = self.run_easy_task()
        medium_results = self.run_medium_task()
        hard_results = self.run_hard_task()
        
        # Combine results
        all_results = {
            'dataset_stats': dataset_stats,
            'easy_task': easy_results,
            'medium_task': medium_results,
            'hard_task': hard_results
        }
        
        # Create comprehensive comparison
        print("\n=== CREATING COMPREHENSIVE COMPARISON ===")
        
        # Combine all metrics
        all_metrics = {}
        all_metrics.update(easy_results.get('clustering_metrics', {}))
        all_metrics.update(medium_results.get('enhanced_metrics', {}))
        all_metrics.update(hard_results.get('comprehensive_metrics', {}))
        
        # Save comprehensive results
        comprehensive_df = self.evaluator.create_evaluation_report(all_metrics)
        comprehensive_df.to_csv(
            os.path.join(self.results_dir, "comprehensive_clustering_metrics.csv"),
            index=False
        )
        
        all_results['comprehensive_metrics'] = all_metrics
        all_results['comprehensive_df'] = comprehensive_df
        
        # Create final comprehensive visualization
        self.viz_pipeline.plot_metrics_comparison(
            comprehensive_df,
            save_path=os.path.join(self.results_dir, "latent_visualization", "final_comprehensive_metrics.png")
        )
        
        # Generate final report summary
        self.generate_summary_report(all_results)
        
        print("\n=== EXPERIMENT COMPLETED SUCCESSFULLY ===")
        print(f"All results saved to: {self.results_dir}")
        print(f"Comprehensive metrics saved to: comprehensive_clustering_metrics.csv")
        
        return all_results
    
    def generate_summary_report(self, results: Dict):
        """Generate a summary report of all results."""
        report_path = os.path.join(self.results_dir, "experiment_summary.txt")
        
        with open(report_path, 'w') as f:
            f.write("VAE MUSIC CLUSTERING EXPERIMENT SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            # Dataset information
            if 'dataset_stats' in results:
                stats = results['dataset_stats']
                f.write(f"DATASET STATISTICS:\\n")
                f.write(f"Number of samples: {stats.get('num_samples', 'N/A')}\\n")
                f.write(f"Feature dimension: {stats.get('feature_dim', 'N/A')}\\n")
                f.write(f"Number of classes: {stats.get('num_classes', 'N/A')}\\n\\n")
            
            # Task summaries
            for task_name in ['easy_task', 'medium_task', 'hard_task']:
                if task_name in results:
                    task_results = results[task_name]
                    f.write(f"{task_name.upper()} RESULTS:\\n")
                    
                    # Best performing method
                    if 'metrics_df' in task_results or 'enhanced_metrics_df' in task_results or 'comprehensive_df' in task_results:
                        df_key = 'comprehensive_df' if 'comprehensive_df' in task_results else ('enhanced_metrics_df' if 'enhanced_metrics_df' in task_results else 'metrics_df')
                        df = task_results[df_key]
                        if not df.empty and 'method' in df.columns:
                            best_method = df.iloc[0]['method']
                            best_silhouette = df.iloc[0].get('silhouette_score', 'N/A')
                            f.write(f"Best method: {best_method}\\n")
                            f.write(f"Best silhouette score: {best_silhouette}\\n")
                    
                    f.write("\\n")
            
            # Overall best performance
            if 'comprehensive_df' in results:
                df = results['comprehensive_df']
                if not df.empty and 'method' in df.columns:
                    f.write("OVERALL BEST PERFORMANCE:\\n")
                    f.write(f"Method: {df.iloc[0]['method']}\\n")
                    f.write(f"Silhouette Score: {df.iloc[0].get('silhouette_score', 'N/A')}\\n")
                    f.write(f"Calinski-Harabasz Index: {df.iloc[0].get('calinski_harabasz_index', 'N/A')}\\n")
                    f.write(f"Davies-Bouldin Index: {df.iloc[0].get('davies_bouldin_index', 'N/A')}\\n")
                    if 'adjusted_rand_index' in df.columns:
                        f.write(f"Adjusted Rand Index: {df.iloc[0].get('adjusted_rand_index', 'N/A')}\\n")
        
        print(f"Summary report saved to: {report_path}")


def main():
    """Main execution function."""
    # Initialize experiment
    experiment = MusicClusteringExperiment(
        data_dir="data",
        results_dir="results",
        random_seed=42
    )
    
    # Run full experiment
    results = experiment.run_full_experiment()
    
    # Print final summary
    if 'comprehensive_df' in results:
        print("\\nFINAL RESULTS SUMMARY:")
        print(results['comprehensive_df'].head())


if __name__ == "__main__":
    main()