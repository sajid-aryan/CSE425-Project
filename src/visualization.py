import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class VisualizationPipeline:
    """Comprehensive visualization pipeline for VAE clustering results."""
    
    def __init__(self, style: str = 'whitegrid', palette: str = 'Set1', 
                 figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize visualization pipeline.
        
        Args:
            style: Seaborn style
            palette: Color palette
            figsize: Default figure size
        """
        self.style = style
        self.palette = palette
        self.figsize = figsize
        
        # Set style
        sns.set_style(style)
        plt.rcParams['figure.figsize'] = figsize
    
    def plot_latent_space_2d(self, features_2d: np.ndarray, labels: np.ndarray,
                           title: str = "Latent Space Visualization", 
                           true_labels: Optional[np.ndarray] = None,
                           save_path: Optional[str] = None) -> None:
        """
        Plot 2D latent space visualization.
        
        Args:
            features_2d: 2D features for plotting
            labels: Cluster labels
            title: Plot title
            true_labels: Ground truth labels (optional)
            save_path: Path to save plot (optional)
        """
        if true_labels is not None:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            
            # Plot predicted clusters
            scatter = axes[0].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=labels, cmap='tab10', alpha=0.7)
            axes[0].set_title(f"{title} - Predicted Clusters")
            axes[0].set_xlabel("Dimension 1")
            axes[0].set_ylabel("Dimension 2")
            plt.colorbar(scatter, ax=axes[0])
            
            # Plot true labels
            scatter = axes[1].scatter(features_2d[:, 0], features_2d[:, 1], 
                                    c=true_labels, cmap='tab10', alpha=0.7)
            axes[1].set_title(f"{title} - True Labels")
            axes[1].set_xlabel("Dimension 1")
            axes[1].set_ylabel("Dimension 2")
            plt.colorbar(scatter, ax=axes[1])
        else:
            fig, ax = plt.subplots(1, 1, figsize=self.figsize)
            scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                               c=labels, cmap='tab10', alpha=0.7)
            ax.set_title(title)
            ax.set_xlabel("Dimension 1")
            ax.set_ylabel("Dimension 2")
            plt.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_latent_space_3d(self, features_3d: np.ndarray, labels: np.ndarray,
                           title: str = "3D Latent Space Visualization") -> None:
        """
        Plot 3D latent space visualization using Plotly.
        
        Args:
            features_3d: 3D features for plotting
            labels: Cluster labels
            title: Plot title
        """
        fig = go.Figure(data=[go.Scatter3d(
            x=features_3d[:, 0],
            y=features_3d[:, 1],
            z=features_3d[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=labels,
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title="Cluster")
            )
        )])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='Dimension 1',
                yaxis_title='Dimension 2',
                zaxis_title='Dimension 3'
            ),
            width=800,
            height=600
        )
        
        fig.show()
    
    def plot_clustering_comparison(self, visualization_data: Dict[str, Dict[str, np.ndarray]],
                                 methods: List[str] = None,
                                 save_path: Optional[str] = None) -> None:
        """
        Compare different clustering methods side by side.
        
        Args:
            visualization_data: Dictionary with visualization data for each method
            methods: List of methods to compare
            save_path: Path to save plot (optional)
        """
        if methods is None:
            methods = list(visualization_data.keys())
        
        n_methods = len(methods)
        cols = min(3, n_methods)
        rows = (n_methods + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if n_methods == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, method in enumerate(methods):
            if method in visualization_data:
                data = visualization_data[method]
                features_2d = data.get('features_2d', data.get('tsne_2d'))
                labels = data.get('cluster_labels', data.get('labels'))
                
                if features_2d is not None and labels is not None:
                    ax = axes[i] if i < len(axes) else axes[-1]
                    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], 
                                       c=labels, cmap='tab10', alpha=0.7, s=30)
                    ax.set_title(f"{method}")
                    ax.set_xlabel("Dimension 1")
                    ax.set_ylabel("Dimension 2")
        
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_metrics_comparison(self, metrics_df: pd.DataFrame,
                              metrics_to_plot: List[str] = None,
                              save_path: Optional[str] = None) -> None:
        """
        Plot comparison of clustering metrics across methods.
        
        Args:
            metrics_df: DataFrame with clustering metrics
            metrics_to_plot: List of metrics to plot
            save_path: Path to save plot (optional)
        """
        if metrics_to_plot is None:
            # Default metrics to plot
            metrics_to_plot = [
                'silhouette_score', 'calinski_harabasz_index', 
                'davies_bouldin_index', 'adjusted_rand_index'
            ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics_to_plot if m in metrics_df.columns]
        
        if not available_metrics:
            print("No valid metrics found in DataFrame")
            return
        
        n_metrics = len(available_metrics)
        cols = 2
        rows = (n_metrics + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(12, 4*rows))
        if n_metrics == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, metric in enumerate(available_metrics):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            # Create bar plot
            methods = metrics_df['method'] if 'method' in metrics_df.columns else metrics_df.index
            values = metrics_df[metric]
            
            bars = ax.bar(range(len(methods)), values, color=sns.color_palette(self.palette, len(methods)))
            ax.set_title(f"{metric.replace('_', ' ').title()}")
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45, ha='right')
            ax.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, value) in enumerate(zip(bars, values)):
                if not np.isnan(value) and not np.isinf(value):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_vae_reconstructions(self, original: np.ndarray, reconstructed: np.ndarray,
                               n_samples: int = 5, save_path: Optional[str] = None) -> None:
        """
        Plot VAE reconstruction examples.
        
        Args:
            original: Original data
            reconstructed: Reconstructed data
            n_samples: Number of samples to show
            save_path: Path to save plot (optional)
        """
        n_samples = min(n_samples, len(original))
        indices = np.random.choice(len(original), n_samples, replace=False)
        
        fig, axes = plt.subplots(2, n_samples, figsize=(2*n_samples, 4))
        if n_samples == 1:
            axes = axes.reshape(2, 1)
        
        for i, idx in enumerate(indices):
            # Original
            if len(original[idx].shape) == 2:  # 2D data (e.g., spectrogram)
                axes[0, i].imshow(original[idx], aspect='auto', cmap='viridis')
                axes[1, i].imshow(reconstructed[idx], aspect='auto', cmap='viridis')
            else:  # 1D data
                axes[0, i].plot(original[idx])
                axes[1, i].plot(reconstructed[idx])
            
            if i == 0:
                axes[0, i].set_ylabel('Original')
                axes[1, i].set_ylabel('Reconstructed')
            
            axes[0, i].set_title(f'Sample {idx}')
            axes[0, i].set_xticks([])
            axes[1, i].set_xticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, training_history: Dict[str, List[float]],
                            save_path: Optional[str] = None) -> None:
        """
        Plot VAE training history.
        
        Args:
            training_history: Dictionary with training metrics
            save_path: Path to save plot (optional)
        """
        metrics = list(training_history.keys())
        n_metrics = len(metrics)
        
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 4))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            values = training_history[metric]
            axes[i].plot(values)
            axes[i].set_title(f"{metric.replace('_', ' ').title()}")
            axes[i].set_xlabel("Epoch")
            axes[i].set_ylabel(metric)
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_cluster_distribution(self, labels: np.ndarray, true_labels: Optional[np.ndarray] = None,
                                genre_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None) -> None:
        """
        Plot cluster size distribution and genre distribution within clusters.
        
        Args:
            labels: Cluster labels
            true_labels: Ground truth labels (optional)
            genre_names: Names for true label categories
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(1, 2 if true_labels is not None else 1, 
                               figsize=(12 if true_labels is not None else 6, 4))
        if true_labels is None:
            axes = [axes]
        
        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        axes[0].bar(unique, counts, color=sns.color_palette(self.palette, len(unique)))
        axes[0].set_title("Cluster Size Distribution")
        axes[0].set_xlabel("Cluster ID")
        axes[0].set_ylabel("Number of Samples")
        axes[0].grid(axis='y', alpha=0.3)
        
        # Genre distribution within clusters
        if true_labels is not None:
            cluster_genre_matrix = []
            unique_clusters = np.unique(labels)
            unique_genres = np.unique(true_labels)
            
            for cluster in unique_clusters:
                cluster_mask = labels == cluster
                cluster_true_labels = true_labels[cluster_mask]
                genre_counts = []
                for genre in unique_genres:
                    genre_counts.append(np.sum(cluster_true_labels == genre))
                cluster_genre_matrix.append(genre_counts)
            
            cluster_genre_matrix = np.array(cluster_genre_matrix)
            
            # Normalize by cluster size
            cluster_genre_matrix_norm = cluster_genre_matrix / cluster_genre_matrix.sum(axis=1, keepdims=True)
            
            im = axes[1].imshow(cluster_genre_matrix_norm.T, aspect='auto', cmap='YlOrRd')
            axes[1].set_title("Genre Distribution within Clusters")
            axes[1].set_xlabel("Cluster ID")
            axes[1].set_ylabel("Genre")
            
            if genre_names:
                axes[1].set_yticks(range(len(genre_names)))
                axes[1].set_yticklabels(genre_names, rotation=45)
            else:
                axes[1].set_yticks(range(len(unique_genres)))
                axes[1].set_yticklabels([f"Genre {i}" for i in unique_genres], rotation=45)
            
            axes[1].set_xticks(range(len(unique_clusters)))
            axes[1].set_xticklabels([f"C{i}" for i in unique_clusters])
            
            plt.colorbar(im, ax=axes[1], label='Proportion')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_latent_space(self, features_2d: np.ndarray, labels: np.ndarray,
                                      true_labels: Optional[np.ndarray] = None,
                                      sample_info: Optional[List[str]] = None) -> None:
        """
        Create interactive latent space visualization using Plotly.
        
        Args:
            features_2d: 2D features for plotting
            labels: Cluster labels
            true_labels: Ground truth labels (optional)
            sample_info: Additional sample information (optional)
        """
        # Create DataFrame for easier plotting
        plot_data = pd.DataFrame({
            'dim1': features_2d[:, 0],
            'dim2': features_2d[:, 1],
            'cluster': labels.astype(str),
        })
        
        if true_labels is not None:
            plot_data['true_label'] = true_labels.astype(str)
        
        if sample_info is not None:
            plot_data['info'] = sample_info
        
        # Create main plot
        fig = px.scatter(
            plot_data, 
            x='dim1', 
            y='dim2', 
            color='cluster',
            hover_data=['true_label'] if true_labels is not None else None,
            title="Interactive Latent Space Visualization"
        )
        
        fig.update_layout(
            xaxis_title="Dimension 1",
            yaxis_title="Dimension 2",
            width=800,
            height=600
        )
        
        fig.show()
    
    def save_all_visualizations(self, results_dict: Dict[str, Any], 
                              output_dir: str = "results/latent_visualization") -> None:
        """
        Save all visualizations to files.
        
        Args:
            results_dict: Dictionary with all results and data
            output_dir: Output directory for saved plots
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save latent space visualizations
        if 'visualization_data' in results_dict:
            for method, data in results_dict['visualization_data'].items():
                if 'features_2d' in data and 'cluster_labels' in data:
                    save_path = os.path.join(output_dir, f"{method}_latent_space.png")
                    self.plot_latent_space_2d(
                        data['features_2d'], 
                        data['cluster_labels'],
                        title=f"{method} Latent Space",
                        save_path=save_path
                    )
        
        # Save metrics comparison
        if 'metrics_comparison' in results_dict:
            save_path = os.path.join(output_dir, "metrics_comparison.png")
            self.plot_metrics_comparison(
                results_dict['metrics_comparison'],
                save_path=save_path
            )
        
        # Save training history
        if 'training_history' in results_dict:
            save_path = os.path.join(output_dir, "training_history.png")
            self.plot_training_history(
                results_dict['training_history'],
                save_path=save_path
            )
        
        print(f"All visualizations saved to {output_dir}")


def create_comprehensive_report_plots(results: Dict[str, Any], 
                                     output_dir: str = "results/latent_visualization") -> None:
    """
    Create all plots needed for the comprehensive report.
    
    Args:
        results: Dictionary with all experimental results
        output_dir: Output directory for plots
    """
    viz = VisualizationPipeline()
    viz.save_all_visualizations(results, output_dir)