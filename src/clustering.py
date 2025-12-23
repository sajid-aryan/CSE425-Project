import numpy as np
import torch
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class ClusteringPipeline:
    """Pipeline for clustering VAE latent representations."""
    
    def __init__(self, n_clusters: int = 10, random_state: int = 42):
        """
        Initialize clustering pipeline.
        
        Args:
            n_clusters: Number of clusters
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.clusterers = {}
        self.cluster_results = {}
        
    def prepare_clustering_algorithms(self) -> Dict[str, Any]:
        """Prepare different clustering algorithms."""
        algorithms = {
            'kmeans': KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10),
            'agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
        return algorithms
    
    def fit_transform(self, features: np.ndarray, normalize: bool = True) -> Dict[str, np.ndarray]:
        """
        Apply clustering algorithms to features.
        
        Args:
            features: Input features to cluster
            normalize: Whether to normalize features
            
        Returns:
            Dictionary with clustering results for each algorithm
        """
        if normalize:
            features_scaled = self.scaler.fit_transform(features)
        else:
            features_scaled = features
        
        algorithms = self.prepare_clustering_algorithms()
        results = {}
        
        for name, algorithm in algorithms.items():
            try:
                if name == 'dbscan':
                    labels = algorithm.fit_predict(features_scaled)
                    # Handle DBSCAN's -1 labels (noise points)
                    unique_labels = set(labels)
                    if -1 in unique_labels:
                        unique_labels.remove(-1)
                    if len(unique_labels) == 0:
                        # If all points are noise, create single cluster
                        labels = np.zeros_like(labels)
                else:
                    labels = algorithm.fit_predict(features_scaled)
                
                results[name] = labels
                self.clusterers[name] = algorithm
                
            except Exception as e:
                print(f"Error with {name} clustering: {e}")
                results[name] = np.zeros(len(features))
        
        self.cluster_results = results
        return results
    
    def predict(self, features: np.ndarray, algorithm: str = 'kmeans') -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            features: New features to cluster
            algorithm: Clustering algorithm to use
            
        Returns:
            Predicted cluster labels
        """
        if algorithm not in self.clusterers:
            raise ValueError(f"Algorithm {algorithm} not fitted")
        
        features_scaled = self.scaler.transform(features)
        
        if algorithm == 'dbscan':
            return self.clusterers[algorithm].fit_predict(features_scaled)
        else:
            return self.clusterers[algorithm].predict(features_scaled)


class BaselineComparison:
    """Baseline clustering methods for comparison."""
    
    def __init__(self, n_components: int = 50, n_clusters: int = 10, random_state: int = 42):
        """
        Initialize baseline comparison.
        
        Args:
            n_components: Number of PCA components
            n_clusters: Number of clusters
            random_state: Random seed
        """
        self.n_components = n_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.pca = PCA(n_components=n_components, random_state=random_state)
        self.scaler = StandardScaler()
        
    def pca_kmeans(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply PCA + K-Means baseline.
        
        Args:
            features: Input features
            
        Returns:
            Tuple of (PCA features, cluster labels)
        """
        # Normalize features
        features_scaled = self.scaler.fit_transform(features)
        
        # Apply PCA
        pca_features = self.pca.fit_transform(features_scaled)
        
        # Apply K-Means
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        labels = kmeans.fit_predict(pca_features)
        
        return pca_features, labels
    
    def direct_kmeans(self, features: np.ndarray) -> np.ndarray:
        """
        Apply direct K-Means to features.
        
        Args:
            features: Input features
            
        Returns:
            Cluster labels
        """
        features_scaled = self.scaler.fit_transform(features)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        return kmeans.fit_predict(features_scaled)


class DimensionalityReduction:
    """Dimensionality reduction for visualization."""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize dimensionality reduction.
        
        Args:
            random_state: Random seed
        """
        self.random_state = random_state
    
    def tsne_transform(self, features: np.ndarray, n_components: int = 2, 
                      perplexity: float = 30.0) -> np.ndarray:
        """
        Apply t-SNE transformation.
        
        Args:
            features: Input features
            n_components: Number of output dimensions
            perplexity: t-SNE perplexity parameter
            
        Returns:
            t-SNE transformed features
        """
        tsne = TSNE(n_components=n_components, perplexity=perplexity, 
                   random_state=self.random_state, n_iter=1000)
        return tsne.fit_transform(features)
    
    def umap_transform(self, features: np.ndarray, n_components: int = 2, 
                      n_neighbors: int = 15, min_dist: float = 0.1) -> np.ndarray:
        """
        Apply UMAP transformation.
        
        Args:
            features: Input features
            n_components: Number of output dimensions
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            
        Returns:
            UMAP transformed features
        """
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors,
                           min_dist=min_dist, random_state=self.random_state)
        return reducer.fit_transform(features)
    
    def pca_transform(self, features: np.ndarray, n_components: int = 2) -> np.ndarray:
        """
        Apply PCA transformation.
        
        Args:
            features: Input features
            n_components: Number of output dimensions
            
        Returns:
            PCA transformed features
        """
        pca = PCA(n_components=n_components, random_state=self.random_state)
        return pca.fit_transform(features)


def extract_vae_features(model: torch.nn.Module, data_loader, 
                        device: str = 'cpu') -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract latent features from VAE model.
    
    Args:
        model: Trained VAE model
        data_loader: Data loader for inference
        device: Device to run inference on
        
    Returns:
        Tuple of (latent features, labels)
    """
    model.eval()
    latent_features = []
    labels = []
    
    with torch.no_grad():
        for data, label in data_loader:
            data = data.to(device)
            
            # Get latent features (mu)
            if hasattr(model, 'encode'):
                mu, _ = model.encode(data)
            else:
                output = model(data)
                mu = output['mu']
            
            latent_features.append(mu.cpu().numpy())
            labels.extend(label.numpy())
    
    latent_features = np.vstack(latent_features)
    labels = np.array(labels)
    
    return latent_features, labels


def compare_clustering_methods(features: np.ndarray, true_labels: Optional[np.ndarray] = None,
                             n_clusters: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Compare different clustering methods and their performance.
    
    Args:
        features: Input features to cluster
        true_labels: Ground truth labels (optional)
        n_clusters: Number of clusters
        
    Returns:
        Dictionary with clustering results and metrics
    """
    from .evaluation import ClusteringEvaluator
    
    # Initialize clustering pipeline
    clustering = ClusteringPipeline(n_clusters=n_clusters)
    
    # Apply clustering algorithms
    cluster_results = clustering.fit_transform(features)
    
    # Initialize baseline comparison
    baseline = BaselineComparison(n_clusters=n_clusters)
    
    # Apply baseline methods
    pca_features, pca_labels = baseline.pca_kmeans(features)
    direct_labels = baseline.direct_kmeans(features)
    
    # Add baseline results
    cluster_results['pca_kmeans'] = pca_labels
    cluster_results['direct_kmeans'] = direct_labels
    
    # Evaluate clustering performance
    evaluator = ClusteringEvaluator()
    results = {}
    
    for method_name, labels in cluster_results.items():
        metrics = evaluator.evaluate_clustering(features, labels, true_labels)
        results[method_name] = metrics
    
    return results


def create_clustering_visualization_data(features: np.ndarray, labels: np.ndarray,
                                       true_labels: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Create data for clustering visualization.
    
    Args:
        features: Input features
        labels: Cluster labels
        true_labels: Ground truth labels (optional)
        
    Returns:
        Dictionary with visualization data
    """
    # Apply dimensionality reduction
    dim_reducer = DimensionalityReduction()
    
    # Get 2D representations
    tsne_2d = dim_reducer.tsne_transform(features, n_components=2)
    umap_2d = dim_reducer.umap_transform(features, n_components=2)
    pca_2d = dim_reducer.pca_transform(features, n_components=2)
    
    visualization_data = {
        'tsne_2d': tsne_2d,
        'umap_2d': umap_2d,
        'pca_2d': pca_2d,
        'cluster_labels': labels
    }
    
    if true_labels is not None:
        visualization_data['true_labels'] = true_labels
    
    return visualization_data