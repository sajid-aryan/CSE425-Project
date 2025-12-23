import numpy as np
import pandas as pd
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, homogeneity_score,
    completeness_score, v_measure_score
)
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class ClusteringEvaluator:
    """Comprehensive evaluation of clustering performance."""
    
    def __init__(self):
        """Initialize clustering evaluator."""
        self.metrics = {}
    
    def silhouette_score_metric(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Silhouette Score.
        
        Args:
            features: Input features
            labels: Cluster labels
            
        Returns:
            Silhouette score (higher is better, range: -1 to 1)
        """
        if len(set(labels)) < 2:
            return 0.0
        try:
            return silhouette_score(features, labels)
        except:
            return 0.0
    
    def calinski_harabasz_index(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Calinski-Harabasz Index.
        
        Args:
            features: Input features
            labels: Cluster labels
            
        Returns:
            Calinski-Harabasz index (higher is better)
        """
        if len(set(labels)) < 2:
            return 0.0
        try:
            return calinski_harabasz_score(features, labels)
        except:
            return 0.0
    
    def davies_bouldin_index(self, features: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate Davies-Bouldin Index.
        
        Args:
            features: Input features
            labels: Cluster labels
            
        Returns:
            Davies-Bouldin index (lower is better)
        """
        if len(set(labels)) < 2:
            return float('inf')
        try:
            return davies_bouldin_score(features, labels)
        except:
            return float('inf')
    
    def adjusted_rand_index(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate Adjusted Rand Index.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Adjusted Rand Index (higher is better, range: -1 to 1)
        """
        try:
            return adjusted_rand_score(true_labels, pred_labels)
        except:
            return 0.0
    
    def normalized_mutual_info(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate Normalized Mutual Information.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Normalized Mutual Information (higher is better, range: 0 to 1)
        """
        try:
            return normalized_mutual_info_score(true_labels, pred_labels)
        except:
            return 0.0
    
    def cluster_purity(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
        """
        Calculate Cluster Purity.
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Cluster purity (higher is better, range: 0 to 1)
        """
        # Convert to numpy arrays
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        # Get unique clusters
        unique_clusters = np.unique(pred_labels)
        total_correct = 0
        total_samples = len(true_labels)
        
        for cluster in unique_clusters:
            # Get points in this cluster
            cluster_mask = pred_labels == cluster
            cluster_true_labels = true_labels[cluster_mask]
            
            if len(cluster_true_labels) > 0:
                # Find the most common true label in this cluster
                unique, counts = np.unique(cluster_true_labels, return_counts=True)
                max_count = np.max(counts)
                total_correct += max_count
        
        return total_correct / total_samples if total_samples > 0 else 0.0
    
    def v_measure_metrics(self, true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate V-measure and its components (homogeneity and completeness).
        
        Args:
            true_labels: Ground truth labels
            pred_labels: Predicted cluster labels
            
        Returns:
            Dictionary with homogeneity, completeness, and v_measure scores
        """
        try:
            homogeneity = homogeneity_score(true_labels, pred_labels)
            completeness = completeness_score(true_labels, pred_labels)
            v_measure = v_measure_score(true_labels, pred_labels)
            
            return {
                'homogeneity': homogeneity,
                'completeness': completeness,
                'v_measure': v_measure
            }
        except:
            return {
                'homogeneity': 0.0,
                'completeness': 0.0,
                'v_measure': 0.0
            }
    
    def evaluate_clustering(self, features: np.ndarray, cluster_labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation.
        
        Args:
            features: Input features used for clustering
            cluster_labels: Predicted cluster labels
            true_labels: Ground truth labels (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        results = {}
        
        # Unsupervised metrics (don't require true labels)
        results['silhouette_score'] = self.silhouette_score_metric(features, cluster_labels)
        results['calinski_harabasz_index'] = self.calinski_harabasz_index(features, cluster_labels)
        results['davies_bouldin_index'] = self.davies_bouldin_index(features, cluster_labels)
        
        # Supervised metrics (require true labels)
        if true_labels is not None:
            results['adjusted_rand_index'] = self.adjusted_rand_index(true_labels, cluster_labels)
            results['normalized_mutual_info'] = self.normalized_mutual_info(true_labels, cluster_labels)
            results['cluster_purity'] = self.cluster_purity(true_labels, cluster_labels)
            
            # V-measure components
            v_metrics = self.v_measure_metrics(true_labels, cluster_labels)
            results.update(v_metrics)
        
        # Additional statistics
        results['num_clusters'] = len(np.unique(cluster_labels))
        results['num_samples'] = len(cluster_labels)
        
        # Cluster size statistics
        unique, counts = np.unique(cluster_labels, return_counts=True)
        results['min_cluster_size'] = int(np.min(counts))
        results['max_cluster_size'] = int(np.max(counts))
        results['avg_cluster_size'] = float(np.mean(counts))
        results['cluster_size_std'] = float(np.std(counts))
        
        return results
    
    def create_evaluation_report(self, results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Create a comprehensive evaluation report from multiple clustering results.
        
        Args:
            results_dict: Dictionary mapping method names to their evaluation results
            
        Returns:
            DataFrame with evaluation results for all methods
        """
        report_data = []
        
        for method_name, metrics in results_dict.items():
            row = {'method': method_name}
            row.update(metrics)
            report_data.append(row)
        
        df = pd.DataFrame(report_data)
        
        # Round numerical columns for better readability
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].round(4)
        
        return df
    
    def rank_methods(self, results_dict: Dict[str, Dict[str, float]], 
                    primary_metric: str = 'silhouette_score') -> pd.DataFrame:
        """
        Rank clustering methods based on evaluation metrics.
        
        Args:
            results_dict: Dictionary mapping method names to their evaluation results
            primary_metric: Primary metric for ranking
            
        Returns:
            DataFrame with ranked methods
        """
        df = self.create_evaluation_report(results_dict)
        
        # Define whether higher or lower is better for each metric
        higher_better = [
            'silhouette_score', 'calinski_harabasz_index', 'adjusted_rand_index',
            'normalized_mutual_info', 'cluster_purity', 'homogeneity', 
            'completeness', 'v_measure'
        ]
        lower_better = ['davies_bouldin_index']
        
        # Calculate ranks for each metric
        rank_columns = []
        for col in df.columns:
            if col in higher_better:
                df[f'{col}_rank'] = df[col].rank(ascending=False)
                rank_columns.append(f'{col}_rank')
            elif col in lower_better:
                df[f'{col}_rank'] = df[col].rank(ascending=True)
                rank_columns.append(f'{col}_rank')
        
        # Calculate average rank
        if rank_columns:
            df['avg_rank'] = df[rank_columns].mean(axis=1)
            df = df.sort_values('avg_rank')
        
        return df


class MetricsComparison:
    """Compare clustering results across different methods and datasets."""
    
    def __init__(self):
        """Initialize metrics comparison."""
        self.results_history = []
    
    def add_experiment(self, experiment_name: str, method_results: Dict[str, Dict[str, float]]):
        """
        Add experiment results for comparison.
        
        Args:
            experiment_name: Name of the experiment
            method_results: Results for different methods in this experiment
        """
        for method_name, metrics in method_results.items():
            result_entry = {
                'experiment': experiment_name,
                'method': method_name,
                **metrics
            }
            self.results_history.append(result_entry)
    
    def get_comparison_dataframe(self) -> pd.DataFrame:
        """
        Get comparison DataFrame with all experiments.
        
        Returns:
            DataFrame with all experimental results
        """
        return pd.DataFrame(self.results_history)
    
    def get_method_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for each method across all experiments.
        
        Returns:
            DataFrame with mean performance for each method
        """
        df = self.get_comparison_dataframe()
        if df.empty:
            return pd.DataFrame()
        
        # Group by method and calculate mean performance
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        summary = df.groupby('method')[numeric_columns].agg(['mean', 'std']).round(4)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns]
        summary = summary.reset_index()
        
        return summary
    
    def plot_comparison(self, metric: str = 'silhouette_score') -> None:
        """
        Plot comparison of methods across experiments.
        
        Args:
            metric: Metric to plot
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = self.get_comparison_dataframe()
        if df.empty or metric not in df.columns:
            print(f"No data available for metric: {metric}")
            return
        
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=df, x='method', y=metric)
        plt.xticks(rotation=45)
        plt.title(f'Comparison of {metric} across methods')
        plt.tight_layout()
        plt.show()


def save_evaluation_results(results: Dict[str, Dict[str, float]], 
                          filename: str = 'clustering_evaluation_results.csv'):
    """
    Save evaluation results to CSV file.
    
    Args:
        results: Dictionary with evaluation results
        filename: Output filename
    """
    evaluator = ClusteringEvaluator()
    df = evaluator.create_evaluation_report(results)
    df.to_csv(filename, index=False)
    print(f"Evaluation results saved to {filename}")


def load_evaluation_results(filename: str) -> pd.DataFrame:
    """
    Load evaluation results from CSV file.
    
    Args:
        filename: Input filename
        
    Returns:
        DataFrame with evaluation results
    """
    return pd.read_csv(filename)