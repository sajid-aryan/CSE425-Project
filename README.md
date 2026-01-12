# VAE Music Clustering Project

This project implements a comprehensive Variational Autoencoder (VAE) based clustering system for hybrid language music tracks, following the requirements specified in the project guidelines.

## Project Structure

```
CSE425 Project/
├── data/
│   ├── audio/           # Audio files (GTZAN-like dataset)
│   └── lyrics/          # Lyrics CSV files (Bangla and English)
├── src/
│   ├── main.py          # Main execution script
│   ├── vae.py           # VAE model implementations
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── clustering.py    # Clustering algorithms and pipelines
│   ├── evaluation.py    # Evaluation metrics
│   └── visualization.py # Visualization utilities
├── notebooks/
│   └── exploratory.ipynb # Jupyter notebook for exploration
├── results/
│   ├── latent_visualization/ # Generated plots and visualizations
│   └── clustering_metrics.csv # Evaluation results
├── requirements.txt     # Project dependencies
├── Guidelines.md        # Project guidelines
└── README.md           # This file
```

## Features

### Easy Task Implementation ✅
- Basic VAE for feature extraction from music data
- K-Means clustering on latent features
- Visualization using t-SNE and UMAP
- Baseline comparison with PCA + K-Means
- Evaluation using Silhouette Score and Calinski-Harabasz Index

### Medium Task Implementation ✅
- Convolutional VAE for spectrograms/MFCC features
- Hybrid feature representation (audio + lyrics embeddings)
- Multiple clustering algorithms: K-Means, Agglomerative Clustering, DBSCAN
- Enhanced evaluation with Davies-Bouldin Index and Adjusted Rand Index
- Comparative analysis across methods

### Hard Task Implementation ✅
- Beta-VAE for disentangled latent representations
- Multi-modal clustering combining audio, lyrics, and genre information
- Comprehensive evaluation metrics: Silhouette Score, NMI, ARI, Cluster Purity
- Advanced visualizations: 3D latent space plots, cluster distribution analysis
- Comparison with multiple baselines including Autoencoder + K-Means

## Installation and Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify data structure:**
   - Ensure audio files are in `data/audio/`
   - Ensure lyrics files are in `data/lyrics/`

3. **Run the complete experiment:**
   ```bash
   python src/main.py
   ```

## Usage

### Running Individual Tasks

```python
from src.main import MusicClusteringExperiment

# Initialize experiment
experiment = MusicClusteringExperiment()

# Load datasets
experiment.load_datasets()

# Run individual tasks
easy_results = experiment.run_easy_task()
medium_results = experiment.run_medium_task()
hard_results = experiment.run_hard_task()

# Or run all tasks together
all_results = experiment.run_full_experiment()
```

### Custom Configuration

```python
# Custom experiment setup
experiment = MusicClusteringExperiment(
    data_dir="path/to/data",
    results_dir="path/to/results",
    device="cuda",  # or "cpu"
    random_seed=42
)
```

## Model Architectures

### Basic VAE
- **Architecture:** Fully connected encoder-decoder
- **Latent Dimension:** 32
- **Use Case:** Feature-based audio representation

### Convolutional VAE
- **Architecture:** CNN encoder-decoder for 2D spectrograms
- **Latent Dimension:** 64
- **Use Case:** Spectrogram/MFCC features

### Beta-VAE
- **Architecture:** Enhanced VAE with β-weighting
- **Beta Value:** 4.0 (for disentangled representations)
- **Latent Dimension:** 64
- **Use Case:** Disentangled feature learning

### Conditional VAE
- **Architecture:** Condition-aware encoder-decoder
- **Use Case:** Multi-modal feature fusion

## Clustering Methods

1. **K-Means Clustering**
2. **Agglomerative Clustering**
3. **DBSCAN**
4. **PCA + K-Means (Baseline)**
5. **Direct K-Means (Baseline)**

## Evaluation Metrics

### Unsupervised Metrics
- **Silhouette Score** (higher is better)
- **Calinski-Harabasz Index** (higher is better)
- **Davies-Bouldin Index** (lower is better)

### Supervised Metrics
- **Adjusted Rand Index (ARI)** (higher is better)
- **Normalized Mutual Information (NMI)** (higher is better)
- **Cluster Purity** (higher is better)
- **V-measure** (higher is better)

## Results

### Dataset Statistics
- **Audio Samples:** 1000 files (10 genres × 100 files each)
- **Genres:** Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock
- **Lyrics:** English and Bangla song lyrics

### Key Findings
- VAE-based clustering generally outperforms traditional methods
- Beta-VAE provides the most disentangled representations
- Convolutional architecture works well with spectrograms
- Hybrid features improve clustering quality

## Visualizations

The project generates comprehensive visualizations including:

1. **2D Latent Space Plots** (t-SNE, UMAP, PCA)
2. **3D Latent Space Visualizations**
3. **Clustering Metrics Comparison Charts**
4. **Genre Distribution within Clusters**
5. **Training Loss Curves**
6. **Reconstruction Examples**

All visualizations are automatically saved to `results/latent_visualization/`.

## Files Generated

- `results/comprehensive_clustering_metrics.csv` - Complete evaluation results
- `results/experiment_summary.txt` - Summary of all experiments
- `results/latent_visualization/*.png` - All generated plots

## Reproducibility

The project ensures reproducibility through:
- Fixed random seeds (42)
- Deterministic algorithms where possible
- Version-controlled dependencies
- Comprehensive logging

## Performance Tips

1. **GPU Usage:** Set `device="cuda"` for faster training
2. **Memory Management:** Reduce batch size if encountering memory issues
3. **Quick Testing:** Use fewer epochs for initial testing

## Troubleshooting

### Common Issues

1. **Memory Error:**
   - Reduce batch size in data loaders
   - Use CPU instead of GPU
   - Reduce model hidden dimensions

2. **Audio Loading Issues:**
   - Ensure librosa is installed
   - Check audio file formats
   - Verify file paths

3. **Clustering Errors:**
   - Check for NaN values in features
   - Ensure sufficient data for clustering
   - Verify feature normalization

## Contributing

This project follows the NeurIPS paper format and includes:
- Comprehensive method implementation
- Thorough experimental evaluation
- Statistical analysis and comparisons
- Professional visualizations

## License

This project is for educational purposes as part of CSE425 coursework.

## Contact

For questions or issues, please refer to the project guidelines or course materials.