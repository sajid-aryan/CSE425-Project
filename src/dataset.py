import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class AudioDataset(Dataset):
    """Dataset class for audio files."""
    
    def __init__(
        self,
        audio_dir: str,
        sample_rate: int = 22050,
        duration: float = 30.0,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        *,
        max_files: Optional[int] = None,
        cache_dir: Optional[str] = None,
        use_cache: bool = True,
        include_tonnetz: bool = False,
    ):
        """
        Initialize AudioDataset.
        
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate for audio
            duration: Duration to use from each audio file (seconds)
            n_mfcc: Number of MFCC features
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.max_files = max_files
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self.include_tonnetz = include_tonnetz
        
        # Get all audio files
        self.audio_files = []
        self.labels = []
        self.genre_to_idx = {}
        
        for genre_file in os.listdir(audio_dir):
            if genre_file.endswith('.au'):
                # Extract genre from filename (e.g., "blues.00000.au" -> "blues")
                genre = genre_file.split('.')[0]
                if genre not in self.genre_to_idx:
                    self.genre_to_idx[genre] = len(self.genre_to_idx)
                
                self.audio_files.append(os.path.join(audio_dir, genre_file))
                self.labels.append(self.genre_to_idx[genre])

        if self.max_files is not None and self.max_files > 0:
            self.audio_files = self.audio_files[: self.max_files]
            self.labels = self.labels[: self.max_files]
        
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
        print(f"Found {len(self.audio_files)} audio files with {len(self.genre_to_idx)} genres")
        print(f"Genres: {list(self.genre_to_idx.keys())}")

        if self.use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get audio features and label for given index."""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load cached features if available. This avoids re-extracting features
        # every epoch (librosa feature extraction is CPU-bound and expensive).
        if self.use_cache:
            cache_key = os.path.basename(audio_path)
            cache_key = cache_key.replace(os.sep, '_').replace(':', '_')
            cache_name = f"sr{self.sample_rate}_dur{int(self.duration)}_mfcc{self.n_mfcc}_tonnetz{int(self.include_tonnetz)}__{cache_key}.npy"
            cache_path = os.path.join(self.cache_dir, cache_name)
            try:
                if os.path.exists(cache_path):
                    features = np.load(cache_path).astype(np.float32, copy=False)
                    return torch.from_numpy(features), label
            except Exception:
                # Fall back to on-the-fly extraction
                pass
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                   duration=self.duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            # Return zeros if loading fails
            audio = np.zeros(int(self.sample_rate * self.duration))
        
        # Extract features
        features = self.extract_features(audio)

        if self.use_cache:
            try:
                np.save(cache_path, features.astype(np.float32, copy=False))
            except Exception:
                pass
        
        return torch.FloatTensor(features), label
    
    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract audio features from waveform."""
        features = []
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, 
                                   n_mfcc=self.n_mfcc, n_fft=self.n_fft,
                                   hop_length=self.hop_length)
        features.extend([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            np.max(mfccs, axis=1),
            np.min(mfccs, axis=1)
        ])
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
        
        features.extend([
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
        ])
        
        # Chroma features
        try:
            chroma = librosa.feature.chroma_stft(y=audio, sr=self.sample_rate)
            features.append([np.mean(chroma, axis=1)])
        except Exception:
            # 12-dim chroma fallback
            features.append([np.zeros(12, dtype=np.float32)])

        # Tonnetz features (optional; relatively slow)
        if self.include_tonnetz:
            try:
                tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sample_rate)
                features.append([np.mean(tonnetz, axis=1)])
            except Exception:
                # 6-dim tonnetz fallback
                features.append([np.zeros(6, dtype=np.float32)])
        
        # Flatten all features
        flattened_features = []
        for feature_group in features:
            if isinstance(feature_group, list):
                for feature in feature_group:
                    if isinstance(feature, np.ndarray):
                        flattened_features.extend(feature.flatten())
                    else:
                        flattened_features.append(feature)
            else:
                flattened_features.extend(feature_group.flatten())
        
        features_array = np.array(flattened_features)
        
        # Normalize features to prevent gradient explosion
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Z-score normalization
        mean = np.mean(features_array)
        std = np.std(features_array) + 1e-8  # Add small epsilon to prevent division by zero
        features_array = (features_array - mean) / std
        
        return features_array


class SpectrogramDataset(Dataset):
    """Dataset class for spectrogram data."""
    
    def __init__(self, audio_dir: str, sample_rate: int = 22050, 
                 duration: float = 30.0, n_fft: int = 2048, 
                 hop_length: int = 512, n_mels: int = 128):
        """
        Initialize SpectrogramDataset.
        
        Args:
            audio_dir: Directory containing audio files
            sample_rate: Target sample rate for audio
            duration: Duration to use from each audio file (seconds)
            n_fft: FFT window size
            hop_length: Hop length for STFT
            n_mels: Number of mel frequency bins
        """
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        # Get all audio files
        self.audio_files = []
        self.labels = []
        self.genre_to_idx = {}
        
        for genre_file in os.listdir(audio_dir):
            if genre_file.endswith('.au'):
                genre = genre_file.split('.')[0]
                if genre not in self.genre_to_idx:
                    self.genre_to_idx[genre] = len(self.genre_to_idx)
                
                self.audio_files.append(os.path.join(audio_dir, genre_file))
                self.labels.append(self.genre_to_idx[genre])
        
        self.idx_to_genre = {v: k for k, v in self.genre_to_idx.items()}
    
    def __len__(self) -> int:
        return len(self.audio_files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get spectrogram and label for given index."""
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, 
                                   duration=self.duration)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            audio = np.zeros(int(self.sample_rate * self.duration))
        
        # Generate mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate,
                                                n_fft=self.n_fft, hop_length=self.hop_length,
                                                n_mels=self.n_mels)
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
        
        # Add channel dimension for ConvVAE
        log_mel_spec = np.expand_dims(log_mel_spec, axis=0)
        
        return torch.FloatTensor(log_mel_spec), label


class LyricsDataset(Dataset):
    """Dataset class for lyrics data."""
    
    def __init__(self, lyrics_files: List[str], max_length: int = 512):
        """
        Initialize LyricsDataset.
        
        Args:
            lyrics_files: List of CSV files containing lyrics
            max_length: Maximum sequence length for text processing
        """
        self.max_length = max_length
        self.lyrics_data = []
        self.languages = []
        
        for lyrics_file in lyrics_files:
            df = pd.read_csv(lyrics_file)
            
            # Determine language from filename
            if 'bangla' in lyrics_file.lower() or 'bengali' in lyrics_file.lower():
                language = 'bangla'
            elif 'english' in lyrics_file.lower():
                language = 'english'
            else:
                language = 'unknown'
            
            # Process lyrics
            for _, row in df.iterrows():
                lyrics = str(row.get('lyrics', ''))
                if len(lyrics.strip()) > 0:
                    self.lyrics_data.append(lyrics)
                    self.languages.append(language)
        
        print(f"Loaded {len(self.lyrics_data)} lyrics samples")
        print(f"Languages: {set(self.languages)}")
    
    def __len__(self) -> int:
        return len(self.lyrics_data)
    
    def __getitem__(self, idx: int) -> Tuple[str, str]:
        """Get lyrics and language for given index."""
        return self.lyrics_data[idx], self.languages[idx]


class HybridDataset(Dataset):
    """Dataset combining audio and lyrics features."""
    
    def __init__(self, audio_dataset: AudioDataset, lyrics_dataset: LyricsDataset,
                 text_encoder=None):
        """
        Initialize HybridDataset.
        
        Args:
            audio_dataset: AudioDataset instance
            lyrics_dataset: LyricsDataset instance
            text_encoder: Text encoder for lyrics (e.g., from transformers)
        """
        self.audio_dataset = audio_dataset
        self.lyrics_dataset = lyrics_dataset
        self.text_encoder = text_encoder
        
        # Use minimum length for alignment
        self.length = min(len(audio_dataset), len(lyrics_dataset))
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """Get audio features, lyrics embedding, audio label, and lyrics language."""
        audio_features, audio_label = self.audio_dataset[idx % len(self.audio_dataset)]
        lyrics_text, lyrics_language = self.lyrics_dataset[idx % len(self.lyrics_dataset)]
        
        # Simple text embedding (can be replaced with transformer embeddings)
        lyrics_embedding = self.encode_text(lyrics_text)
        
        return audio_features, lyrics_embedding, audio_label, lyrics_language
    
    def encode_text(self, text: str) -> torch.Tensor:
        """Encode text to vector representation."""
        if self.text_encoder is not None:
            # Use transformer-based encoding
            try:
                encoding = self.text_encoder.encode(text)
                return torch.FloatTensor(encoding)
            except:
                pass
        
        # Simple bag-of-words encoding as fallback
        words = text.lower().split()
        vocab_size = 1000  # Fixed vocabulary size
        encoding = np.zeros(vocab_size)
        
        for word in words:
            # Simple hash-based word indexing
            word_idx = hash(word) % vocab_size
            encoding[word_idx] += 1
        
        # Normalize
        if np.sum(encoding) > 0:
            encoding = encoding / np.sum(encoding)
        
        return torch.FloatTensor(encoding)


def create_data_loaders(
    audio_dir: str,
    lyrics_files: List[str],
    batch_size: int = 32,
    test_split: float = 0.2,
    dataset_type: str = 'features',
    *,
    max_files: Optional[int] = None,
    cache_dir: Optional[str] = None,
    include_tonnetz: bool = False,
    duration: float = 10.0,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test data loaders.
    
    Args:
        audio_dir: Directory containing audio files
        lyrics_files: List of lyrics CSV files
        batch_size: Batch size for data loaders
        test_split: Fraction of data for testing
        dataset_type: Type of dataset ('features' or 'spectrogram')
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    if dataset_type == 'spectrogram':
        dataset = SpectrogramDataset(audio_dir)
    else:
        dataset = AudioDataset(
            audio_dir,
            duration=duration,
            max_files=max_files,
            cache_dir=cache_dir,
            include_tonnetz=include_tonnetz,
        )
    
    # Split dataset
    total_size = len(dataset)
    test_size = int(total_size * test_split)
    train_size = total_size - test_size
    
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, test_loader


def load_lyrics_data(lyrics_dir: str) -> pd.DataFrame:
    """
    Load and combine lyrics data from CSV files.
    
    Args:
        lyrics_dir: Directory containing lyrics CSV files
    
    Returns:
        Combined DataFrame with lyrics data
    """
    lyrics_files = []
    for file in os.listdir(lyrics_dir):
        if file.endswith('.csv'):
            lyrics_files.append(os.path.join(lyrics_dir, file))
    
    combined_df = pd.DataFrame()
    
    for file in lyrics_files:
        df = pd.read_csv(file)
        
        # Add language column based on filename
        if 'bangla' in file.lower() or 'bengali' in file.lower():
            df['language'] = 'bangla'
        elif 'english' in file.lower():
            df['language'] = 'english'
        else:
            df['language'] = 'unknown'
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    
    return combined_df


def get_feature_statistics(dataset: Dataset) -> Dict[str, float]:
    """
    Get statistics about dataset features.
    
    Args:
        dataset: Dataset to analyze
    
    Returns:
        Dictionary with feature statistics
    """
    all_features = []
    all_labels = []
    
    for i in range(min(100, len(dataset))):  # Sample first 100 for efficiency
        features, label = dataset[i]
        all_features.append(features.numpy().flatten())
        all_labels.append(label)
    
    all_features = np.array(all_features)
    
    return {
        'num_samples': len(dataset),
        'feature_dim': all_features.shape[1],
        'num_classes': len(set(all_labels)),
        'feature_mean': np.mean(all_features),
        'feature_std': np.std(all_features),
        'feature_min': np.min(all_features),
        'feature_max': np.max(all_features)
    }